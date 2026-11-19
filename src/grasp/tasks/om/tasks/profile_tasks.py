"""
The six stages of the KG profile (data_models/profile.py), each a GRASP task on one KG:
    om-profile-space       a.  entity space
    om-profile-partitions  b.  exclusive partitions (kinds) and which do not compete
    om-profile-roles       c.  KG-wide roles and structure relations
    om-profile-groups      d.  overlapping groups, with the task in view; exclusion
    om-profile-group       e.  one group: local names and deviations from the KG-wide roles
    om-profile-revision    f.  open flags resolved or dismissed, findings condensed
"""
import copy
from functools import partial
from typing import Callable

from pydantic import ValidationError

from grasp.tasks.om.data_models.profile import (
    ROLE_FIELDS,
    Finding,
    Flag,
    FlagStatus,
    KgProfile,
    Roles,
    Stage,
    describe_profile,
    role_elements,
    short_iri,
    unknown_name_message,
)
from grasp.tasks.om.foundation.configuration import configured_namespaces
from grasp.tasks.om.foundation.types import IRI, LocalNameType
from grasp.tasks.om.foundation.rdf_vocabulary import (
    UNCATEGORIZED,
    namespace_of,
)
from grasp.tasks.om.kg_operations.grouping import (
    estimated_group_sizes,
)
from grasp.tasks.om.kg_operations.large_kg_sampling import (
    class_counts,
    count_triples,
    is_large,
    large_prefix_counts,
)
from grasp.tasks.om.kg_operations.profile_statistics import (
    canon_roles,
    local_name_type,
    predicate_coverage,
    prefix_counts,
    prefix_totals,
)
from grasp.tasks.om.tasks.profile_stage import (
    NUM_KINDS,
    NUM_NAMESPACES,
    NUM_PREDICATES,
    SPACE_GROUPING,
    EntitySpaceEditor,
    GroupEditMode,
    GroupEditor,
    PartitionEditor,
    ProfileStageTask,
    RoleEditor,
    entity_examples,
    format_flag,
    format_predicate_overview,
    format_values,
    group_name_stats,
    format_share,
    size_stats,
    size_text,
    unassigned_predicates,
)
from grasp.utils import FunctionCallException, format_section


# --- a. entity space ----------------------------------------------------------------
class SpaceTask(ProfileStageTask):
    """
    Stage a: the entity space. prepare counts the entities per IRI prefix (in
    a large KG, estimated from class samples) for the input text; the editor
    gets the samples to inspect prefixes of a large KG.
    """
    name = "om-profile-space"
    stage = Stage.SPACE
    prompt_file = "profile_space.txt"
    space_editor: EntitySpaceEditor
    triples: int | None
    large: bool

    def prepare(self, input: dict) -> str:
        self.triples = count_triples(self.kg_manager)
        self.large = is_large(self.triples)
        prefix_examples: dict[str, list[tuple[IRI, str]]] = {}
        class_prefix_counts: dict[tuple[str, str], int] = {}
        if self.large:
            # a full count per prefix does not finish: estimate it from class samples
            large_counts = large_prefix_counts(self.kg_manager, class_counts(self.kg_manager))
            all_counts, named_totals, totals = large_counts.counts, large_counts.named_totals, large_counts.totals
            prefix_examples, class_prefix_counts = large_counts.examples, all_counts
        else:
            named, all_counts = prefix_counts(self.kg_manager, True), prefix_counts(self.kg_manager, False)
            named_totals, totals = prefix_totals(named), prefix_totals(all_counts)

        # the default space
        configured = configured_namespaces(self.config, self.kg_manager.kg)
        if self.input_profile.entity_space:
            space, source = list(self.input_profile.entity_space), "given in the input profile"
        elif configured:
            space, source = list(configured), "configured"
        elif named_totals:
            space, source = [named_totals.most_common(1)[0][0]], "prefix with the most named entities"
        elif totals:
            space, source = [totals.most_common(1)[0][0]], "prefix with the most entities"
        else:
            space, source = [], "no entities found"
        self.stats = {"triples": self.triples, "large": self.large, "prefixes": dict(totals.most_common()),
                      "named": dict(named_totals.most_common())}
        self.default = self.input_profile.model_copy(
            update={"entity_space": space, "triples": self.triples, "large": self.large}, deep=True)

        # the default is chosen above from the counts made anyway, not by the editor
        self.space_editor = EntitySpaceEditor.from_kg_profile(
            self.kg_manager, self.default, create_missing=False,
            prefix_examples=prefix_examples, class_prefix_counts=class_prefix_counts,
        )

        # the input text: entities per host and prefix, examples
        hosts: dict[str, list[str]] = {}
        for prefix, _ in totals.most_common():
            hosts.setdefault(namespace_of(prefix), []).append(prefix)
        approx = "~" if self.large else ""
        lines = []
        for host, prefixes in sorted(hosts.items(), key=lambda x: -sum(totals[p] for p in x[1]))[:NUM_NAMESPACES]:
            if prefixes != [host]:
                lines.append(f"  {host}: {approx}{sum(totals[p] for p in prefixes)} entities, "
                             f"{approx}{sum(named_totals.get(p, 0) for p in prefixes)} with a canon name")
            for prefix in prefixes[:NUM_NAMESPACES]:
                kinds = sorted(((k, n) for (other, k), n in all_counts.items() if other == prefix), key=lambda x: -x[1])
                more = f", ... {len(kinds) - NUM_KINDS} more" if len(kinds) > NUM_KINDS else ""
                kinds_text = ", ".join(f"{k} {approx}{n}" for k, n in kinds[:NUM_KINDS]) + more
                indent = "    " if prefixes != [host] else "  "
                lines.append(f"{indent}{prefix}: {approx}{totals[prefix]} entities ({kinds_text}), "
                             f"{approx}{named_totals.get(prefix, 0)} with a canon name")
            if len(prefixes) > NUM_NAMESPACES:
                lines.append(f"    ... {len(prefixes) - NUM_NAMESPACES} more prefixes")
        if len(hosts) > NUM_NAMESPACES:
            lines.append(f"  ... {len(hosts) - NUM_NAMESPACES} more hosts")
        examples = []
        for prefix, _ in totals.most_common(3):
            examples.append(f"{prefix}:")
            examples += (self.space_editor.sampled_examples(prefix, 4) if self.large
                         else entity_examples(self.kg_manager, SPACE_GROUPING, [prefix], "", 4))
        size = (f"Size: {self.triples} triples." if self.triples is not None
                else "Size: the triples cannot be counted (the query times out).")
        if self.large:
            size += (" This is a large knowledge graph: entities are counted per class (rdf:type) and "
                     "estimated from samples of each class (~); untyped entities are not counted.")
            group = "class (rdf:type)"
        else:
            group = "kind"
        return "\n\n".join([
            size,
            f"Default entity space ({source}): " + (", ".join(space) or "none"),
            f"Entities per IRI prefix (host, and below it the first path segment) and {group}:\n" + "\n".join(lines),
            "Examples:\n" + "\n".join(examples),
        ])

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.space_editor.functions()

    def refers_to_groups(self) -> bool:
        return False  # partitions and groups come only in later stages

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        space = self.space_editor.checked_entity_space()
        profile = self.input_profile.model_copy(
            update={"entity_space": space, "triples": self.triples, "large": self.large}, deep=True)
        return profile, f"Entity space: {', '.join(space)}"


# --- b. partitions ----------------------------------------------------------------

class PartitionsTask(ProfileStageTask):
    """Stage b1: the exclusive partitions and which of them do not compete."""
    name = "om-profile-partitions"
    stage = Stage.PARTITIONS
    prompt_file = "profile_partitions.txt"
    partition_editor: PartitionEditor

    def prepare(self, input: dict) -> str:
        self.partition_editor = PartitionEditor.from_kg_profile(
            self.kg_manager, self.input_profile, create_missing=True
            )
        self.stats = self.partition_editor.size_stats()
        self.default = self.partition_editor.with_partitions(self.input_profile)

        sections = [self.partition_editor.input_for_llm()]
        for name in self.partition_editor.names() + [UNCATEGORIZED]:
            if self.partition_editor.sizes.get(name):
                title = (f"Partition {name}" if name != UNCATEGORIZED
                         else f"Subjects that match no partition, {UNCATEGORIZED}, not aligned")
                sections.append(self.partition_editor.describe_member(name, title))
        return "\n\n".join(sections)

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.partition_editor.functions()

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        self.partition_editor.check_partitions()
        profile = self.partition_editor.with_partitions(self.input_profile)
        self.stats = self.partition_editor.size_stats()
        summary = "Partitions: " + ", ".join(f"{name} {self.stats['sizes'].get(name, 0)}"
                                             for name in profile.partition_names())
        dropped = self.partition_editor.empty_partition_names()
        if dropped:
            summary += f"; dropped as empty: {', '.join(dropped)}"
        return profile, summary


# --- c. KG-wide roles ----------------------------------------------------------------

class RolesTask(ProfileStageTask):
    """
    Stage c: the KG-wide roles. The task holds only its editor; what it
    computes for its input text (sizes, coverage, the default) stays local
    to prepare, the editor keeps what it needs for its functions.
    """
    name = "om-profile-roles"
    stage = Stage.ROLES
    prompt_file = "profile_roles.txt"
    role_editor: RoleEditor

    def prepare(self, input: dict) -> str:
        self.role_editor = RoleEditor.from_kg_profile(self.kg_manager, self.input_profile, create_missing=True)
        default = self.role_editor.roles
        entity_space, grouping = self.role_editor.entity_space, self.role_editor.grouping
        partition_names, sizes = self.role_editor.scope, self.role_editor.sizes
        estimates = estimated_group_sizes(self.kg_manager, grouping, entity_space)
        self.default = self.input_profile.model_copy(update={"entity_space": entity_space, "roles": default}, deep=True)
        self.stats = size_stats(self.kg_manager, sizes, estimates)

        coverage = predicate_coverage(self.kg_manager, grouping, entity_space)
        canon = ", ".join(short_iri(key[2]) if key[1] == "predicate" else key[2]
                          for key in role_elements(canon_roles()))
        partition_sizes = ", ".join(
            f"{name} ({size_text(self.kg_manager, sizes, estimates, name)})" for name in partition_names
            )
        return "\n\n".join([
            "Partitions: " + partition_sizes,
            f"Default canon checked: {canon}. Canon elements below "
            "1% coverage in every partition are left out of the default.",
            "Default roles (coverage as ?s per partition):\n" + self.role_editor.input_for_llm(),
            "Most frequent predicates: share of each partition's entities that are their subject "
            "(as ?s), kinds of their values (?o), [role] if in the default:\n"
            + format_predicate_overview(coverage, sizes, partition_names, default, NUM_PREDICATES),
        ])

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.role_editor.functions()

    def stage_rules(self) -> list[str]:
        return ["Test every predicate or pattern you add with try_pattern first."]

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        roles = self.role_editor.checked_roles()
        profile = self.default.model_copy(update={"roles": roles}, deep=True)
        changed = [f for f in (*ROLE_FIELDS, "structure") if getattr(roles, f) != getattr(self.default.roles, f)]
        return profile, "Changed: " + (", ".join(changed) or "nothing (default roles)")


# --- d. groups ----------------------------------------------------------------------

class GroupsTask(ProfileStageTask):
    """Stage b2: the overlapping groups and their exclusion. Input besides kg
    and profile: partner_profile, the profile of the other KG of the task
    (optional; shown to the model if given)."""
    name = "om-profile-groups"
    stage = Stage.GROUPS
    prompt_file = "profile_groups.txt"
    group_editor: GroupEditor

    def prepare(self, input: dict) -> str:
        self.group_editor = GroupEditor.from_kg_profile(self.kg_manager, self.input_profile, create_missing=True)
        self.stats = self.group_editor.size_stats()
        self.default = self.group_editor.with_groups(self.input_profile)

        sections = []
        partner = input.get("partner_profile")
        if partner:
            notes = describe_profile(KgProfile.model_validate(partner))
            sections.append(format_section("Profile of the other knowledge graph of the task",
                                           "\n".join(f"- {note}" for note in notes)))
        sections.append(self.group_editor.input_for_llm())
        for name in self.group_editor.names():
            if self.group_editor.sizes.get(name):
                sections.append(self.group_editor.describe_member(name, f"Group {name}"))
        return "\n\n".join(sections)

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.group_editor.functions()

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        self.group_editor.check_groups()
        profile = self.group_editor.with_groups(self.input_profile)
        self.stats = self.group_editor.size_stats()
        summary = "Groups: " + ", ".join(
            f"{group.name} {self.stats['sizes'].get(group.name, 0)}" + (" (excluded)" if group.excluded else "")
            for group in profile.groups
        )
        dropped = self.group_editor.empty_group_names()
        if dropped:
            summary += f"; dropped as empty: {', '.join(dropped)}"
        return profile, summary


# --- e. one group --------------------------------------------------------------------

class GroupTask(ProfileStageTask):
    """Stage d: one group, its local name type and its deviations from the
    KG-wide roles. Input besides kg and profile: group, mode (full or light)."""
    name = "om-profile-group"
    stage = Stage.GROUP
    prompt_files = {
        GroupEditMode.FULL: "profile_group.txt",
        GroupEditMode.LIGHT: "profile_group_light.txt"
        }
    prompt_file = prompt_files[GroupEditMode.FULL]  # set by prepare for the mode
    group_name: str
    edit_mode: GroupEditMode
    role_editor: RoleEditor
    name_type: LocalNameType  # the default local name type

    def prepare(self, input: dict) -> str:
        profile = self.input_profile
        group_name = input.get("group")
        if group_name is None or group_name not in profile.group_names():
            raise ValueError(f"Unknown group {group_name}; groups: {', '.join(profile.group_names())}")
        self.group_name = group_name
        self.edit_mode = GroupEditMode(input.get("mode") or GroupEditMode.FULL)
        self.prompt_file = self.prompt_files[self.edit_mode]

        self.role_editor = RoleEditor.from_group_profile(self.kg_manager, profile, group_name)
        effective = self.role_editor.roles
        entity_space, grouping, sizes = self.role_editor.entity_space, self.role_editor.grouping, self.role_editor.sizes
        estimates = estimated_group_sizes(self.kg_manager, grouping, entity_space)
        group_profile = profile.get_group(group_name)

        stats = group_name_stats(self.kg_manager, grouping, entity_space, group_name, effective)
        self.name_type = local_name_type(stats)
        self.stats = {"size": size_stats(self.kg_manager, sizes, estimates)["sizes"].get(group_name, 0),
                      "mode": self.edit_mode.value, "local_names": stats.model_dump()}
        self.default = profile.with_replaced_group(group_profile.model_copy(update={"local_name_type": self.name_type}))

        derived = self.role_editor.derived_roles or Roles()
        unused = [key for key in role_elements(profile.roles) if key not in role_elements(derived)
                  and key not in role_elements(group_profile.added)]
        coverage = predicate_coverage(self.kg_manager, grouping, entity_space).get(group_name, {})
        size = sizes.get(group_name, 0)
        others = [f"  {short_iri(p)}  as ?s: {format_share(c.entities, size)}  |  values: {format_values(c.values, c.literals, c.blanks)}"
                  for p, c in unassigned_predicates(effective, coverage)][:NUM_PREDICATES]
        definition = group_profile.definition
        agreement = "–" if stats.agreement is None else f"{stats.agreement:.0%}"
        speaking = "–" if stats.speaking is None else f"{stats.speaking:.0%}"
        return "\n\n".join([
            f"Group {group_name}: {size_text(self.kg_manager, sizes, estimates, group_name)}"
            + (f"; {definition.features}" if definition.features else "")
            + f"; membership {{ {definition.membership_sparql} }}",
            f"Local names (sample of {stats.sample_size}, {stats.labelled} with a name): {agreement} agree with "
            f"a name, {speaking} look like words -> default local_name_type: {self.name_type.value}",
            "Roles of this group: the KG-wide roles used here [kg], and its own additions [group] "
            "(coverage as ?s):\n" + self.role_editor.input_for_llm(),
            "KG-wide elements not used in this group (dropped): "
            + (", ".join(short_iri(k[2]) for k in unused) or "none"),
            "Predicates of this group's entities that have no role yet: share of its entities that "
            "are their subject (as ?s), kinds of their values (?o):\n" + ("\n".join(others) or "  none"),
        ])

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.role_editor.functions()

    def finish_properties(self) -> dict[str, dict]:
        return {"local_name_type": {
            "type": "string", "enum": [t.value for t in LocalNameType],
            "description": "If the default is wrong: label if local names are names, code if identifiers, other if mixed.",
        }}

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        self.role_editor.check_roles()
        added, removed = self.role_editor.group_deviation()
        name_type = LocalNameType(fn_args.get("local_name_type") or self.name_type)
        current = self.default.get_group(self.group_name)
        updated = current.model_copy(update={"local_name_type": name_type, "added": added, "removed": removed})
        profile = self.default.with_replaced_group(updated)
        count = lambda r: len(role_elements(r))
        return profile, (f"Group {self.group_name}: local names {name_type.value}, "
                         f"{count(added)} elements added or overridden, {count(removed)} removed")


# --- f. revision: resolve the flags, condense the findings ------------------------------

# functions whose successful call makes dependent editors outdated
SPACE_CHANGES = {"set_entity_space"}
PARTITION_CHANGES = {"set_partition", "remove_partition"}
GROUP_CHANGES = {"set_group", "exclude_group", "remove_group"}
KG_ROLE_CHANGES = {"edit_role", "set_structure_relation", "remove_structure_relation"}
FLAG_RESOLUTIONS = (FlagStatus.RESOLVED, FlagStatus.DISMISSED)
GROUP_PARAMETER = {"type": "string", "description": (
    "The group whose roles to test or change (its own deviation from the KG-wide roles); "
    "omit it for the KG-wide roles.")}


class ProfileRevisionTask(ProfileStageTask):
    """
    Revision after the profile stages: every open flag is resolved (by a
    change of the profile) or dismissed, each with a resolution, and the
    findings are condensed. Every part of the profile can be changed with
    the functions of the stages; the input profile must be complete.

    The editors depend on each other: the entity space on nothing, the
    partitions, groups and KG-wide roles on the space, the KG-wide roles on
    the partitions, the roles of a group on the groups and the KG-wide roles.
    After a change, the task rebuilds the dependent editors from the current
    profile (current_profile: the working states of all editors put
    together). A group's role editor is created on first use; rebuilt, it
    keeps its deviation (added, removed) and derives the rest from the
    current KG-wide roles again (a rebase).

    In a large KG, the entity space cannot be changed: its class samples
    are not in the profile.
    """
    name = "om-profile-revision"
    stage = Stage.REVISION
    prompt_file = "profile_revision.txt"
    entity_space_editor: EntitySpaceEditor
    partition_editor: PartitionEditor
    group_editor: GroupEditor
    partition_role_editor: RoleEditor             # the KG-wide roles, coverage per partition
    group_role_editors: dict[str, RoleEditor]     # group name -> its roles; created on first use
    flags: list[Flag]                             # working copies, with the resolutions
    findings: list[Finding]                       # working copy, replaced by set_findings

    def prepare(self, input: dict) -> str:
        profile = self.input_profile
        try:
            self.entity_space_editor = EntitySpaceEditor.from_kg_profile(
                self.kg_manager, profile, create_missing=False
                )
            self.partition_editor = PartitionEditor.from_kg_profile(self.kg_manager, profile, create_missing=False)
            self.group_editor = GroupEditor.from_kg_profile(self.kg_manager, profile, create_missing=False)
            self.partition_role_editor = RoleEditor.from_kg_profile(self.kg_manager, profile, create_missing=False)
        except ValueError as e:
            raise ValueError(f"The revision needs a complete profile: {e}") from e
        self.group_role_editors = {}
        self.flags = [flag.model_copy() for flag in profile.flags]
        self.findings = list(profile.findings)
        # without finish, the profile stays as it is, the flags stay open
        self.default = profile.model_copy(deep=True)
        self.stats = {"open_flags": len(self.open_flag_ids()), "findings": len(self.findings)}

        sections = [
            self.entity_space_editor.input_for_llm()
            + (" (a large knowledge graph: the entity space cannot be changed here)" if profile.large else ""),
            self.partition_editor.input_for_llm(),
            self.group_editor.input_for_llm(),
            "KG-wide roles (coverage as ?s per partition):\n" + self.partition_role_editor.input_for_llm(),
            "The roles of a group: show_roles with the group.",
        ]
        open_flags = [format_flag(flag) for flag in self.flags if flag.status == FlagStatus.OPEN]
        sections.append(format_section("Open flags", "\n".join(open_flags) or "none"))
        return "\n\n".join(sections)

    # --- the current profile and the rebuild of dependent editors ---

    def current_profile(self) -> KgProfile:
        """The profile from the working states of all editors, unchecked; partitions
        and groups without entities are left out."""
        profile = self.partition_editor.with_partitions(self.input_profile)
        profile = self.group_editor.with_groups(profile)
        groups = []
        for group in profile.groups:
            role_editor = self.group_role_editors.get(group.name)
            if role_editor is not None:
                added, removed = role_editor.group_deviation()
                group = group.model_copy(update={"added": added, "removed": removed})
            groups.append(group)
        return profile.model_copy(update={
            "entity_space": list(self.entity_space_editor.entity_space),
            "roles": self.partition_role_editor.roles.model_copy(deep=True),
            "groups": groups,
            "flags": [flag.model_copy() for flag in self.flags],
            "findings": list(self.findings),
        })

    def rebuild_after_space_change(self) -> None:
        """Every other editor counts on the entity space: all are rebuilt."""
        profile = self.current_profile()
        self.partition_editor = PartitionEditor.from_kg_profile(self.kg_manager, profile, create_missing=False)
        self.group_editor = GroupEditor.from_kg_profile(self.kg_manager, profile, create_missing=False)
        self.partition_role_editor = RoleEditor.from_kg_profile(self.kg_manager, profile, create_missing=False)
        self.rebuild_group_role_editors(profile)

    def rebuild_partition_role_editor(self) -> None:
        """The KG-wide roles show their coverage per partition."""
        self.partition_role_editor = RoleEditor.from_kg_profile(
            self.kg_manager, self.current_profile(), create_missing=False
            )

    def rebuild_group_role_editors(self, profile: KgProfile | None = None) -> None:
        """The roles of the groups (rebase); editors of groups that are gone are dropped."""
        profile = profile or self.current_profile()
        self.group_role_editors = {
            group_name: RoleEditor.from_group_profile(self.kg_manager, profile, group_name)
            for group_name in self.group_role_editors if group_name in profile.group_names()
            }

    def rebuilding(
            self,
            functions: list[tuple[dict, Callable[[dict], str]]],
            changing: set[str],
            rebuild: Callable[[], None]
            ) -> list[tuple[dict, Callable[[dict], str]]]:
        """The functions; those named in changing rebuild the dependent editors after a successful call."""
        def with_rebuild(handler: Callable[[dict], str]) -> Callable[[dict], str]:
            def handle(fn_args: dict) -> str:
                result = handler(fn_args)
                try:
                    rebuild()
                except ValueError as e:
                    raise FunctionCallException(f"Changed, but the dependent parts could not be updated: {e}") from e
                return result
            return handle
        return [(definition, with_rebuild(handler) if definition["name"] in changing else handler)
                for definition, handler in functions]

    # --- the roles: one set of functions for the KG-wide roles and those of each group ---

    def group_role_editor(self, group_name: str) -> RoleEditor:
        """The role editor of a group, created on first use from the current profile."""
        if group_name not in self.group_role_editors:
            profile = self.current_profile()
            if group_name not in profile.group_names():
                raise FunctionCallException(unknown_name_message("group", group_name, profile.group_names()))
            self.group_role_editors[group_name] = RoleEditor.from_group_profile(self.kg_manager, profile, group_name)
        return self.group_role_editors[group_name]

    def role_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        """The functions of the role editors, with a group parameter that chooses the editor."""
        functions = []
        for definition, _ in self.partition_role_editor.functions():
            definition = copy.deepcopy(definition)
            properties = definition["parameters"]["properties"]
            properties["group"] = GROUP_PARAMETER
            if "show_examples_from" in properties:
                properties["show_examples_from"] = {"type": "string", "description": (
                    "KG-wide roles: the partition to show example matches from; default: the partition with "
                    "the most entities as ?s. Not used for a group.")}
            functions.append((definition, partial(self.call_role_function, definition["name"])))
        functions.append(({
            "name": "show_roles",
            "description": "Shows the roles with their coverage: KG-wide, or of a group ([kg]: KG-wide, [group]: its own).",
            "parameters": {
                "type": "object",
                "properties": {"group": GROUP_PARAMETER},
                "additionalProperties": False,
            },
        }, self.show_roles))
        return functions

    def call_role_function(self, function_name: str, fn_args: dict) -> str:
        group_name = fn_args.get("group")
        arguments = {key: value for key, value in fn_args.items() if key != "group"}
        if group_name is None:
            handlers = dict((d["name"], h) for d, h in self.partition_role_editor.functions())
            handler = handlers[function_name]
            if function_name in KG_ROLE_CHANGES:
                return self.rebuilding([({"name": function_name}, handler)], KG_ROLE_CHANGES,
                                       self.rebuild_group_role_editors)[0][1](arguments)
            return handler(arguments)
        arguments.pop("show_examples_from", None)
        role_editor = self.group_role_editor(group_name)
        handlers = dict((d["name"], h) for d, h in role_editor.functions())
        return handlers[function_name](arguments)

    def show_roles(self, fn_args: dict) -> str:
        group_name = fn_args.get("group")
        if group_name is None:
            return "KG-wide roles (coverage as ?s per partition):\n" + self.partition_role_editor.input_for_llm()
        return (f"Roles of group {group_name} (coverage as ?s):\n"
                + self.group_role_editor(group_name).input_for_llm())

    # --- flags and findings ---

    def open_flag_ids(self) -> list[str]:
        return [flag.id for flag in self.flags if flag.status == FlagStatus.OPEN]

    def flag_and_finding_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            ({
                "name": "resolve_flag",
                "description": (
                    "Closes a flag: resolved if it was right and you changed the profile, dismissed if it was "
                    "wrong. The resolution says what you changed or checked, and why."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string", "enum": [status.value for status in FLAG_RESOLUTIONS]},
                        "resolution": {"type": "string"},
                    },
                    "required": ["id", "status", "resolution"],
                    "additionalProperties": False,
                },
            }, self.resolve_flag),
            ({
                "name": "set_findings",
                "description": (
                    "Replaces all findings by a condensed list: merge repetitions, drop findings that are "
                    "wrong or no longer apply, correct the rest, keep what matters for matching."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "group": {"type": "string", "description": self.about_description()},
                            },
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    }},
                    "required": ["findings"],
                    "additionalProperties": False,
                },
            }, self.set_findings),
        ]

    def resolve_flag(self, fn_args: dict) -> str:
        flag_id = fn_args.get("id")
        flag = next((flag for flag in self.flags if flag.id == flag_id), None)
        if flag is None:
            raise FunctionCallException(f"No flag {flag_id}; flags: {', '.join(f.id for f in self.flags) or 'none'}")
        try:
            status = FlagStatus(fn_args.get("status"))
        except ValueError as e:
            raise FunctionCallException(f"Invalid status: {e}") from e
        if status not in FLAG_RESOLUTIONS:
            raise FunctionCallException("A flag is closed as resolved or dismissed")
        resolution = fn_args.get("resolution") or ""
        if not resolution:
            raise FunctionCallException("Give the resolution: what you changed or checked, and why")
        flag.status, flag.resolution_description = status, resolution
        return f"Flag {flag.id} {status.value}. Open flags: {', '.join(self.open_flag_ids()) or 'none'}"

    def set_findings(self, fn_args: dict) -> str:
        profile = self.current_profile()
        known = set(profile.partition_names()) | set(profile.group_names())
        try:
            findings = [Finding(stage=self.stage, **finding) for finding in fn_args.get("findings") or []]
        except (ValidationError, TypeError) as e:
            raise FunctionCallException(f"Invalid findings: {e}") from e
        unknown = {finding.group for finding in findings if finding.group and finding.group not in known}
        if unknown:
            raise FunctionCallException(f"Unknown groups or partitions {sorted(unknown)}; known: {', '.join(sorted(known))}")
        before = len(self.findings)
        self.findings = findings
        return f"Findings replaced: {before} -> {len(findings)}"

    # --- stage interface ---

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        functions = []
        if not self.input_profile.large:
            functions += self.rebuilding(self.entity_space_editor.functions(), SPACE_CHANGES,
                                         self.rebuild_after_space_change)
        functions += self.rebuilding(self.partition_editor.functions(), PARTITION_CHANGES,
                                     self.rebuild_partition_role_editor)
        functions += self.rebuilding(self.group_editor.functions(), GROUP_CHANGES, self.rebuild_group_role_editors)
        functions += self.group_editor.local_name_type_functions()
        functions += self.role_functions()
        functions += self.flag_and_finding_functions()
        return functions

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        self.entity_space_editor.checked_entity_space()
        self.partition_editor.check_partitions()
        self.group_editor.check_groups()
        self.partition_role_editor.check_roles()
        for group_name, role_editor in self.group_role_editors.items():
            try:
                role_editor.check_roles()
            except FunctionCallException as e:
                raise FunctionCallException(f"Group {group_name}: {e}") from e
        profile = self.current_profile()
        counts = {status: sum(flag.status == status for flag in self.flags) for status in FlagStatus}
        summary = (f"Flags: {counts[FlagStatus.RESOLVED]} resolved, {counts[FlagStatus.DISMISSED]} dismissed, "
                   f"{counts[FlagStatus.OPEN]} open ({', '.join(self.open_flag_ids()) or 'none'}); "
                   f"findings: {self.stats['findings']} -> {len(self.findings)}")
        return profile, summary


PROFILE_TASKS = [SpaceTask, PartitionsTask, RolesTask, GroupsTask, GroupTask, ProfileRevisionTask]
