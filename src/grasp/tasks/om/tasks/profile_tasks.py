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
from collections import Counter
from functools import partial
from typing import Callable, Literal


from grasp.tasks.om.data_models.profile import (
    ROLE_FIELDS,
    Flag,
    FlagStatus,
    KgProfile,
    PredicateRoles,
    Stage,
    describe_profile,
    RoleElementKind,
    role_element_order,
    role_elements_of,
    short_iri,
    unknown_name_message,
)
from grasp.tasks.om.foundation.configuration import configured_namespaces
from grasp.tasks.om.kg_operations.grouping import retrieve_grouping_by_entity
from grasp.tasks.om.foundation.rendering import Item, Section, SectionContent, Table, format_display_content
from grasp.tasks.om.foundation.types import LocalNameType
from grasp.tasks.om.foundation.rdf_vocabulary import (
    UNCATEGORIZED,
    namespace_of,
)
from grasp.tasks.om.kg_operations.profile_statistics import (
    LARGE_TRIPLES,
    count_triples,
    is_large,
    canon_roles,
    local_name_type,
    predicate_coverage,
    group_name_stats,
    count_entities_by_prefix,
    count_kinds_by_prefix,
    retrieve_named_entities,
)
from grasp.tasks.om.tasks.profile_stage import (
    MAX_FINDINGS_AFTER_REVISION,
    NUM_MEMBERS_LISTED_PER_PREFIX,
    NUM_NAMESPACES,
    NUM_PREDICATES,
    DEFAULT_SPACE_PARTITION_GROUPING,
    EntitySpaceEditor,
    GroupEditMode,
    GroupEditor,
    PartitionEditor,
    ProfileFunctionName,
    ProfileStageTask,
    RoleEditor,
    get_entity_examples,
    format_predicate_overview,
    format_value_type_shares,
    format_share,
    get_required_argument,
    size_text,
    unassigned_predicates,
)
from grasp.utils import FunctionCallException


# --- a. entity space ----------------------------------------------------------------
class SpaceTask(ProfileStageTask):
    """
    Stage a: the entity space. prepare counts the
    entities per IRI prefix for the input text.
    """
    name = "om-profile-space"
    stage = Stage.SPACE
    prompt_file = "profile_space.txt"
    space_editor: EntitySpaceEditor
    triples: int | None

    def prepare(self, input: dict) -> Section:
        self.triples = count_triples(self.kg_manager)
        if is_large(self.triples):
            raise ValueError(
                f"Large knowledge graphs (more than {LARGE_TRIPLES:_} triples, or whose triples cannot "
                "be counted) are not supported yet."
                )
        # the entities of the default partitions (kinds), with the rest
        kinds_by_entity = retrieve_grouping_by_entity(self.kg_manager, None, ())
        kinds_by_prefix = count_kinds_by_prefix(kinds_by_entity)
        prefix_counts_all_entities = count_entities_by_prefix(kinds_by_entity)
        prefix_counts_named_entities = count_entities_by_prefix(
            retrieve_named_entities(self.kg_manager) & kinds_by_entity.keys()
            )

        # the default space
        configured = configured_namespaces(self.config, self.kg_manager.kg)
        if self.input_profile.entity_space:
            space, source = list(self.input_profile.entity_space), "given in the input profile"
        elif configured:
            space, source = list(configured), "configured"
        elif prefix_counts_named_entities:
            space, source = [prefix_counts_named_entities.most_common(1)[0][0]], "prefix with the most named entities"
        elif prefix_counts_all_entities:
            space, source = [prefix_counts_all_entities.most_common(1)[0][0]], "prefix with the most entities"
        else:
            space, source = [], "no entities found"
        self.stats = {
            "triples": self.triples,
            "prefixes": dict(prefix_counts_all_entities.most_common()),
            "named": dict(prefix_counts_named_entities.most_common())
            }
        self.default_profile = self.input_profile.model_copy(
            update={"entity_space": space, "triples": self.triples}, deep=True)

        # the default is chosen above from the counts, not by the editor
        self.space_editor = EntitySpaceEditor.from_kg_profile(
            self.kg_manager, self.default_profile, create_missing=False,
        )

        # the input: entities per host and prefix, examples
        hosts: dict[str, list[str]] = {}
        for prefix, _ in prefix_counts_all_entities.most_common():
            hosts.setdefault(namespace_of(prefix), []).append(prefix)

        def prefix_item(prefix: str) -> Item:
            members = kinds_by_prefix.get(prefix, Counter()).most_common()
            more_text = (
                f", ... {len(members) - NUM_MEMBERS_LISTED_PER_PREFIX} more"
                if len(members) > NUM_MEMBERS_LISTED_PER_PREFIX else ""
            )
            members_text = ", ".join(
                f"{member}: {n}" for member, n in members[:NUM_MEMBERS_LISTED_PER_PREFIX]
                ) + more_text
            return Item(
                f"{prefix}: {prefix_counts_all_entities[prefix]} entities ({members_text}), "
                f"{prefix_counts_named_entities.get(prefix, 0)} with a canon name"
                )

        prefix_items: list[SectionContent] = []
        for host, prefixes in sorted(hosts.items(), key=lambda x: -sum(prefix_counts_all_entities[p] for p in x[1]))[:NUM_NAMESPACES]:
            if prefixes == [host]:
                prefix_items.append(prefix_item(host))
                continue
            children = [prefix_item(prefix) for prefix in prefixes[:NUM_NAMESPACES]]
            if len(prefixes) > NUM_NAMESPACES:
                children.append(Item(f"... {len(prefixes) - NUM_NAMESPACES} more prefixes"))
            prefix_items.append(Item(
                f"{host}: {sum(prefix_counts_all_entities[p] for p in prefixes)} entities, "
                f"{sum(prefix_counts_named_entities.get(p, 0) for p in prefixes)} with a canon name", children))
        if len(hosts) > NUM_NAMESPACES:
            prefix_items.append(Item(f"... {len(hosts) - NUM_NAMESPACES} more hosts"))
        example_items: list[SectionContent] = [
            Item(
                f"{prefix}:",
                get_entity_examples(
                    self.kg_manager, DEFAULT_SPACE_PARTITION_GROUPING, [prefix],
                    self.space_editor.names_pattern(), None, 4
                    )
                )
            for prefix, _ in prefix_counts_all_entities.most_common(3)
            ]
        size = f"Size: {self.triples} triples."

        return Section("Current stage: entity space", [
            size,
            f"Default entity space ({source}): " + (", ".join(space) or "none"),
            ], [
            Section(
                "Entities per IRI prefix (host, and below it the first path segment) and kind",
                prefix_items
                ),
            Section("Examples", example_items),
            ])

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.space_editor.functions()

    def refers_to_groups(self) -> bool:
        return False  # partitions and groups come only in later stages

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        space = self.space_editor.checked_entity_space()
        profile = self.input_profile.model_copy(
            update={"entity_space": space, "triples": self.triples}, deep=True)
        return profile, f"Entity space: {', '.join(space)}"


# --- b. partitions ----------------------------------------------------------------

class PartitionsTask(ProfileStageTask):
    """Stage b1: the exclusive partitions and which of them do not compete."""
    name = "om-profile-partitions"
    stage = Stage.PARTITIONS
    prompt_file = "profile_partitions.txt"
    partition_editor: PartitionEditor

    def prepare(self, input: dict) -> Section:
        self.partition_editor = PartitionEditor.from_kg_profile(
            self.kg_manager, self.input_profile, create_missing=True
            )
        self.stats = self.partition_editor.size_stats()
        self.default_profile = self.partition_editor.with_partitions(self.input_profile)

        subsections = list(self.partition_editor.input_for_llm())
        for name in self.partition_editor.member_names() + [UNCATEGORIZED]:
            if self.partition_editor.group_sizes.get(name):
                title = (f"Partition {name}" if name != UNCATEGORIZED
                         else f"Subjects that match no partition, {UNCATEGORIZED}, not aligned")
                subsections.append(self.partition_editor.describe_member(name, title))
        return Section("Current stage: partitions", subsections=subsections)

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

    def prepare(self, input: dict) -> Section:
        self.role_editor = RoleEditor.from_kg_profile(self.kg_manager, self.input_profile, create_missing=True)
        default = self.role_editor.roles
        entity_space, grouping = self.role_editor.entity_space, self.role_editor.grouping
        partition_names, sizes = self.role_editor.scope, self.role_editor.unit_sizes
        self.default_profile = self.input_profile.model_copy(update={"entity_space": entity_space, "roles": default}, deep=True)
        self.stats = {"sizes": sizes}

        coverage = predicate_coverage(self.kg_manager, grouping, entity_space)
        canon = ", ".join(
            short_iri(element.definition) if element.kind == RoleElementKind.PREDICATE
            else str(element.definition) if element.kind == RoleElementKind.PATTERN
            else f"relation {element.role_name}"
            for element in sorted(role_elements_of(canon_roles()), key=role_element_order)
            if not (element.is_structure_relation and element.kind != RoleElementKind.RELATION_PROPERTIES)
            )
        partition_sizes = ", ".join(
            f"{name} ({size_text(sizes, name)})" for name in partition_names
            )
        return Section("Current stage: KG-wide roles", [
            "Partitions: " + partition_sizes,
            f"Default canon checked: {canon}. Canon elements below "
            "1% coverage in every partition are left out of the default. "
            "The default roles are below.",
            ], [
            *self.role_editor.input_for_llm(),
            Section("Most frequent predicates", [
                "Share of each partition's entities that are their subject (as ?s), kinds of their "
                "values (?o), and the role or structure relation they are in:",
                format_predicate_overview(coverage, sizes, partition_names, default, NUM_PREDICATES),
                ]),
            ])

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.role_editor.functions()

    def stage_rules(self) -> list[str]:
        return [f"Test every predicate or pattern you add with {ProfileFunctionName.TRY_PATTERN} first."]

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        roles = self.role_editor.checked_roles()
        profile = self.default_profile.model_copy(update={"roles": roles}, deep=True)
        changed = [f for f in (*ROLE_FIELDS, "structure") if getattr(roles, f) != getattr(self.default_profile.roles, f)]
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

    def prepare(self, input: dict) -> Section:
        self.group_editor = GroupEditor.from_kg_profile(self.kg_manager, self.input_profile, create_missing=True)
        self.stats = self.group_editor.size_stats()
        self.default_profile = self.group_editor.with_groups(self.input_profile)

        subsections = []
        partner = input.get("partner_profile")
        if partner:
            notes = describe_profile(KgProfile.model_validate(partner))
            subsections.append(
                Section("Profile of the other knowledge graph of the task", notes.content, notes.subsections))
        subsections += self.group_editor.input_for_llm()
        for name in self.group_editor.member_names():
            if self.group_editor.group_sizes.get(name):
                subsections.append(self.group_editor.describe_member(name, f"Group {name}"))
        return Section("Current stage: groups", subsections=subsections)

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

    def id_prefix(self) -> str:
        return f"{self.stage}-{self.group_name}"

    def prepare(self, input: dict) -> Section:
        profile = self.input_profile
        group_name = input.get("group")
        if group_name is None or group_name not in profile.group_names():
            raise ValueError(f"Unknown group {group_name}; groups: {', '.join(profile.group_names())}")
        self.group_name = group_name
        self.edit_mode = GroupEditMode(input.get("mode") or GroupEditMode.FULL)
        self.prompt_file = self.prompt_files[self.edit_mode]

        self.role_editor = RoleEditor.from_group_profile(self.kg_manager, profile, group_name)
        effective = self.role_editor.roles
        entity_space, grouping, sizes = self.role_editor.entity_space, self.role_editor.grouping, self.role_editor.unit_sizes
        group_profile = profile.get_group(group_name)

        stats = group_name_stats(self.kg_manager, grouping, entity_space, group_name, effective)
        self.name_type = local_name_type(stats)
        self.stats = {"size": sizes.get(group_name, 0),
                      "mode": self.edit_mode.value, "local_names": stats.model_dump()}
        self.default_profile = profile.with_replaced_group(group_profile.model_copy(update={"local_name_type": self.name_type}))

        derived = self.role_editor.derived_roles or PredicateRoles()
        derived_elements, added = role_elements_of(derived), set(group_profile.added)
        unused = sorted(
            (element for element in role_elements_of(profile.roles)
             if element not in derived_elements and element not in added
             and element.kind != RoleElementKind.RELATION_PROPERTIES),
            key=role_element_order,
            )
        coverage = predicate_coverage(self.kg_manager, grouping, entity_space).get(group_name, {})
        size = sizes.get(group_name, 0)
        unassigned = unassigned_predicates(effective, coverage)[:NUM_PREDICATES]
        definition = group_profile.definition
        agreement = "–" if stats.agreement is None else f"{stats.agreement:.0%}"
        speaking = "–" if stats.speaking is None else f"{stats.speaking:.0%}"
        return Section(f"Current stage: group {group_name}", [
            f"Group {group_name}: {size_text(sizes, group_name)}"
            + (f"; {definition.features}" if definition.features else ""),
            Item(f"membership: {{ {definition.membership_sparql} }}"),
            f"Local names (sample of {stats.sample_size}, {stats.labelled} with a name predicate): {agreement} agree with "
            f"a name predicate, {speaking} look like words -> default local_name_type: {self.name_type.value}",
            ], [
            *self.role_editor.input_for_llm(),
            Section("KG-wide elements not used in this group (dropped)", [
                Item(f"{element.role_name}: {element.kind.value} {short_iri(str(element.definition))}")
                for element in unused
                ] or ["none"]),
            Section("Predicates of this group's entities that have no role yet", [
                "Share of its entities that are their subject (as ?s), kinds of their values (?o):",
                Table(("predicate", "as ?s", "values (?o)"), [
                    (short_iri(p), format_share(c.entities, size),
                     format_value_type_shares(c.values, c.literals, c.blanks))
                    for p, c in unassigned
                    ]) if unassigned else "none",
                ]),
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
        current = self.default_profile.get_group(self.group_name)
        updated = current.model_copy(update={"local_name_type": name_type, "added": added, "removed": removed})
        profile = self.default_profile.with_replaced_group(updated)
        return profile, (f"Group {self.group_name}: local names {name_type.value}, "
                         f"{len(added)} elements added or overridden, {len(removed)} removed")


# --- f. revision: resolve the flags, condense the findings ------------------------------

# functions whose successful call makes dependent editors outdated
SPACE_CHANGES = {ProfileFunctionName.SET_ENTITY_SPACE}
PARTITION_CHANGES = {ProfileFunctionName.SET_PARTITION, ProfileFunctionName.REMOVE_PARTITION}
GROUP_CHANGES = {ProfileFunctionName.SET_GROUP, ProfileFunctionName.EXCLUDE_GROUP, ProfileFunctionName.REMOVE_GROUP}
KG_ROLE_CHANGES = {
    ProfileFunctionName.EDIT_ROLE, ProfileFunctionName.SET_STRUCTURE_RELATION, ProfileFunctionName.REMOVE_STRUCTURE_RELATION
    }
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
    """
    name = "om-profile-revision"
    stage = Stage.REVISION
    is_revision_task = True
    prompt_file = "profile_revision.txt"
    entity_space_editor: EntitySpaceEditor
    partition_editor: PartitionEditor
    group_editor: GroupEditor
    partition_role_editor: RoleEditor             # the KG-wide roles, coverage per partition
    group_role_editors: dict[str, RoleEditor]     # group name -> its roles; created on first use

    def prepare(self, input: dict) -> Section:
        input_profile = self.input_profile
        try:
            self.entity_space_editor = EntitySpaceEditor.from_kg_profile(
                self.kg_manager, input_profile, create_missing=False
                )
            self.partition_editor = PartitionEditor.from_kg_profile(self.kg_manager, input_profile, create_missing=False)
            self.group_editor = GroupEditor.from_kg_profile(self.kg_manager, input_profile, create_missing=False)
            self.partition_role_editor = RoleEditor.from_kg_profile(self.kg_manager, input_profile, create_missing=False)
        except ValueError as e:
            raise ValueError(f"The revision needs a complete profile: {e}") from e
        self.group_role_editors = {}
        self.new_flags = {
            id: flag.model_copy() for id, flag in input_profile.flags.items()
            }
        self.new_findings = {
            id: finding.model_copy() for id, finding in input_profile.findings.items()
            }
        # without finish, the profile stays as it is, the flags stay open
        self.default_profile = input_profile.model_copy(deep=True)
        self.stats = {
            "open_flags": len(self.open_flag_ids()), "findings": len(self.new_findings)
            }

        return Section("Current stage: revision", [
            "The whole profile with its statistics. The roles of a group: "
            f"{ProfileFunctionName.SHOW_ROLES} with the group.",
            ], [
            *self.entity_space_editor.input_for_llm(),
            *self.partition_editor.input_for_llm(),
            *self.group_editor.input_for_llm(),
            *self.partition_role_editor.input_for_llm(),
            Section(
                "Open flags with their evidence",
                self.notes_with_current_evidence("flags")
                ),
            Section(
                "Findings with their evidence",
                self.notes_with_current_evidence("findings")
                )
            ])

    def notes_with_current_evidence(self, kind: Literal["flags", "findings"]) -> list[SectionContent]:
        items = []
        notes = (
            self.new_findings if kind == "findings"
            else {
                id: flag for id, flag in self.new_flags.items()
                if flag.status == FlagStatus.OPEN}
            )
        for note_id, note in notes.items():
            try:
                evidence = self.run_evidence_query(
                    {"evidence_query": note.evidence_query}
                ).formatted
            except FunctionCallException as error:
                evidence = f"The evidence query fails now: {error}"
            items.append(Item(
                f"[{note_id}]{f' on {note.concerns}' if isinstance(note, Flag) else ""} {note.content}",
                [Item("Evidence from the knowledge graph:\n" + evidence)]
                ))
        return items or ["none"]

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
            "name": ProfileFunctionName.SHOW_ROLES,
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
            return format_display_content(
                Section("KG-wide roles", subsections=self.partition_role_editor.input_for_llm()))
        return format_display_content(Section(
            f"Roles of group {group_name}", subsections=self.group_role_editor(group_name).input_for_llm()))

    # --- flags and findings ---

    def open_flag_ids(self) -> list[str]:
        return [flag_id for flag_id, flag in self.new_flags.items()
                if flag.status == FlagStatus.OPEN]

    def flag_and_finding_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            ({
                "name": ProfileFunctionName.RESOLVE_FLAG,
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
        ]

    def resolve_flag(self, fn_args: dict) -> str:
        flag_id: str = get_required_argument(fn_args, "id")
        flag = self.new_flags.get(flag_id)
        if flag is None:
            raise FunctionCallException(
                f"No flag {flag_id}; flags: {', '.join(self.new_flags) or 'none'}"
                )
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

    # --- stage interface ---

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        functions = self.rebuilding(self.entity_space_editor.functions(), SPACE_CHANGES,
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
        if len(self.new_findings) > MAX_FINDINGS_AFTER_REVISION:
            raise FunctionCallException(
                f"At most {MAX_FINDINGS_AFTER_REVISION} findings after the revision ({len(self.new_findings)} now): "
                f"merge findings that say the same ({ProfileFunctionName.ADD_FINDING} the merged one, "
                f"{ProfileFunctionName.REMOVE_FINDING} the others) and remove wrong ones"
                )
        profile = self.current_profile()
        counts = {status: sum(flag.status == status for flag in self.new_flags.values()) for status in FlagStatus}
        summary = (f"Flags: {counts[FlagStatus.RESOLVED]} resolved, {counts[FlagStatus.DISMISSED]} dismissed, "
                   f"{counts[FlagStatus.OPEN]} open ({', '.join(self.open_flag_ids()) or 'none'}); "
                   f"findings: {self.stats['findings']} -> {len(self.new_findings)}")
        return profile, summary


PROFILE_TASKS = [SpaceTask, PartitionsTask, RolesTask, GroupsTask, GroupTask, ProfileRevisionTask]
