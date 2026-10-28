"""
The four stages of the KG profile (profile_model.py), each a GRASP task on one KG:
    om-profile-space       a. entity space
    om-profile-partitions  b. partitions: check the kinds; split, redefine, exclude
    om-profile-roles       c. KG-wide roles and structure relations
    om-profile-partition   d. one partition: local names and deviations from the KG-wide roles
"""
from enum import StrEnum
import re
from typing import Callable

from grasp.manager import KgManager
from grasp.tasks.om.kg_profile import (
    KINDS, Coverage, LocalNameStats, Partitions, Sides, canon_roles, complement, default_entity_space,
    default_roles, distinguishing_features, element_sparql, element_usage, kind_partitions, local_name_stats,
    local_name_type, member_coverage, membership_overlap, prefix_counts, prefix_totals,
    partition_query, partition_sizes, predicate_coverage, sample_entities, type_coverage,
)
from grasp.tasks.om.model import (
    LABEL_PREDICATES, RDF_TYPE, UNCATEGORIZED, VOCABULARY, NameType, namespace_of, sparql_path,
)
from grasp.tasks.om.profile_model import (
    ROLE_FIELDS, KgProfile, PartitionDefinition, PartitionOrigin, PartitionProfile, Roles, Stage, Usage,
    derived_roles, diff_roles, effective_roles, role_elements, roles_from_elements, short_iri,
)
from grasp.tasks.om.profile_stage import (
    RoleEditingTask, ProfileStageTask, run_rows, share,
)
from grasp.tasks.om.render import clip
from grasp.utils import FunctionCallException, format_section

NUM_NAMESPACES = 15
NUM_FEATURES = 10
NUM_PREDICATES = 25
PARTITION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def entity_examples(task: ProfileStageTask, partitions: Partitions | None, space: list[str],
                    where: str, k: int) -> list[str]:
    """Up to k entities with a name and their partition."""
    names = sparql_path(LABEL_PREDICATES)
    rows = run_rows(task.manager, partition_query(
        task.manager, partitions, space, "?entity ?partition ?name",
        f"{where}\nOPTIONAL {{ ?entity {names} ?name }}", f"LIMIT {k * 4}",
    ))
    seen, lines = set(), []
    for row in rows:
        entity = row["entity"].value
        if entity in seen:
            continue
        seen.add(entity)
        name = f' "{clip(row["name"].value, 50)}"' if "name" in row else ""
        lines.append(f"  {entity}{name} ({row['partition'].value})")
        if len(lines) == k:
            break
    return lines


def ensure_space(task: ProfileStageTask) -> list[str]:
    if task.input_profile.entity_space:
        return list(task.input_profile.entity_space)
    space, _ = default_entity_space(task.manager, task.config)
    return space


def ensure_partitions(task: ProfileStageTask, space: list[str]) -> Partitions:
    if task.input_profile.partitions:
        return {name: p.definition for name, p in task.input_profile.partitions.items()}
    sizes = partition_sizes(task.manager, None, space)
    return kind_partitions(k for k in KINDS if sizes.get(k))


# --- a. entity space ----------------------------------------------------------------

class SpaceTask(ProfileStageTask):
    name = "om-profile-space"
    stage = Stage.SPACE
    prompt_file = "profile_space.txt"

    def prepare(self, input: dict) -> str:
        if self.input_profile.entity_space:
            space, source = list(self.input_profile.entity_space), "given in the input profile"
        else:
            space, source = default_entity_space(self.manager, self.config)
        named, all_counts = prefix_counts(self.manager, True), prefix_counts(self.manager, False)
        named_totals, totals = prefix_totals(named), prefix_totals(all_counts)
        self.stats = {"prefixes": dict(totals.most_common()), "named": dict(named_totals.most_common())}
        self.default = self.input_profile.model_copy(update={"entity_space": space}, deep=True)
        hosts: dict[str, list[str]] = {}
        for prefix, _ in totals.most_common():
            hosts.setdefault(namespace_of(prefix), []).append(prefix)
        lines = []
        for host, prefixes in sorted(hosts.items(), key=lambda x: -sum(totals[p] for p in x[1]))[:NUM_NAMESPACES]:
            if prefixes != [host]:
                lines.append(f"  {host}: {sum(totals[p] for p in prefixes)} entities, "
                             f"{sum(named_totals.get(p, 0) for p in prefixes)} with a canon name")
            for prefix in prefixes[:NUM_NAMESPACES]:
                kinds = ", ".join(f"{k} {n}" for (other, k), n in sorted(all_counts.items(), key=lambda x: -x[1])
                                  if other == prefix)
                indent = "    " if prefixes != [host] else "  "
                lines.append(f"{indent}{prefix}: {totals[prefix]} entities ({kinds}), "
                             f"{named_totals.get(prefix, 0)} with a canon name")
            if len(prefixes) > NUM_NAMESPACES:
                lines.append(f"    ... {len(prefixes) - NUM_NAMESPACES} more prefixes")
        if len(hosts) > NUM_NAMESPACES:
            lines.append(f"  ... {len(hosts) - NUM_NAMESPACES} more hosts")
        examples = []
        for prefix, _ in totals.most_common(3):
            examples.append(f"{prefix}:")
            examples += entity_examples(self, None, [prefix], "", 4)
        return "\n\n".join([
            f"Default entity space ({source}): " + (", ".join(space) or "none"),
            "Entities per IRI prefix (host, and below it the first path segment) and kind:\n" + "\n".join(lines),
            "Examples:\n" + "\n".join(examples),
        ])

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [({
            "name": "inspect_namespace",
            "description": ("Shows the entities under an IRI prefix: their number per kind, how many have a "
                            "canon name, and examples with names and types."),
            "parameters": {
                "type": "object",
                "properties": {"prefix": {"type": "string", "description": "IRI prefix, e.g. http://example.org/onto#"}},
                "required": ["prefix"],
                "additionalProperties": False,
            },
        }, self.inspect_namespace)]

    def finish_properties(self) -> dict[str, dict]:
        return {"entity_space": {
            "type": "array", "items": {"type": "string"},
            "description": "IRI prefixes of the entities to align.",
        }}

    def finish_required(self) -> list[str]:
        return ["entity_space"]

    def inspect_namespace(self, fn_args: dict) -> str:
        prefix = fn_args.get("prefix") or ""
        if not prefix:
            raise FunctionCallException("prefix required")
        sizes = run_sizes(self, None, [prefix])
        if not sizes:
            return f"No entities with IRIs starting with {prefix}."
        named = run_rows(self.manager, partition_query(
            self.manager, None, [prefix], "(COUNT(DISTINCT ?entity) AS ?n)",
            f"FILTER EXISTS {{ ?entity {sparql_path(LABEL_PREDICATES)} ?name }}",
        ))
        n_named = int(named[0]["n"].value) if named and "n" in named[0] else 0
        types = run_rows(self.manager, partition_query(
            self.manager, None, [prefix], "?t (COUNT(DISTINCT ?entity) AS ?n)",
            f"?entity <{RDF_TYPE}> ?t", "GROUP BY ?t ORDER BY DESC(?n) LIMIT 8",
        ))
        return "\n".join([
            f"{prefix}: " + ", ".join(f"{p} {n}" for p, n in sizes.items()) + f"; {n_named} with a canon name",
            "Most frequent types: " + (", ".join(f"{short_iri(r['t'].value)} {r['n'].value}" for r in types if "t" in r) or "none"),
            "Examples:", *entity_examples(self, None, [prefix], "", 8),
        ])

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        space = [ns for ns in fn_args.get("entity_space") or [] if ns]
        if not space:
            raise FunctionCallException("entity_space must name at least one IRI prefix")
        empty = [ns for ns in space if not run_sizes(self, None, [ns])]
        if empty:
            raise FunctionCallException(f"No entities under {empty}")
        profile = self.input_profile.model_copy(update={"entity_space": space}, deep=True)
        return profile, f"Entity space: {', '.join(space)}"


def run_sizes(task: ProfileStageTask, partitions: Partitions | None, space: list[str]) -> dict[str, int]:
    try:
        return partition_sizes(task.manager, partitions, space)
    except Exception as e:
        raise FunctionCallException(f"The SPARQL query failed: {e}") from e


# --- b. partitions ------------------------------------------------------------------

def format_features(inside: dict[str, int], size_inside: int, outside: dict[str, int], size_outside: int,
                    other: str) -> list[str]:
    return [f"  {short_iri(f.key)}: {f.inside:.0%} inside, {f.outside:.0%} in {other}"
            for f in distinguishing_features(inside, size_inside, outside, size_outside, NUM_FEATURES)]


class PartitionsTask(ProfileStageTask):
    name = "om-profile-partitions"
    stage = Stage.PARTITIONS
    prompt_file = "profile_partitions.txt"

    def prepare(self, input: dict) -> str:
        self.space = ensure_space(self)
        self.definitions: Partitions = ensure_partitions(self, self.space)
        self.tested: set[str] = set()
        self.sizes = partition_sizes(self.manager, self.definitions, self.space)
        self.predicates = {p: {k: c.entities for k, c in counts.items()}
                           for p, counts in predicate_coverage(self.manager, self.definitions, self.space).items()}
        self.types = type_coverage(self.manager, self.definitions, self.space)
        self.stats = {"sizes": self.sizes}
        self.default = self.with_partitions(self.definitions)
        sections = [self.format_sizes()]
        for name in self.definitions:
            size = self.sizes.get(name, 0)
            if not size:
                continue
            lines = [f"Partition {name} ({size} entities). Examples:",
                     *entity_examples(self, self.definitions, self.space, f'FILTER(?partition = "{name}")', 3)]
            outside, size_outside = complement(self.predicates, self.sizes, name)
            lines.append("Predicates (share of entities with the predicate):")
            lines += format_features(self.predicates.get(name, {}), size, outside, size_outside, "other partitions")
            outside, size_outside = complement(self.types, self.sizes, name)
            if self.types.get(name):
                lines.append("Types:")
                lines += format_features(self.types[name], size, outside, size_outside, "other partitions")[:5]
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def format_sizes(self) -> str:
        lines = []
        for name, definition in self.definitions.items():
            how = ("by kind" if definition.membership is None
                   else f"{definition.origin.value}{' from ' + definition.parent if definition.parent else ''}: "
                        f"{{ {definition.membership} }}")
            excluded = " [excluded]" if definition.excluded else ""
            lines.append(f"  {name}: {self.sizes.get(name, 0)} entities ({how}){excluded}")
        if self.sizes.get(UNCATEGORIZED):
            lines.append(f"  {UNCATEGORIZED}: {self.sizes[UNCATEGORIZED]} entities (no partition matches)")
        return "Partitions in order (an entity belongs to the first that matches):\n" + "\n".join(lines)

    def with_partitions(self, definitions: Partitions) -> KgProfile:
        """The input profile with these partitions; profiles of kept partitions stay."""
        old = self.input_profile.partitions
        partitions = {
            name: (old[name].model_copy(update={"definition": d}) if name in old else PartitionProfile(definition=d))
            for name, d in definitions.items()
        }
        return self.input_profile.model_copy(update={"entity_space": self.space, "partitions": partitions}, deep=True)

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        membership = {"type": "string", "description": (
            "SPARQL group pattern over ?entity, e.g. ?entity <http://example.org/p> ?v")}
        return [
            ({
                "name": "test_partition",
                "description": (
                    "Tests a membership pattern: how many entities of the entity space match, in which "
                    "current partitions they are, which predicates and types distinguish them from the "
                    "rest of the partition they overlap most (or compare_with), and examples. A pattern "
                    "must be tested before it can be set."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"membership": membership,
                                   "compare_with": {"type": "string", "description": "Partition to compare with."}},
                    "required": ["membership"],
                    "additionalProperties": False,
                },
            }, self.test_partition),
            ({
                "name": "set_partition",
                "description": (
                    "Defines a partition by a tested membership pattern, or redefines an existing one. "
                    "With parent, it is split from that partition and placed before it; a new partition "
                    "without parent is placed first. Shows the partition sizes afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A short lower case noun for the group."},
                        "membership": membership,
                        "parent": {"type": "string"},
                        "features": {"type": "string", "description": "What distinguishes its entities."},
                    },
                    "required": ["name", "membership", "features"],
                    "additionalProperties": False,
                },
            }, self.set_partition),
            ({
                "name": "exclude_partition",
                "description": ("Excludes a partition from the alignment (entities that would not be matched for this "
                                "task), or includes it again."),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "reason": {"type": "string"},
                                   "excluded": {"type": "boolean", "default": True}},
                    "required": ["name", "reason"],
                    "additionalProperties": False,
                },
            }, self.exclude_partition),
            ({
                "name": "remove_partition",
                "description": "Removes a partition you defined; its entities fall to the next matching partition.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }, self.remove_partition),
        ]

    def known_partition(self, name: str | None) -> str:
        if name not in self.definitions:
            raise FunctionCallException(f"Unknown partition {name}; partitions: {', '.join(self.definitions)}")
        return name

    def test_partition(self, fn_args: dict) -> str:
        membership = (fn_args.get("membership") or "").strip()
        if "?entity" not in membership:
            raise FunctionCallException("The membership pattern must use ?entity")
        try:
            overlap = membership_overlap(self.manager, self.definitions, self.space, membership)
            inside, inside_types = member_coverage(self.manager, self.space, membership)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        size = sum(overlap.values())
        self.tested.add(membership)
        if not size:
            return "No entity of the entity space matches."
        compare = fn_args.get("compare_with")
        if compare is not None:
            self.known_partition(compare)
        else:
            compare = max((p for p in overlap if p in self.sizes), key=overlap.get, default=None)
        lines = [f"{size} entities match; by current partition: "
                 + ", ".join(f"{p} {share(n, self.sizes.get(p, 0))}" for p, n in sorted(overlap.items(), key=lambda x: -x[1]))]
        if compare is not None:
            rest = self.sizes.get(compare, 0) - overlap.get(compare, 0)
            for title, counts, ins in (("Predicates", self.predicates, inside), ("Types", self.types, inside_types)):
                outside = {k: max(0, n - ins.get(k, 0)) for k, n in counts.get(compare, {}).items()}
                lines.append(f"{title} (share of the matches vs. the rest of {compare}):")
                lines += format_features(ins, size, outside, rest, f"rest of {compare}")[:NUM_FEATURES if title == "Predicates" else 5]
        lines.append("Examples:")
        names = sparql_path(LABEL_PREDICATES)
        rows = run_rows(self.manager, (
            f"SELECT DISTINCT ?entity ?name WHERE {{ {{ SELECT DISTINCT ?entity WHERE {{ {membership} }} }} "
            f"OPTIONAL {{ ?entity {names} ?name }} }} LIMIT 6"
        ))
        lines += [f"  {r['entity'].value}" + (f' "{clip(r["name"].value, 50)}"' if "name" in r else "")
                  for r in rows if "entity" in r]
        return "\n".join(lines)

    def set_partition(self, fn_args: dict) -> str:
        name = fn_args.get("name") or ""
        membership = (fn_args.get("membership") or "").strip()
        parent = fn_args.get("parent")
        if not PARTITION_NAME.match(name) or name == UNCATEGORIZED:
            raise FunctionCallException(f"Invalid partition name {name!r}: lower case letters, digits and _")
        if membership not in self.tested:
            raise FunctionCallException("Test the membership pattern with test_partition first (exactly this pattern)")
        if parent is not None and (self.known_partition(parent) == name):
            raise FunctionCallException("A partition cannot be split from itself")
        definition = PartitionDefinition(
            membership=membership, parent=parent, features=fn_args.get("features") or "",
            origin=PartitionOrigin.SPLIT if parent else PartitionOrigin.REDEFINED,
        )
        items = [(n, d) for n, d in self.definitions.items() if n != name]
        if name in self.definitions:
            index = list(self.definitions).index(name)
        elif parent is not None:
            index = [n for n, _ in items].index(parent)
        else:
            index = 0
        items.insert(index, (name, definition))
        self.update_definitions(dict(items))
        return f"Partition {name} set.\n{self.format_sizes()}"

    def exclude_partition(self, fn_args: dict) -> str:
        name = self.known_partition(fn_args.get("name"))
        definition = self.definitions[name]
        excluded = fn_args.get("excluded", True)
        reason = fn_args.get("reason") or ""
        features = definition.features
        if excluded and reason:
            features = f"{features}; excluded: {reason}" if features else f"excluded: {reason}"
        self.definitions[name] = definition.model_copy(update={"excluded": excluded, "features": features})
        return f"Partition {name} {'excluded' if excluded else 'included'}.\n{self.format_sizes()}"

    def remove_partition(self, fn_args: dict) -> str:
        name = self.known_partition(fn_args.get("name"))
        if self.definitions[name].origin == PartitionOrigin.DEFAULT:
            raise FunctionCallException(f"{name} is a default partition; exclude it or redefine it instead")
        self.update_definitions({n: d for n, d in self.definitions.items() if n != name})
        return f"Partition {name} removed.\n{self.format_sizes()}"

    def update_definitions(self, definitions: Partitions) -> None:
        sizes = run_sizes(self, definitions, self.space)
        self.definitions, self.sizes = definitions, sizes
        try:
            self.predicates = {p: {k: c.entities for k, c in counts.items()}
                               for p, counts in predicate_coverage(self.manager, definitions, self.space).items()}
            self.types = type_coverage(self.manager, definitions, self.space)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        kept = {name: d for name, d in self.definitions.items() if self.sizes.get(name)}
        if not kept:
            raise FunctionCallException("No partition has entities")
        self.stats = {"sizes": self.sizes}
        dropped = [name for name in self.definitions if name not in kept]
        summary = "Partitions: " + ", ".join(f"{n} {self.sizes[n]}" for n in kept)
        if dropped:
            summary += f"; dropped as empty: {', '.join(dropped)}"
        return self.with_partitions(kept), summary


# --- c. KG-wide roles ----------------------------------------------------------------

def active_partitions(profile: KgProfile) -> list[str]:
    return [name for name, p in profile.partitions.items() if not p.definition.excluded]


def format_values(values: int, literals: int, blanks: int) -> str:
    """Kinds of the values (objects) of a predicate."""
    if not values:
        return ""
    iris = values - literals - blanks
    parts = [f"{iris / values:.0%} IRI"]
    if blanks:
        parts.append(f"{blanks / values:.0%} blank node")
    parts.append(f"{literals / values:.0%} literal")
    return ", ".join(parts)


def format_predicate_overview(coverage: dict[str, dict[str, Coverage]], sizes: dict[str, int],
                              partitions: list[str], roles: Roles, k: int) -> str:
    """The predicates with the highest coverage in some partition: the share of
    each partition's entities that are its subject (as ?s), the kinds of its
    values (?o) over these partitions, and the role it has."""
    in_role = {key[2]: key[0] for key in role_elements(roles) if key[1] == "predicate"}
    best: dict[str, float] = {}
    for p in partitions:
        for predicate, c in coverage.get(p, {}).items():
            if sizes.get(p):
                best[predicate] = max(best.get(predicate, 0.0), c.entities / sizes[p])
    lines = []
    for predicate in sorted(best, key=lambda x: -best[x])[:k]:
        per_partition = []
        values = literals = blanks = 0
        for p in partitions:
            c = coverage.get(p, {}).get(predicate)
            if c is not None:
                per_partition.append(f"{p} {c.entities / sizes[p]:.0%}")
                values, literals, blanks = values + c.values, literals + c.literals, blanks + c.blanks
        role = f" [{in_role[predicate]}]" if predicate in in_role else ""
        lines.append(f"  {short_iri(predicate)}{role}  as ?s: {', '.join(per_partition)}"
                     f"  |  values: {format_values(values, literals, blanks)}")
    return "\n".join(lines)


class RolesTask(RoleEditingTask):
    name = "om-profile-roles"
    stage = Stage.ROLES
    prompt_file = "profile_roles.txt"

    def prepare(self, input: dict) -> str:
        self.space = ensure_space(self)
        profile = self.input_profile
        if not profile.partitions:
            profile = profile.model_copy(update={"partitions": {
                n: PartitionProfile(definition=d) for n, d in ensure_partitions(self, self.space).items()}})
        self.partitions = {name: p.definition for name, p in profile.partitions.items()}
        self.scope = active_partitions(profile)
        self.sizes = partition_sizes(self.manager, self.partitions, self.space)
        sides: dict[str, Sides] = {}
        usage = element_usage(self.manager, self.partitions, self.space, canon_roles(), sides)
        default = default_roles(usage, self.sizes, self.scope)
        self.init_roles(default, usage, sides, canon_roles())
        coverage = predicate_coverage(self.manager, self.partitions, self.space)
        self.stats = {"sizes": self.sizes}
        self.default = profile.model_copy(update={"entity_space": self.space, "roles": default}, deep=True)
        canon = ", ".join(short_iri(key[2]) if key[1] == "predicate" else key[2]
                          for key in role_elements(canon_roles()))
        return "\n\n".join([
            "Partitions in scope: " + ", ".join(f"{p} ({self.sizes.get(p, 0)} entities)" for p in self.scope),
            f"Default canon checked: {canon}. Canon elements below "
            "1% coverage in every partition are left out of the default.",
            "Default roles (coverage as ?s per partition):\n" + self.format_roles(default),
            "Most frequent predicates: share of each partition's entities that are their subject "
            "(as ?s), kinds of their values (?o), [role] if in the default:\n"
            + format_predicate_overview(coverage, self.sizes, self.scope, default, NUM_PREDICATES),
        ])

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.role_functions()

    def stage_rules(self) -> list[str]:
        return ["Test every predicate or pattern you add with try_pattern first."]

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        roles = self.checked_roles()
        profile = self.default.model_copy(update={"roles": roles}, deep=True)
        changed = [f for f in (*ROLE_FIELDS, "structure") if getattr(roles, f) != getattr(self.default.roles, f)]
        return profile, "Changed: " + (", ".join(changed) or "nothing (default roles)")


# --- d. one partition ----------------------------------------------------------------

def partition_name_stats(manager: KgManager, partitions: Partitions, space: list[str],
                         partition: str, roles: Roles) -> LocalNameStats:
    sample = sample_entities(manager, partitions, space, where=f'FILTER(?partition = "{partition}")')
    return local_name_stats(manager, sample.get(partition, []), roles.names)


def partition_defaults(manager: KgManager, profile: KgProfile, partition: str
                       ) -> tuple[PartitionProfile, LocalNameStats]:
    """Profile of a partition without LLM: the local name type from its sample, its delta unchanged."""
    partitions = {name: p.definition for name, p in profile.partitions.items()}
    current = profile.partitions[partition]
    usage = element_usage(manager, partitions, profile.entity_space, profile.roles).get(partition, {})
    effective = effective_roles(profile.roles, current, usage)
    stats = partition_name_stats(manager, partitions, profile.entity_space, partition, effective)
    return current.model_copy(update={"local_name_type": local_name_type(stats)}, deep=True), stats


# stage d runs in full mode for a partition with at least MIN_UNASSIGNED
# predicates without a role, each covering MIN_UNASSIGNED_SHARE of its entities
# as subject (vocabulary predicates such as rdf:type not counted), or for a
# partition formed in stage b; otherwise in light mode
MIN_UNASSIGNED = 3
MIN_UNASSIGNED_SHARE = 0.05


class PartitionMode(StrEnum):
    FULL = "full"    # roles of the partition, findings
    LIGHT = "light"  # mainly matching pitfalls of the partition as findings


def unassigned_predicates(roles: Roles, coverage: dict[str, Coverage]) -> list[tuple[str, Coverage]]:
    """Predicates of a partition in no element of its roles (as predicate or
    inside a pattern), most frequent first."""
    elements = role_elements(roles)
    predicates = {key[2] for key in elements if key[1] == "predicate"}
    patterns = [element_sparql(key, value) for key, value in elements.items() if key[1] != "predicate"]
    covered = lambda p: p in predicates or any(f"<{p}>" in pattern for pattern in patterns)
    return [(p, c) for p, c in sorted(coverage.items(), key=lambda x: -x[1].entities) if not covered(p)]


def partition_mode(manager: KgManager, profile: KgProfile, partition: str) -> tuple[PartitionMode, list[str]]:
    """(mode of stage d for a partition, the unassigned predicates that count)."""
    partitions = {name: p.definition for name, p in profile.partitions.items()}
    current = profile.partitions[partition]
    usage = element_usage(manager, partitions, profile.entity_space, profile.roles).get(partition, {})
    effective = effective_roles(profile.roles, current, usage)
    coverage = predicate_coverage(manager, partitions, profile.entity_space).get(partition, {})
    size = partition_sizes(manager, partitions, profile.entity_space).get(partition, 0)
    counted = [p for p, c in unassigned_predicates(effective, coverage)
               if not p.startswith(VOCABULARY) and size and c.entities / size >= MIN_UNASSIGNED_SHARE]
    formed = current.definition.origin != PartitionOrigin.DEFAULT
    return (PartitionMode.FULL if formed or len(counted) >= MIN_UNASSIGNED else PartitionMode.LIGHT), counted


class PartitionTask(RoleEditingTask):
    name = "om-profile-partition"
    stage = Stage.PARTITION
    prompt_file = "profile_partition.txt"
    prompt_files = {PartitionMode.FULL: "profile_partition.txt", PartitionMode.LIGHT: "profile_partition_light.txt"}

    def prepare(self, input: dict) -> str:
        profile = self.input_profile
        self.partition = input.get("partition")
        self.mode = PartitionMode(input.get("mode") or PartitionMode.FULL)
        self.prompt_file = self.prompt_files[self.mode]
        if self.partition not in profile.partitions:
            raise ValueError(f"Unknown partition {self.partition}; partitions: {', '.join(profile.partitions)}")
        self.space = list(profile.entity_space)
        self.partitions = {name: p.definition for name, p in profile.partitions.items()}
        self.scope = [self.partition]
        self.sizes = partition_sizes(self.manager, self.partitions, self.space)
        current = profile.partitions[self.partition]
        sides: dict[str, Sides] = {}
        kg_usage = element_usage(self.manager, self.partitions, self.space,
                                 roles_from_elements(role_elements(profile.roles) | role_elements(current.added)), sides)
        self.added_keys = set(role_elements(current.added))
        self.derived = derived_roles(profile.roles, kg_usage.get(self.partition, {}))
        effective = effective_roles(profile.roles, current, kg_usage.get(self.partition, {}))
        self.init_roles(effective, kg_usage, sides, effective)
        stats = partition_name_stats(self.manager, self.partitions, self.space, self.partition, effective)
        self.name_type = local_name_type(stats)
        self.stats = {"size": self.sizes.get(self.partition, 0), "mode": self.mode.value,
                      "local_names": stats.model_dump()}
        default = current.model_copy(update={"local_name_type": self.name_type})
        self.default = profile.model_copy(update={"partitions": {**profile.partitions, self.partition: default}}, deep=True)

        unused = [key for key in role_elements(profile.roles) if key not in role_elements(self.derived)
                  and key not in role_elements(current.added)]
        coverage = predicate_coverage(self.manager, self.partitions, self.space).get(self.partition, {})
        size = self.sizes.get(self.partition, 0)
        others = [f"  {short_iri(p)}  as ?s: {share(c.entities, size)}  |  values: {format_values(c.values, c.literals, c.blanks)}"
                  for p, c in unassigned_predicates(effective, coverage)][:NUM_PREDICATES]
        definition = current.definition
        agreement = "–" if stats.agreement is None else f"{stats.agreement:.0%}"
        speaking = "–" if stats.speaking is None else f"{stats.speaking:.0%}"
        return "\n\n".join([
            f"Partition {self.partition}: {size} entities"
            + (f"; {definition.features}" if definition.features else "")
            + (f"; membership {{ {definition.membership} }}" if definition.membership else "; by kind"),
            f"Local names (sample of {stats.sample_size}, {stats.labelled} with a name): {agreement} agree with "
            f"a name, {speaking} look like words -> default local_name_type: {self.name_type.value}",
            "Roles of this partition: the KG-wide roles used here [kg], and its own additions [partition] "
            "(coverage as ?s):\n" + self.format_roles(effective, self.origin),
            "KG-wide elements not used in this partition (dropped): "
            + (", ".join(short_iri(k[2]) for k in unused) or "none"),
            "Predicates of this partition's entities that have no role yet: share of its entities that "
            "are their subject (as ?s), kinds of their values (?o):\n" + ("\n".join(others) or "  none"),
        ])

    def origin(self, key) -> str:
        return "partition" if key in self.added_keys else "kg"

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.role_functions()

    def finish_properties(self) -> dict[str, dict]:
        return {"local_name_type": {
            "type": "string", "enum": [t.value for t in NameType],
            "description": "If the default is wrong: label if local names are names, code if identifiers, other if mixed.",
        }}

    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        roles = self.checked_roles()
        added, removed = diff_roles(self.derived, roles)
        name_type = NameType(fn_args.get("local_name_type") or self.name_type)
        current = self.default.partitions[self.partition]
        updated = current.model_copy(update={"local_name_type": name_type, "added": added, "removed": removed})
        profile = self.default.model_copy(update={"partitions": {**self.default.partitions, self.partition: updated}}, deep=True)
        count = lambda r: len(role_elements(r))
        return profile, (f"Partition {self.partition}: local names {name_type.value}, "
                         f"{count(added)} elements added or overridden, {count(removed)} removed")


PROFILE_TASKS = [SpaceTask, PartitionsTask, RolesTask, PartitionTask]
