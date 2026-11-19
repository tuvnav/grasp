"""
Base of the profile stage tasks (profile_tasks.py). Each stage reads the
profile of the stages before it (input: kg, profile), computes its default and
statistics, lets the LLM check and change its part, and outputs the profile
with its part; without a finish call, its default.

Shared here: the input layout (earlier results with the findings as unverified
observations), finish with findings and flags, and the editors of the profile
parts, which hold a part under editing and its functions (RoleEditor: roles,
for the role stages).
"""
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, ClassVar, Literal, Self

from pydantic import BaseModel, ValidationError

from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.profile import (
    PredicatePattern,
    StructureRelation,
    ROLE_DESCRIPTIONS,
    ROLE_FIELDS,
    STAGES,
    ElementKey,
    Finding,
    Flag,
    GroupDefinition,
    GroupOrigin,
    GroupProfile,
    KgProfile,
    Partition,
    ProfileStageOutput,
    Roles,
    Stage,
    Usage,
    WithinKgNonCompetingPartitionPair,
    check_structure,
    derived_roles,
    describe_predicate_pattern,
    describe_profile,
    describe_structure_relation,
    diff_roles,
    effective_roles,
    role_elements,
    roles_from_elements,
    short_iri,
)
from grasp.tasks.om.foundation.configuration import (
    load_task_instructions,
)
from grasp.tasks.om.foundation.types import IRI, LocalNameType
from grasp.tasks.om.foundation.rdf_vocabulary import (
    LABEL_PREDICATES,
    RDF_TYPE,
    UNCATEGORIZED,
    VOCABULARY,
)
from grasp.tasks.om.kg_operations.entity_rendering import clip
from grasp.tasks.om.kg_operations.grouping import (
    Grouping,
    estimated_group_sizes,
    group_sizes,
    grouping_to_sparql_query,
    groups_to_grouping,
    kind_partitions,
    partitions_to_grouping,
)
from grasp.tasks.om.kg_operations.large_kg_sampling import (
    KgSample,
    class_counts,
    current_sample,
    default_class_groups,
    type_membership,
    use_sample,
    values_block,
)
from grasp.tasks.om.kg_operations.profile_statistics import (
    Coverage,
    LocalNameStats,
    Sides,
    canon_roles,
    complement,
    default_entity_space,
    default_roles,
    distinguishing_features,
    element_sparql,
    element_usage,
    local_name_stats,
    local_name_type,
    member_coverage,
    members_query,
    membership_overlap,
    predicate_coverage,
    rest,
    sample_entities,
    type_coverage,
)
from grasp.tasks.om.sparql_access.lookups import (
    select_rows,
    sparql_path,
)
from grasp.tasks.om.tasks.prompting import load_prompt
from grasp.utils import FunctionCallException, format_enumerate, format_section

FINISH = "finish"
LARGE_KG_NOTE = (
    "This knowledge graph is very large. Sizes of partitions and groups are estimates (~); all other "
    "statistics (coverage, value kinds, examples, local names, try_pattern) are computed on samples "
    "(up to 200 entities per membership pattern), so shares are estimates too. In your own "
    "SPARQL queries, always restrict the entities, e.g. to a class or to a few IRIs, and use LIMIT: "
    "queries that scan the whole graph time out."
)
# findings per finish: only the most useful ones end up in prompts of later stages
MAX_FINDINGS = 5
NUM_EXAMPLES = 8
MAX_VALUE_CHARS = 80


def inline_schema(model: type[BaseModel], drop: set[str] = frozenset()) -> dict:
    """JSON schema of a model with its $defs inlined (not every model API
    resolves $ref), without titles, without the given properties and without
    discriminator mappings (they point into $defs; the const kind of each
    alternative of a oneOf is enough)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            node = {k: resolve(v) for k, v in node.items() if k not in ("title", "discriminator")}
            if "properties" in node:
                node["properties"] = {k: v for k, v in node["properties"].items() if k not in drop}
                if "required" in node:
                    node["required"] = [r for r in node["required"] if r not in drop]
            return node
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def share(n: int, size: int) -> str:
    return f"{n}/{size} ({n / size:.0%})" if size else str(n)


def run_rows(manager: KgManager, sparql: str) -> list[dict]:
    try:
        return select_rows(manager, sparql)
    except Exception as e:
        raise FunctionCallException(f"The SPARQL query failed: {e}") from e


# --- helpers of the stages and editors ------------------------------------------------

# defaults of the profile parts

NUM_NAMESPACES = 15


# kinds or classes listed per prefix
NUM_KINDS = 8


NUM_FEATURES = 10


NUM_PREDICATES = 25


NAME = re.compile(r"^[a-z][a-z0-9_]*$")


# stage a counts entities per prefix by kind, without the subjects outside all kinds
SPACE_GROUPING = partitions_to_grouping(kind_partitions(), mark_uncategorized_as_rest=False)


# name of all entities of a grouping, to compare a group with the rest
ALL = "_all"


def entity_examples(manager: KgManager, grouping: Grouping, space: list[str], where: str, k: int) -> list[str]:
    """Up to k entities with a name and their partition or group."""
    names = sparql_path(LABEL_PREDICATES)
    rows = run_rows(manager, grouping_to_sparql_query(
        manager, grouping, space, "?s ?group ?name",
        f"{where}\nOPTIONAL {{ ?s {names} ?name }}", f"LIMIT {k * 4}",
    ))
    seen, lines = set(), []
    for row in rows:
        entity = row["s"].value
        if entity in seen:
            continue
        seen.add(entity)
        name = f' "{clip(row["name"].value, 50)}"' if "name" in row else ""
        lines.append(f"  {entity}{name} ({row['group'].value})")
        if len(lines) == k:
            break
    return lines


def ensure_space(
        kg_manager: KgManager,
        kg_profile: KgProfile,
        config: GraspConfig | None = None
        ) -> list[str]:
    if kg_profile.entity_space:
        return list(kg_profile.entity_space)
    if config:
        space, _ = default_entity_space(kg_manager, config)
        return space
    raise ValueError(
        "Given KgProfile doesn't contain entity namespace and no GraspConfig was given."
        "Either is required."
        )


def default_partitions(kg_manager: KgManager, kg_profile: KgProfile, entity_space: list[str]) -> dict[str, Partition]:
    """The kind partitions with entities (in a large KG without the alternatives
    that cannot be evaluated there)."""
    large = kg_profile.large
    partitions = kind_partitions(large=large)
    sizes = estimated_group_sizes(
        kg_manager, partitions_to_grouping(
            partitions, mark_uncategorized_as_rest=False
            ), entity_space
        )
    return {p.name: p for p in partitions if sizes.get(p.name)}


def ensure_partitions(
        kg_manager: KgManager, kg_profile: KgProfile, entity_space: list[str]
        ) -> dict[str, Partition]:
    if kg_profile.partitions:
        return {p.name: p for p in kg_profile.partitions}
    return default_partitions(kg_manager, kg_profile, entity_space)


def default_non_competing_pairs(partitions: dict[str, Partition]) -> list[WithinKgNonCompetingPartitionPair]:
    """Properties do not compete with the other kinds (the default before stage b decides)."""
    if "property" not in partitions:
        return []
    return [WithinKgNonCompetingPartitionPair(partition_a="property", partition_b=name,
                                              reason="properties and entities are not equivalent")
            for name in partitions if name != "property"]


def ensure_groups(
        kg_manager: KgManager, kg_profile: KgProfile, entity_space: list[str]
        ) -> dict[str, GroupDefinition]:
    """The groups of the input profile; otherwise one per partition and, in a
    large KG, one per large class."""
    profile = kg_profile
    if profile.groups:
        return {g.name: g.definition for g in profile.groups}
    groups = {name: GroupDefinition(name=p.name, membership_sparql=p.membership_sparql, origin=GroupOrigin.PARTITION,
                                    features=p.features)
              for name, p in ensure_partitions(
                  kg_manager, kg_profile, entity_space
                  ).items()}
    if profile.large:
        for name, type_iri, _ in default_class_groups(
                kg_manager, tuple(entity_space), class_counts(kg_manager)
                ):
            if name not in groups:
                groups[name] = GroupDefinition(
                    name=name,
                    membership_sparql=type_membership(type_iri),
                    origin=GroupOrigin.CLASS,
                    features=f"typed {short_iri(type_iri)}"
                    )
    return groups


# sizes, features and definitions of partitions and groups


def size_text(manager: KgManager, sizes: dict[str, int], estimates: dict[str, int], name: str) -> str:
    """Size of a partition or group: counted, or in a large KG estimated with the size of its sample."""
    if current_sample(manager) is None:
        return f"{sizes.get(name, 0)} entities"
    estimate = estimates.get(name)
    return f"{'~' + str(estimate) if estimate is not None else 'unknown number of'} entities, sample {sizes.get(name, 0)}"


def size_stats(
        manager: KgManager, sizes: dict[str, int], estimates: dict[str, int]
        ) -> dict[str, dict[str, int]]:
    """
    Sizes for the output: estimates in a large KG, with the sample sizes.
    """
    if current_sample(manager) is None:
        return {"sizes": sizes}
    return {"sizes": estimates, "sample_sizes": sizes}


def run_sizes(manager: KgManager, grouping: Grouping, space: list[str]) -> dict[str, int]:
    try:
        return group_sizes(manager, grouping, space)
    except Exception as e:
        raise FunctionCallException(f"The SPARQL query failed: {e}") from e


def format_features(inside: dict[str, int], size_inside: int, outside: dict[str, int], size_outside: int,
                    other: str) -> list[str]:
    return [f"  {short_iri(f.key)}: {f.inside:.0%} inside, {f.outside:.0%} in {other}"
            for f in distinguishing_features(inside, size_inside, outside, size_outside, NUM_FEATURES)]


def membership_argument(fn_args: dict) -> str:
    membership = (fn_args.get("membership") or "").strip()
    if not re.search(r"\?s\b", membership):
        raise FunctionCallException("The membership pattern must use ?s")
    return membership


def valid_name(name: str, unit: str) -> str:
    if not NAME.match(name or "") or name in (UNCATEGORIZED, ALL):
        raise FunctionCallException(f"Invalid {unit} name {name!r}: lower case letters, digits and _")
    return name


def test_function(unit: str, description: str) -> dict:
    return {
        "name": f"test_{unit}",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"membership": MEMBERSHIP,
                           "compare_with": {"type": "string", "description": f"The {unit} to compare with."}},
            "required": ["membership"],
            "additionalProperties": False,
        },
    }


MEMBERSHIP = {"type": "string", "description": (
    "SPARQL group pattern over ?s, e.g. ?s <http://example.org/p> ?o")}


# role overviews and the groups of stage e


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
                              members: list[str], roles: Roles, k: int) -> str:
    """The predicates with the highest coverage in some partition: the share of
    each partition's entities that are its subject (as ?s), the kinds of its
    values (?o) over these partitions, and the role it has."""
    in_role = {key[2]: key[0] for key in role_elements(roles) if key[1] == "predicate"}
    best: dict[str, float] = {}
    for p in members:
        for predicate, c in coverage.get(p, {}).items():
            if sizes.get(p):
                best[predicate] = max(best.get(predicate, 0.0), c.entities / sizes[p])
    lines = []
    for predicate in sorted(best, key=lambda x: -best[x])[:k]:
        per_member = []
        values = literals = blanks = 0
        for p in members:
            c = coverage.get(p, {}).get(predicate)
            if c is not None:
                per_member.append(f"{p} {c.entities / sizes[p]:.0%}")
                values, literals, blanks = values + c.values, literals + c.literals, blanks + c.blanks
        role = f" [{in_role[predicate]}]" if predicate in in_role else ""
        lines.append(f"  {short_iri(predicate)}{role}  as ?s: {', '.join(per_member)}"
                     f"  |  values: {format_values(values, literals, blanks)}")
    return "\n".join(lines)


# exclude_group warns if the excluded groups cover more of the entity space
LARGE_EXCLUSION_SHARE = 0.5


def single_grouping(profile: KgProfile, group: str) -> Grouping:
    """The grouping of one group only, without the entities of excluded groups
    (they are not aligned): stage e looks at no other group."""
    excluded = tuple(profile.get_group(name).definition.membership_sparql
                     for name in profile.excluded_group_names() if name != group)
    return Grouping(((group, profile.get_group(group).definition.membership_sparql),), exclusive=False, minus=excluded)


def group_name_stats(manager: KgManager, grouping: Grouping, space: list[str], group: str,
                     roles: Roles) -> LocalNameStats:
    sample = sample_entities(manager, grouping, space, where=f'FILTER(?group = "{group}")')
    return local_name_stats(manager, sample.get(group, []), roles.names)


def group_defaults(manager: KgManager, profile: KgProfile, group: str) -> tuple[GroupProfile, LocalNameStats]:
    """Profile of a group without LLM: the local name type from its sample, its delta unchanged."""
    grouping = single_grouping(profile, group)
    current = profile.get_group(group)
    usage = element_usage(manager, grouping, profile.entity_space, profile.roles).get(group, {})
    effective = effective_roles(profile.roles, current, usage)
    stats = group_name_stats(manager, grouping, profile.entity_space, group, effective)
    return current.model_copy(update={"local_name_type": local_name_type(stats)}, deep=True), stats


# stage e runs in full mode for a group with at least MIN_UNASSIGNED predicates
# without a role, each covering MIN_UNASSIGNED_SHARE of its entities as subject
# (vocabulary predicates such as rdf:type not counted), or for a group formed in
# stage d; otherwise in light mode
MIN_UNASSIGNED = 3


MIN_UNASSIGNED_SHARE = 0.05


class GroupEditMode(StrEnum):
    FULL = "full"    # roles of the group, findings
    LIGHT = "light"  # mainly matching pitfalls of the group as findings


MIN_GROUP_SIZE_FOR_GROUP_STAGE = 50


MAX_GROUP_TASK_STEPS_BY_MODE = {GroupEditMode.FULL: 30, GroupEditMode.LIGHT: 12}


def unassigned_predicates(roles: Roles, coverage: dict[str, Coverage]) -> list[tuple[str, Coverage]]:
    """Predicates of a group in no element of its roles (as predicate or
    inside a pattern), most frequent first."""
    elements = role_elements(roles)
    predicates = {key[2] for key in elements if key[1] == "predicate"}
    patterns = [element_sparql(key, value) for key, value in elements.items() if key[1] != "predicate"]
    covered = lambda p: p in predicates or any(f"<{p}>" in pattern for pattern in patterns)
    return [(p, c) for p, c in sorted(coverage.items(), key=lambda x: -x[1].entities) if not covered(p)]


def group_mode(manager: KgManager, profile: KgProfile, group: str) -> tuple[GroupEditMode, list[str]]:
    """(mode of stage e for a group, the unassigned predicates that count)."""
    grouping = single_grouping(profile, group)
    current = profile.get_group(group)
    usage = element_usage(manager, grouping, profile.entity_space, profile.roles).get(group, {})
    effective = effective_roles(profile.roles, current, usage)
    coverage = predicate_coverage(manager, grouping, profile.entity_space).get(group, {})
    size = group_sizes(manager, grouping, profile.entity_space).get(group, 0)
    counted = [p for p, c in unassigned_predicates(effective, coverage)
               if not p.startswith(VOCABULARY) and size and c.entities / size >= MIN_UNASSIGNED_SHARE]
    formed = current.definition.origin == GroupOrigin.FORMED
    return (GroupEditMode.FULL if formed or len(counted) >= MIN_UNASSIGNED else GroupEditMode.LIGHT), counted


@dataclass(eq=False)
class Editor(ABC):
    """
    One part of a KG profile under editing, with the functions the LLM edits
    it with. A stage task holds the editors of its parts: it puts their
    overviews into its input, offers their functions, and on finish takes
    their checked results (each editor has its own result method, as the
    results differ in type).
    """
    kg_manager: KgManager
    entity_space: list[str]

    @abstractmethod
    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        """The functions for the LLM: (definition, handler) pairs."""

    @abstractmethod
    def input_for_llm(self) -> str:
        """The overview of the current state, for the input of the task."""


@dataclass(eq=False)
class EntitySpaceEditor(Editor):
    """
    The entity space: IRI prefixes of the entities to align. In a large KG,
    the prefixes are inspected on the class samples that the space stage
    draws (prefix_examples, class_prefix_counts); without them, a large KG's
    space cannot be inspected.
    """
    large_kg: bool
    # large KG: prefix -> sampled (entity, class) pairs
    prefix_examples: dict[str, list[tuple[IRI, str]]] = field(default_factory=dict)
    # large KG: (prefix, class) -> estimated entities
    class_prefix_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entity_space = list(self.entity_space)

    @classmethod
    def from_kg_profile(
            cls,
            kg_manager: KgManager,
            kg_profile: KgProfile,
            *,
            create_missing: bool,
            config: GraspConfig | None = None,
            prefix_examples: dict[str, list[tuple[IRI, str]]] | None = None,
            class_prefix_counts: dict[tuple[str, str], int] | None = None,
            ) -> Self:
        """
        The entity space of the profile. create_missing: without one, the
        default space (configured namespaces or the prefix with the most named
        entities; needs config); otherwise the editor starts empty. The class
        samples of a large KG are not in the profile: the space stage hands
        them in.
        """
        entity_space = (ensure_space(kg_manager, kg_profile, config) if create_missing
                        else list(kg_profile.entity_space))
        return cls(
            kg_manager=kg_manager,
            entity_space=entity_space,
            large_kg=kg_profile.large,
            prefix_examples=prefix_examples or {},
            class_prefix_counts=class_prefix_counts or {},
        )

    def input_for_llm(self) -> str:
        return "Entity space: " + (", ".join(self.entity_space) or "none")

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            ({
                "name": "inspect_namespace",
                "description": ("Shows the entities under an IRI prefix: their number per kind, how many have a "
                                "canon name, and examples with names and types."),
                "parameters": {
                    "type": "object",
                    "properties": {"prefix": {"type": "string", "description": "IRI prefix, e.g. http://example.org/onto#"}},
                    "required": ["prefix"],
                    "additionalProperties": False,
                },
            }, self.inspect_namespace),
            ({
                "name": "set_entity_space",
                "description": "Sets the entity space: the IRI prefixes of the entities to align.",
                "parameters": {
                    "type": "object",
                    "properties": {"prefixes": {"type": "array", "items": {"type": "string"}}},
                    "required": ["prefixes"],
                    "additionalProperties": False,
                },
            }, self.set_entity_space),
        ]

    def inspect_namespace(self, fn_args: dict) -> str:
        prefix = fn_args.get("prefix") or ""
        if not prefix:
            raise FunctionCallException("prefix required")
        if self.large_kg:
            return self.inspect_sampled(prefix)
        sizes = run_sizes(self.kg_manager, SPACE_GROUPING, [prefix])
        if not sizes:
            return f"No entities with IRIs starting with {prefix}."
        named = run_rows(self.kg_manager, grouping_to_sparql_query(
            self.kg_manager, SPACE_GROUPING, [prefix], "(COUNT(DISTINCT ?s) AS ?n)",
            f"FILTER EXISTS {{ ?s {sparql_path(LABEL_PREDICATES)} ?name }}",
        ))
        named_count = int(named[0]["n"].value) if named and "n" in named[0] else 0
        types = run_rows(self.kg_manager, grouping_to_sparql_query(
            self.kg_manager, SPACE_GROUPING, [prefix], "?t (COUNT(DISTINCT ?s) AS ?n)",
            f"?s <{RDF_TYPE}> ?t", "GROUP BY ?t ORDER BY DESC(?n) LIMIT 8",
        ))
        return "\n".join([
            f"{prefix}: " + ", ".join(f"{p} {n}" for p, n in sizes.items()) + f"; {named_count} with a canon name",
            "Most frequent types: " + (", ".join(f"{short_iri(r['t'].value)} {r['n'].value}" for r in types if "t" in r) or "none"),
            "Examples:", *entity_examples(self.kg_manager, SPACE_GROUPING, [prefix], "", 8),
        ])

    def inspect_sampled(self, prefix: str) -> str:
        """inspect_namespace of a large KG, from the class samples."""
        classes: Counter[str] = Counter()
        for (other, class_iri), count in self.class_prefix_counts.items():
            if other.startswith(prefix) or prefix.startswith(other):
                classes[class_iri] += count
        examples = self.sampled_examples(prefix, 8)
        if not examples:
            return (f"No sampled entity has an IRI starting with {prefix}. (In this large knowledge graph, "
                    "entities are only sampled per class; untyped entities are not seen.)")
        return "\n".join([
            f"{prefix}: classes of the sampled entities under this prefix or its namespace (estimated members): "
            + ", ".join(f"{c} ~{n}" for c, n in classes.most_common(NUM_NAMESPACES)),
            "Examples:", *examples,
        ])

    def sampled_examples(self, prefix: str, k: int) -> list[str]:
        """Sampled entities under a prefix (large KG) with a name and their class."""
        pairs = [(entity, class_iri) for other, sampled in self.prefix_examples.items()
                 if other.startswith(prefix) or prefix.startswith(other)
                 for entity, class_iri in sampled if entity.startswith(prefix)][:k]
        if not pairs:
            return []
        names: dict[str, str] = {}
        rows = run_rows(self.kg_manager, f"SELECT ?s ?name WHERE {{ {values_block([e for e, _ in pairs])} "
                        f"?s {sparql_path(LABEL_PREDICATES)} ?name }}")
        for row in rows:
            names.setdefault(row["s"].value, row["name"].value)
        return [f"  {e}" + (f' "{clip(names[e], 50)}"' if e in names else "") + f" ({c})" for e, c in pairs]

    def has_entities(self, prefix: str) -> bool:
        if not self.large_kg:
            return bool(run_sizes(self.kg_manager, SPACE_GROUPING, [prefix]))
        if any(e.startswith(prefix) for sampled in self.prefix_examples.values() for e, _ in sampled):
            return True
        try:
            return bool(select_rows(
                self.kg_manager, f'SELECT ?s WHERE {{ ?s ?p ?o FILTER(STRSTARTS(STR(?s), "{prefix}")) }} LIMIT 1'))
        except Exception:
            return True  # cannot tell in time; trust the model

    def set_entity_space(self, fn_args: dict) -> str:
        prefixes = [prefix for prefix in fn_args.get("prefixes") or [] if prefix]
        if not prefixes:
            raise FunctionCallException("Name at least one IRI prefix")
        empty = [prefix for prefix in prefixes if not self.has_entities(prefix)]
        if empty:
            raise FunctionCallException(f"No entities under {empty}")
        self.entity_space = prefixes
        return self.input_for_llm()

    def checked_entity_space(self) -> list[str]:
        """The result: the entity space, if it names a prefix."""
        if not self.entity_space:
            raise FunctionCallException("The entity space is empty; set it with set_entity_space")
        return list(self.entity_space)


@dataclass(eq=False)
class GroupingEditor[T: (Partition, GroupDefinition)](Editor):
    """
    Partitions (stage b) or groups (stage d): definitions by membership
    patterns, tested before they are set, with sizes and distinguishing
    features. The statistics are computed on creation and after every change
    of the definitions. Subclasses say how the definitions form a Grouping.
    """
    unit: ClassVar[Literal["partition", "group"]]  # in texts for the model
    definitions: dict[str, T]  # name -> definition, in order
    tested: set[str] = field(init=False)  # membership patterns tested with test_<unit>
    sizes: dict[str, int] = field(init=False)  # counted; in a large KG on the sample
    estimates: dict[str, int] = field(init=False)  # estimated in a large KG, else counted
    predicates: dict[str, dict[str, int]] = field(init=False)  # member -> predicate -> entities
    types: dict[str, dict[str, int]] = field(init=False)  # member -> type -> entities

    def __post_init__(self) -> None:
        self.definitions = dict(self.definitions)
        self.tested = set()
        self.update_statistics()

    @abstractmethod
    def grouping(self) -> Grouping:
        """The definitions as a Grouping, for the statistics queries."""

    @abstractmethod
    def outside(self, kind: Literal["predicates", "types"], name: str) -> tuple[dict[str, int], int]:
        """Counts of predicates or types and size of the entities outside one member."""

    def names(self) -> list[str]:
        return list(self.definitions)

    def size_stats(self) -> dict:
        return size_stats(self.kg_manager, self.sizes, self.estimates)

    def size_text(self, name: str) -> str:
        return size_text(self.kg_manager, self.sizes, self.estimates, name)

    def update_statistics(self) -> None:
        grouping = self.grouping()
        try:
            self.sizes = group_sizes(self.kg_manager, grouping, self.entity_space)
            self.estimates = estimated_group_sizes(self.kg_manager, grouping, self.entity_space)
            self.predicates = {member: {predicate: c.entities for predicate, c in counts.items()}
                               for member, counts in predicate_coverage(self.kg_manager, grouping, self.entity_space).items()}
            self.types = type_coverage(self.kg_manager, grouping, self.entity_space)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e

    def describe_member(self, name: str, title: str) -> str:
        """A member with examples and the predicates and types that distinguish it."""
        size = self.sizes.get(name, 0)
        lines = [f"{title} ({self.size_text(name)}). Examples:",
                 *entity_examples(self.kg_manager, self.grouping(), self.entity_space, f'FILTER(?group = "{name}")', 3)]
        outside, size_outside = self.outside("predicates", name)
        lines.append("Predicates (share of entities with the predicate):")
        lines += format_features(self.predicates.get(name, {}), size, outside, size_outside, f"other {self.unit}s")
        if self.types.get(name):
            outside, size_outside = self.outside("types", name)
            lines.append("Types:")
            lines += format_features(self.types[name], size, outside, size_outside, f"other {self.unit}s")[:5]
        return "\n".join(lines)

    def update_definitions(self, definitions: dict[str, T]) -> None:
        """Sets the definitions and their statistics; on a failed query, nothing changes."""
        old = self.definitions
        self.definitions = definitions
        try:
            self.update_statistics()
        except FunctionCallException:
            self.definitions = old
            raise

    def known_name(self, name: str | None) -> str:
        if name is None or name not in self.definitions:
            raise FunctionCallException(f"Unknown {self.unit} {name}; {self.unit}s: {', '.join(self.names())}")
        return name

    def test_definition(self, fn_args: dict) -> str:
        membership = membership_argument(fn_args)
        try:
            size, overlap = membership_overlap(self.kg_manager, self.grouping(), self.entity_space, membership)
            inside, inside_types = member_coverage(self.kg_manager, self.entity_space, membership)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        self.tested.add(membership)
        if not size:
            return "No entity of the entity space matches."
        sample = current_sample(self.kg_manager)
        if sample is not None:
            matches = sample.sample(membership)
            estimate = (f"~{round(matches.count * matches.space_share)}" if matches.count is not None
                        else "an unknown number of")
            size_line = f"{estimate} entities of the entity space match (statistics on a sample of {size}); "
        else:
            size_line = f"{size} entities match; "
        compare = fn_args.get("compare_with")
        if compare is not None:
            self.known_name(compare)
        else:
            compare = max((member for member in overlap if member in self.sizes),
                          key=lambda member: overlap[member], default=None)
        where = f"by current {self.unit} (share of the {self.unit} that matches)"
        lines = [size_line + where + ": "
                 + ", ".join(f"{p} {share(n, self.sizes.get(p, 0))}" for p, n in sorted(overlap.items(), key=lambda x: -x[1]))]
        if compare is not None:
            rest_size = self.sizes.get(compare, 0) - overlap.get(compare, 0)
            for title, counts, ins in (("Predicates", self.predicates, inside), ("Types", self.types, inside_types)):
                outside = {k: max(0, n - ins.get(k, 0)) for k, n in counts.get(compare, {}).items()}
                lines.append(f"{title} (share of the matches vs. the rest of {compare}):")
                lines += format_features(ins, size, outside, rest_size, f"rest of {compare}")[:NUM_FEATURES if title == "Predicates" else 5]
        lines.append("Examples:")
        names = sparql_path(LABEL_PREDICATES)
        rows = run_rows(self.kg_manager, (
            f"SELECT DISTINCT ?s ?name WHERE {{ {members_query(membership, self.entity_space, self.kg_manager)} "
            f"OPTIONAL {{ ?s {names} ?name }} }} LIMIT 6"
        ))
        lines += [f"  {r['s'].value}" + (f' "{clip(r["name"].value, 50)}"' if "name" in r else "")
                  for r in rows if "s" in r]
        return "\n".join(lines)

    def checked_membership(self, fn_args: dict) -> str:
        membership = membership_argument(fn_args)
        if membership not in self.tested:
            raise FunctionCallException(f"Test the membership pattern with test_{self.unit} first (exactly this pattern)")
        return membership


@dataclass(eq=False)
class PartitionEditor(GroupingEditor[Partition]):
    """
    The exclusive partitions (an entity belongs to the first that matches)
    and the pairs of partitions whose entities do not compete.
    """
    unit = "partition"
    non_competing_pairs: list[WithinKgNonCompetingPartitionPair]
    # the subjects outside all partitions as a member of their own (small KG only: a large one is not scanned)
    mark_uncategorized: bool

    def __post_init__(self) -> None:
        self.non_competing_pairs = list(self.non_competing_pairs)
        super().__post_init__()

    @classmethod
    def from_kg_profile(cls, kg_manager: KgManager, kg_profile: KgProfile, *, create_missing: bool) -> Self:
        """
        The partitions and non-competing pairs of the profile. create_missing:
        without partitions, the kind partitions with entities and the default
        pairs (properties do not compete); otherwise a ValueError.
        """
        if not kg_profile.partitions and not create_missing:
            raise ValueError("The profile has no partitions")
        entity_space = ensure_space(kg_manager, kg_profile)
        definitions = ensure_partitions(kg_manager, kg_profile, entity_space)
        non_competing_pairs = (
            list(kg_profile.non_competing_pairs)
            if kg_profile.partitions
            else default_non_competing_pairs(definitions)
            )
        return cls(
            kg_manager=kg_manager,
            entity_space=entity_space,
            definitions=definitions,
            non_competing_pairs=non_competing_pairs,
            mark_uncategorized=not kg_profile.large,
        )

    def grouping(self) -> Grouping:
        return partitions_to_grouping(list(self.definitions.values()), mark_uncategorized_as_rest=self.mark_uncategorized)

    def outside(self, kind: Literal["predicates", "types"], name: str) -> tuple[dict[str, int], int]:
        counts = self.predicates if kind == "predicates" else self.types
        return complement(counts, self.sizes, name)

    def input_for_llm(self) -> str:
        return self.format_sizes() + "\n\n" + self.format_non_competing_pairs()

    def format_sizes(self) -> str:
        lines = [f"  {name}: {self.size_text(name)}: {{ {partition.membership_sparql} }}"
                 + (f"; {partition.features}" if partition.features else "")
                 for name, partition in self.definitions.items()]
        if self.sizes.get(UNCATEGORIZED):
            lines.append(f"  {UNCATEGORIZED}: {self.sizes[UNCATEGORIZED]} subjects match no partition")
        return "Partitions in order (an entity belongs to the first that matches):\n" + "\n".join(lines)

    def format_non_competing_pairs(self) -> str:
        if not self.non_competing_pairs:
            return "Partitions that do not compete with each other: none (all compete)."
        return "Partitions that do not compete with each other:\n" + "\n".join(
            f"  {pair.partition_a} - {pair.partition_b}" + (f": {pair.reason}" if pair.reason else "")
            for pair in self.non_competing_pairs)

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            (test_function("partition", (
                "Tests a membership pattern: how many entities of the entity space match, in which "
                "current partitions they are, which predicates and types distinguish them from the "
                "rest of the partition they overlap most (or compare_with), and examples. A pattern "
                "must be tested before it can be set.")), self.test_definition),
            ({
                "name": "set_partition",
                "description": (
                    "Defines a partition by a tested membership pattern, or redefines an existing one "
                    "(it keeps its place). A new partition is placed before the partition named in "
                    "before, otherwise first. Shows the partition sizes afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A short lower case noun for the kind."},
                        "membership": MEMBERSHIP,
                        "features": {"type": "string", "description": "What distinguishes its entities."},
                        "before": {"type": "string"},
                    },
                    "required": ["name", "membership", "features"],
                    "additionalProperties": False,
                },
            }, self.set_partition),
            ({
                "name": "remove_partition",
                "description": "Removes a partition; its entities fall to the next matching partition.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }, self.remove_partition),
            ({
                "name": "set_competition",
                "description": (
                    "Sets whether the entities of two partitions compete with each other: if not, an "
                    "entity of one is never a candidate or a rival for an entity of the other."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "partition_a": {"type": "string"}, "partition_b": {"type": "string"},
                        "compete": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["partition_a", "partition_b", "compete", "reason"],
                    "additionalProperties": False,
                },
            }, self.set_competition),
        ]

    def set_partition(self, fn_args: dict) -> str:
        name = valid_name(fn_args.get("name") or "", self.unit)
        membership = self.checked_membership(fn_args)
        before = fn_args.get("before")
        if before is not None and self.known_name(before) == name:
            raise FunctionCallException("A partition cannot be placed before itself")
        definition = Partition(name=name, membership_sparql=membership, features=fn_args.get("features") or "")
        items = [(other, partition) for other, partition in self.definitions.items() if other != name]
        if name in self.definitions:
            index = list(self.definitions).index(name)
        elif before is not None:
            index = [other for other, _ in items].index(before)
        else:
            index = 0
        items.insert(index, (name, definition))
        self.update_definitions(dict(items))
        return f"Partition {name} set.\n{self.format_sizes()}"

    def remove_partition(self, fn_args: dict) -> str:
        name = self.known_name(fn_args.get("name"))
        if len(self.definitions) == 1:
            raise FunctionCallException("The last partition cannot be removed")
        self.update_definitions({other: partition for other, partition in self.definitions.items() if other != name})
        self.non_competing_pairs = [pair for pair in self.non_competing_pairs if name not in pair.partitions()]
        return f"Partition {name} removed.\n{self.format_sizes()}"

    def set_competition(self, fn_args: dict) -> str:
        partition_a = self.known_name(fn_args.get("partition_a"))
        partition_b = self.known_name(fn_args.get("partition_b"))
        if partition_a == partition_b:
            raise FunctionCallException("Entities of one partition always compete with each other")
        self.non_competing_pairs = [pair for pair in self.non_competing_pairs
                                    if pair.partitions() != {partition_a, partition_b}]
        if not fn_args.get("compete"):
            self.non_competing_pairs.append(WithinKgNonCompetingPartitionPair(
                partition_a=partition_a, partition_b=partition_b, reason=fn_args.get("reason") or ""))
        return self.format_non_competing_pairs()

    def with_partitions(self, profile: KgProfile) -> KgProfile:
        """The profile with the entity space, the partitions that have entities and their non-competing pairs."""
        kept = [partition for name, partition in self.definitions.items() if self.sizes.get(name)]
        kept_names = {partition.name for partition in kept}
        non_competing_pairs = [pair for pair in self.non_competing_pairs if pair.partitions() <= kept_names]
        return profile.model_copy(update={
            "entity_space": self.entity_space, "partitions": kept, "non_competing_pairs": non_competing_pairs}, deep=True)

    def check_partitions(self) -> None:
        if not any(self.sizes.get(name) for name in self.definitions):
            raise FunctionCallException("No partition has entities")

    def empty_partition_names(self) -> list[str]:
        """Partitions without entities: with_partitions drops them."""
        return [name for name in self.definitions if not self.sizes.get(name)]


@dataclass(eq=False)
class GroupEditor(GroupingEditor[GroupDefinition]):
    """
    The overlapping groups and their exclusion from the alignment (an entity
    in an excluded group is not aligned), and the local name type of a group.
    """
    unit = "group"
    exclusions: dict[str, str]  # name of an excluded group -> reason
    # set by set_local_name_type (revision); otherwise the group's stays
    local_name_types: dict[str, LocalNameType] = field(default_factory=dict)
    all_size: int = field(init=False)  # entities in any group, to compare a group with the rest
    all_predicates: dict[str, int] = field(init=False)
    all_types: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.exclusions = dict(self.exclusions)
        self.local_name_types = dict(self.local_name_types)
        super().__post_init__()

    @classmethod
    def from_kg_profile(cls, kg_manager: KgManager, kg_profile: KgProfile, *, create_missing: bool) -> Self:
        """
        The groups of the profile with their exclusions. create_missing:
        without groups, one per partition (and in a large KG one per large
        class); otherwise a ValueError.
        """
        if not kg_profile.groups and not create_missing:
            raise ValueError("The profile has no groups")
        entity_space = ensure_space(kg_manager, kg_profile)
        return cls(
            kg_manager=kg_manager,
            entity_space=entity_space,
            definitions=ensure_groups(kg_manager, kg_profile, entity_space),
            exclusions={group.name: group.exclusion_reason for group in kg_profile.groups if group.excluded},
        )

    def grouping(self) -> Grouping:
        return groups_to_grouping(list(self.definitions.values()))

    def update_statistics(self) -> None:
        super().update_statistics()
        merged = self.grouping().merged(ALL)
        try:
            self.all_size = group_sizes(self.kg_manager, merged, self.entity_space).get(ALL, 0)
            self.all_predicates = {predicate: c.entities for predicate, c in
                                   predicate_coverage(self.kg_manager, merged, self.entity_space).get(ALL, {}).items()}
            self.all_types = type_coverage(self.kg_manager, merged, self.entity_space).get(ALL, {})
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e

    def outside(self, kind: Literal["predicates", "types"], name: str) -> tuple[dict[str, int], int]:
        if kind == "predicates":
            return rest(self.all_predicates, self.all_size, self.predicates.get(name, {}), self.sizes.get(name, 0))
        return rest(self.all_types, self.all_size, self.types.get(name, {}), self.sizes.get(name, 0))

    def input_for_llm(self) -> str:
        return self.format_sizes()

    def format_sizes(self) -> str:
        lines = []
        for name, definition in self.definitions.items():
            notes = [definition.origin.value] + ([f"part of {definition.parent}"] if definition.parent else [])
            excluded = f" [excluded: {self.exclusions[name]}]" if name in self.exclusions else ""
            lines.append(f"  {name}: {self.size_text(name)} ({', '.join(notes)}): {{ {definition.membership_sparql} }}"
                         + (f"; {definition.features}" if definition.features else "") + excluded)
        return f"Groups ({self.all_size} entities in any group; groups may overlap):\n" + "\n".join(lines)

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            (test_function("group", (
                "Tests a membership pattern: how many entities of the entity space match, how much of "
                "each current group they cover, which predicates and types distinguish them from the "
                "rest of the group they overlap most (or compare_with), and examples. A pattern must "
                "be tested before it can be set.")), self.test_definition),
            ({
                "name": "set_group",
                "description": (
                    "Defines a group by a tested membership pattern, or redefines an existing one. "
                    "Groups may overlap. Shows the group sizes afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A short lower case noun for the group."},
                        "membership": MEMBERSHIP,
                        "features": {"type": "string",
                                     "description": "What distinguishes its entities, so that they can be recognized."},
                        "parent": {"type": "string", "description": "A group it is part of, if one."},
                    },
                    "required": ["name", "membership", "features"],
                    "additionalProperties": False,
                },
            }, self.set_group),
            ({
                "name": "exclude_group",
                "description": ("Excludes a group from the alignment (entities that would not be matched for this "
                                "task), or includes it again. An entity in an excluded group is not aligned, even "
                                "if it is in other groups too."),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "reason": {"type": "string"},
                                   "excluded": {"type": "boolean", "default": True}},
                    "required": ["name", "reason"],
                    "additionalProperties": False,
                },
            }, self.exclude_group),
            ({
                "name": "remove_group",
                "description": ("Removes a group. Its entities stay in their other groups; those in no other group "
                                "are no longer aligned."),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }, self.remove_group),
        ]

    def local_name_type_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        """Setting the local name type of a group; stage e sets it in finish instead."""
        return [({
            "name": "set_local_name_type",
            "description": "Sets whether the local names of a group's IRIs are names (label), identifiers (code) or mixed (other).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                               "local_name_type": {"type": "string", "enum": [t.value for t in LocalNameType]}},
                "required": ["name", "local_name_type"],
                "additionalProperties": False,
            },
        }, self.set_local_name_type)]

    def set_group(self, fn_args: dict) -> str:
        name = valid_name(fn_args.get("name") or "", self.unit)
        membership = self.checked_membership(fn_args)
        parent = fn_args.get("parent")
        if parent is not None and self.known_name(parent) == name:
            raise FunctionCallException("A group cannot be part of itself")
        definition = GroupDefinition(name=name, membership_sparql=membership, parent=parent,
                                     features=fn_args.get("features") or "")
        self.update_definitions({**self.definitions, name: definition})
        return f"Group {name} set.\n{self.format_sizes()}"

    def excluded_share(self, exclusions: dict[str, str]) -> float:
        """Share of the entities in any group that are in an excluded group."""
        if not exclusions or not self.all_size:
            return 0.0
        grouping = Grouping(tuple(("excluded", self.definitions[name].membership_sparql) for name in exclusions),
                            exclusive=True)
        return group_sizes(self.kg_manager, grouping, self.entity_space).get("excluded", 0) / self.all_size

    def exclude_group(self, fn_args: dict) -> str:
        name = self.known_name(fn_args.get("name"))
        excluded = fn_args.get("excluded", True)
        exclusions = {other: reason for other, reason in self.exclusions.items() if other != name}
        if excluded:
            exclusions[name] = fn_args.get("reason") or ""
        if set(exclusions) == set(self.definitions):
            raise FunctionCallException("Not every group can be excluded: nothing would be aligned. To exclude "
                                        "part of a group, define that part as a group of its own and exclude it.")
        self.exclusions = exclusions
        try:
            excluded_share = self.excluded_share(exclusions)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        result = f"Group {name} {'excluded' if excluded else 'included'}; {excluded_share:.0%} of the entities in groups are excluded now."
        if excluded and excluded_share > LARGE_EXCLUSION_SHARE:
            result += (" This is a large part. Exclude only entities that would not be matched for this task; "
                       "to exclude part of a group, define that part as a group of its own and exclude it.")
        return f"{result}\n{self.format_sizes()}"

    def remove_group(self, fn_args: dict) -> str:
        name = self.known_name(fn_args.get("name"))
        if len(self.definitions) == 1:
            raise FunctionCallException("The last group cannot be removed")
        membership = self.definitions[name].membership_sparql
        remaining = {other: definition for other, definition in self.definitions.items() if other != name}
        try:
            size, overlap = membership_overlap(self.kg_manager, groups_to_grouping(list(remaining.values())),
                                               self.entity_space, membership)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        self.update_definitions(remaining)
        self.exclusions.pop(name, None)
        self.local_name_types.pop(name, None)
        alone = overlap.get(UNCATEGORIZED, 0)
        return (f"Group {name} removed; {share(alone, size)} of its entities are in no other group and are no "
                f"longer aligned.\n{self.format_sizes()}")

    def set_local_name_type(self, fn_args: dict) -> str:
        name = self.known_name(fn_args.get("name"))
        try:
            name_type = LocalNameType(fn_args.get("local_name_type"))
        except ValueError as e:
            raise FunctionCallException(f"Invalid local_name_type: {e}") from e
        self.local_name_types[name] = name_type
        return f"Local names of group {name}: {name_type.value}"

    def with_groups(self, profile: KgProfile) -> KgProfile:
        """The profile with the entity space and the groups that have entities; the
        profiles (roles, local name type) of kept groups stay."""
        old = {group.name: group for group in profile.groups}
        groups = []
        for name, definition in self.definitions.items():
            if not self.sizes.get(name):
                continue
            update: dict[str, Any] = {"definition": definition, "excluded": name in self.exclusions,
                                      "exclusion_reason": self.exclusions.get(name, "")}
            if name in self.local_name_types:
                update["local_name_type"] = self.local_name_types[name]
            groups.append(old[name].model_copy(update=update) if name in old else GroupProfile(**update))
        return profile.model_copy(update={"entity_space": self.entity_space, "groups": groups}, deep=True)

    def check_groups(self) -> None:
        kept = [name for name in self.definitions if self.sizes.get(name)]
        if not kept:
            raise FunctionCallException("No group has entities")
        if all(name in self.exclusions for name in kept):
            raise FunctionCallException("Every group is excluded: nothing would be aligned")

    def empty_group_names(self) -> list[str]:
        """Groups without entities: with_groups drops them."""
        return [name for name in self.definitions if not self.sizes.get(name)]


@dataclass(eq=False)
class RoleEditor(Editor):
    """
    Editing of roles (KG-wide in stage c, of one group in stage e), with the
    coverage of each element over the partitions or groups in scope. The
    coverage is counted once per element and cached; the task may hand in
    counts it has already made (usage, sides, counted), the rest is counted
    on creation. Created by from_kg_profile (KG-wide) or from_group_profile
    (one group: its effective roles, and its deviation as the result).
    """
    unit: Literal["partition", "group"]  # in texts for the model
    roles: Roles                         # the working roles, changed by the functions
    grouping: Grouping
    sizes: dict[str, int]
    scope: list[str]                     # partitions or groups whose coverage is shown
    # where an element comes from, shown as a mark (stage e: [kg] or [group])
    origin: Callable[[ElementKey], str] | None = None
    usage: dict[str, Usage] = field(default_factory=dict)  # partition or group -> element -> entities
    sides: dict[str, Sides] = field(default_factory=dict)  # partition or group -> relation -> (as ?s, as ?o)
    counted: set[ElementKey] = field(default_factory=set)  # elements counted, also those used nowhere
    # one group: its roles derived from the KG-wide ones (the base of its deviation); KG-wide: None
    derived_roles: Roles | None = None

    def __post_init__(self) -> None:
        # own copies: the editor changes them, the caller's objects (e.g. the default) stay as they are
        self.roles = self.roles.model_copy(deep=True)
        self.usage = {member: dict(counts) for member, counts in self.usage.items()}
        self.sides = {member: dict(counts) for member, counts in self.sides.items()}
        self.counted = set(self.counted)
        self.update_usage(self.roles)

    @classmethod
    def from_kg_profile(cls, kg_manager: KgManager, kg_profile: KgProfile, *, create_missing: bool) -> Self:
        """
        The KG-wide roles of the profile, with their coverage per partition.
        create_missing: without partitions, the kind partitions; without
        roles, the canon elements with coverage; otherwise a ValueError.
        """
        entity_space = ensure_space(kg_manager, kg_profile)
        partitions = list(kg_profile.partitions)
        if not partitions:
            if not create_missing:
                raise ValueError("The profile has no partitions")
            partitions = list(default_partitions(kg_manager, kg_profile, entity_space).values())
        grouping = partitions_to_grouping(partitions, mark_uncategorized_as_rest=False)
        partition_names = [partition.name for partition in partitions]
        sizes = group_sizes(kg_manager, grouping, entity_space)

        roles = kg_profile.roles
        usage: dict[str, Usage] = {}
        sides: dict[str, Sides] = {}
        counted: set[ElementKey] = set()
        if not role_elements(roles):
            if not create_missing:
                raise ValueError("The profile has no roles")
            # the canon is counted for the default: the editor counts only what is added later
            usage = element_usage(kg_manager, grouping, entity_space, canon_roles(), sides)
            roles = default_roles(usage, sizes, partition_names)
            counted = set(role_elements(canon_roles()))
        return cls(
            kg_manager=kg_manager,
            entity_space=entity_space,
            unit="partition",
            roles=roles,
            grouping=grouping,
            sizes=sizes,
            scope=partition_names,
            usage=usage,
            sides=sides,
            counted=counted,
        )

    @classmethod
    def from_group_profile(cls, kg_manager: KgManager, kg_profile: KgProfile, group_name: str) -> Self:
        """
        The effective roles of one group (the KG-wide roles used in it, minus
        its removed, plus its added), with the coverage in the group only
        (without the entities of excluded groups); marks show whether an
        element is the group's own or KG-wide.
        """
        if group_name not in kg_profile.group_names():
            raise ValueError(f"Unknown group {group_name}; groups: {', '.join(kg_profile.group_names())}")
        entity_space = ensure_space(kg_manager, kg_profile)
        grouping = single_grouping(kg_profile, group_name)
        group_profile = kg_profile.get_group(group_name)
        sides: dict[str, Sides] = {}
        usage = element_usage(
            kg_manager, grouping, entity_space,
            roles_from_elements(role_elements(kg_profile.roles) | role_elements(group_profile.added)), sides,
            )
        group_usage = usage.get(group_name, {})
        effective = effective_roles(kg_profile.roles, group_profile, group_usage)
        added_keys = set(role_elements(group_profile.added))

        def origin(key: ElementKey) -> str:
            return "group" if key in added_keys else "kg"

        return cls(
            kg_manager=kg_manager,
            entity_space=entity_space,
            unit="group",
            roles=effective,
            grouping=grouping,
            sizes=group_sizes(kg_manager, grouping, entity_space),
            scope=[group_name],
            origin=origin,
            usage=usage,
            sides=sides,
            counted=set(role_elements(effective)),
            derived_roles=derived_roles(kg_profile.roles, group_usage),
        )

    def input_for_llm(self) -> str:
        return self.format_roles(self.roles, self.origin)

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        pattern_schema = inline_schema(PredicatePattern)
        member = {"type": "string", "enum": self.scope}
        return [
            ({
                "name": "try_pattern",
                "description": (
                    f"Tests a predicate pattern: per {self.unit}, how many entities are ?s, how many "
                    "distinct ?o and literal values it has, and examples with names. Use it for "
                    "attributes (names, codes: ?o is a literal) and structure relations (?o is an entity)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"predicate_pattern": pattern_schema,
                                   "show_examples_from": member | {"description": f"The {self.unit} to show example matches from; default: the {self.unit} with the most entities as ?s."}},
                    "required": ["predicate_pattern"],
                    "additionalProperties": False,
                },
            }, self.try_pattern),
            ({
                "name": "edit_role",
                "description": (
                    "Adds predicates or patterns to a role, or removes them. Shows the role and the "
                    "coverage of each element afterwards. Roles: "
                    + "; ".join(f"{k}: {v}" for k, v in ROLE_DESCRIPTIONS.items())
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": list(ROLE_FIELDS)},
                        "add": pattern_schema | {"description": "Predicates and patterns to add."},
                        "remove_predicates": {"type": "array", "items": {"type": "string"}},
                        "remove_patterns": {"type": "array", "items": {"type": "string"},
                                            "description": "Patterns to remove, exactly as listed."},
                    },
                    "required": ["role"],
                    "additionalProperties": False,
                },
            }, self.edit_role),
            ({
                "name": "set_structure_relation",
                "description": (
                    "Adds a structure relation or replaces the one of the same name. Shows its coverage."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"relation": inline_schema(StructureRelation, drop={"source"})},
                    "required": ["relation"],
                    "additionalProperties": False,
                },
            }, self.set_structure_relation),
            ({
                "name": "remove_structure_relation",
                "description": "Removes a structure relation by its name.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }, self.remove_structure_relation),
        ]

    # --- coverage ---
    def update_usage(self, roles: Roles) -> None:
        """Usage of the elements of roles not counted yet."""
        known = {key for counts in self.usage.values() for key in counts} | self.counted
        new = {k: v for k, v in role_elements(roles).items() if k not in known}
        if not new:
            return
        sides: dict[str, Sides] = {}
        try:
            usage = element_usage(self.kg_manager, self.grouping, self.entity_space, roles_from_elements(new), sides)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        for member, counts in usage.items():
            self.usage.setdefault(member, {}).update(counts)
        for member, counts in sides.items():
            self.sides.setdefault(member, {}).update(counts)
        self.counted |= set(new)

    def format_coverage(self, key: ElementKey) -> str:
        counts = [f"{p} {share(self.usage.get(p, {}).get(key, 0), self.sizes.get(p, 0))}"
                  for p in self.scope]
        return "; ".join(counts)

    def format_field(self, field: str, pattern: PredicatePattern | None,
                     origin: Callable[[ElementKey], str] | None = None) -> list[str]:
        if pattern is None:
            return [f"{field}: none"]
        mark = lambda key: f" [{origin(key)}]" if origin else ""
        return [f"{field}:"] + [f"  {key[1]} {key[2]}{mark(key)}: {self.format_coverage(key)}"
                                for key in role_elements(Roles(**{field: pattern}))]

    def format_relation(self, relation: StructureRelation, origin: Callable[[ElementKey], str] | None = None) -> str:
        key = ("structure", "relation", relation.name)
        mark = f" [{origin(key)}]" if origin else ""
        side = lambda i: "; ".join(f"{p} {share(self.sides.get(p, {}).get(key, (0, 0))[i], self.sizes.get(p, 0))}"
                                   for p in self.scope)
        return f"  {describe_structure_relation(relation)}{mark}\n    as ?s: {side(0)}\n    as ?o: {side(1)}"

    def format_roles(self, roles: Roles, origin: Callable[[ElementKey], str] | None = None) -> str:
        """Each element with its coverage per partition or group in scope."""
        self.update_usage(roles)
        lines = [line for field in ROLE_FIELDS for line in self.format_field(field, getattr(roles, field), origin)]
        lines.append("structure relations:" if roles.structure else "structure relations: none")
        lines += [self.format_relation(relation, origin) for relation in roles.structure]
        return "\n".join(lines)

    # --- functions ---
    def try_pattern(self, fn_args: dict) -> str:
        try:
            pattern = PredicatePattern.model_validate(fn_args.get("predicate_pattern") or {})
        except ValidationError as e:
            raise FunctionCallException(f"Invalid predicate_pattern: {e}") from e
        member = fn_args.get("show_examples_from")
        if member is not None and member not in self.scope:
            raise FunctionCallException(f"Unknown {self.unit} {member}; known: {', '.join(self.scope)}")
        rows = run_rows(self.kg_manager, grouping_to_sparql_query(
            self.kg_manager, self.grouping, self.entity_space,
            "?group (COUNT(DISTINCT ?s) AS ?subjects) (COUNT(DISTINCT ?o) AS ?objects) "
            "(SUM(IF(isLiteral(?o), 1, 0)) AS ?literals) (COUNT(*) AS ?n)",
            "", "GROUP BY ?group", pattern.branches_for_union(),
        ))
        counts = {row["group"].value: row for row in rows if "group" in row}
        value = lambda row, key: int(row[key].value) if key in row else 0
        lines = [f"Pattern: {describe_predicate_pattern(pattern)}"]
        for p in self.scope:
            row = counts.get(p, {})
            lines.append(f"{self.unit.capitalize()} {p}: {share(value(row, 'subjects'), self.sizes.get(p, 0))} entities are ?s; "
                         f"{value(row, 'n')} matches, {value(row, 'objects')} distinct ?o, "
                         f"{value(row, 'literals')} literal matches.")
        if member is None:
            in_scope = [p for p in self.scope if value(counts.get(p, {}), "subjects")]
            member = max(in_scope, key=lambda p: value(counts[p], "subjects"), default=None)
        if member is not None and value(counts.get(member, {}), "subjects"):
            lines.append(f"Examples in {member} (?s -> ?o, with names in quotes):")
            lines += self.format_examples(member, pattern.branches_for_union())
        return "\n".join(lines)

    def format_examples(self, member: str, parts: list[str]) -> list[str]:
        """Example matches with a name of ?s and of ?o (codes alone say little)."""
        names = self.roles.names.predicates if self.roles.names else []
        names_path = sparql_path(list(dict.fromkeys(names + LABEL_PREDICATES)))
        rows = run_rows(self.kg_manager, grouping_to_sparql_query(
            self.kg_manager, self.grouping, self.entity_space, "?s ?o ?s_name ?o_name",
            f'FILTER(?group = "{member}")\n'
            f"OPTIONAL {{ ?s {names_path} ?s_name }}\nOPTIONAL {{ ?o {names_path} ?o_name }}",
            f"LIMIT {NUM_EXAMPLES * 4}", parts,
        ))
        seen, lines = set(), []
        for row in rows:
            if "s" not in row or "o" not in row or (row["s"].value, row["o"].value) in seen:
                continue
            seen.add((row["s"].value, row["o"].value))
            named = lambda var: row[var].value + (f' "{clip(row[var + "_name"].value, 40)}"' if var + "_name" in row else "")
            lines.append(f"  {named('s')} -> {clip(named('o'), MAX_VALUE_CHARS)}")
            if len(lines) == NUM_EXAMPLES:
                break
        return lines

    def edit_role(self, fn_args: dict) -> str:
        role = fn_args.get("role")
        if role not in ROLE_FIELDS:
            raise FunctionCallException(f"Unknown role {role}; roles: {', '.join(ROLE_FIELDS)}")
        current: PredicatePattern | None = getattr(self.roles, role)
        predicates = list(current.predicates) if current else []
        patterns = list(current.patterns) if current else []
        missing = [p for p in fn_args.get("remove_predicates") or [] if p not in predicates]
        missing += [p for p in fn_args.get("remove_patterns") or [] if p not in patterns]
        if missing:
            raise FunctionCallException(
                f"Not in role {role}: {missing}; it has: {describe_predicate_pattern(current) if current else 'nothing'}")
        predicates = [p for p in predicates if p not in (fn_args.get("remove_predicates") or [])]
        patterns = [p for p in patterns if p not in (fn_args.get("remove_patterns") or [])]
        if fn_args.get("add"):
            try:
                add = PredicatePattern.model_validate(fn_args["add"])
            except ValidationError as e:
                raise FunctionCallException(f"Invalid add: {e}") from e
            predicates += [p for p in add.predicates if p not in predicates]
            patterns += [p for p in add.patterns if p not in patterns]
        new = PredicatePattern(predicates=predicates, patterns=patterns) if predicates or patterns else None
        self.update_usage(Roles(**{role: new}))  # may raise on SPARQL errors; then nothing changes
        self.roles = self.roles.model_copy(update={role: new})
        return f"Role now (coverage as ?s per {self.unit}):\n" + "\n".join(self.format_field(role, new))

    def set_structure_relation(self, fn_args: dict) -> str:
        try:
            relation = StructureRelation.model_validate(fn_args.get("relation") or {})
        except ValidationError as e:
            raise FunctionCallException(f"Invalid relation: {e}") from e
        relation.source = "llm"
        key = ("structure", "relation", relation.name)
        # a changed pattern must be counted again
        for counts in self.usage.values():
            counts.pop(key, None)
        self.counted.discard(key)
        self.update_usage(Roles(structure=[relation]))
        text = self.format_relation(relation)
        structure = [r for r in self.roles.structure if r.name != relation.name]
        replaced = len(structure) < len(self.roles.structure)
        self.roles = self.roles.model_copy(update={"structure": structure + [relation]})
        problems = check_structure(self.roles.structure)
        result = f"Structure relation {relation.name} {'replaced' if replaced else 'added'}:\n{text}"
        if problems:
            result += "\nTo resolve before finish: " + "; ".join(problems)
        return result

    def remove_structure_relation(self, fn_args: dict) -> str:
        name = fn_args.get("name")
        names = [r.name for r in self.roles.structure]
        if name not in names:
            raise FunctionCallException(f"No structure relation {name}; relations: {', '.join(names) or 'none'}")
        self.roles = self.roles.model_copy(update={"structure": [r for r in self.roles.structure if r.name != name]})
        return f"Structure relation {name} removed; relations: {', '.join(r.name for r in self.roles.structure) or 'none'}"

    def check_roles(self) -> None:
        """Raises if the structure relations of the working roles are inconsistent."""
        problems = check_structure(self.roles.structure)
        if problems:
            raise FunctionCallException("Structure relations: " + "; ".join(problems))

    def checked_roles(self) -> Roles:
        """The result: the working roles, if their structure relations are consistent."""
        self.check_roles()
        return self.roles.model_copy(deep=True)

    def group_deviation(self) -> tuple[Roles, Roles]:
        """
        One group: (added, removed) of the working roles against the roles
        derived from the KG-wide ones. Unchecked (also for the working state):
        call check_roles before taking it as a result.
        """
        if self.derived_roles is None:
            raise ValueError("Only the roles of a group (from_group_profile) have a deviation")
        return diff_roles(self.derived_roles, self.roles)


# what a model writes for "no group" in a group field
NO_GROUP_VALUES = frozenset({"", "null", "none"})


def group_reference(value: Any) -> str | None:
    """The group (or partition) a finding or flag names; None for no group,
    also if the model wrote one as text (e.g. "null")."""
    if value is None or (isinstance(value, str) and value.strip().lower() in NO_GROUP_VALUES):
        return None
    return value


def next_flag_id(stage: Stage, existing: int) -> str:
    """Id of a new flag of a stage, after the existing flags of the profile."""
    return f"{stage.value}-{existing + 1}"


def format_flag(flag: Flag) -> str:
    """A flag for the input: id, status, who raised it about what, and its resolution."""
    about = flag.concerns_stage.value + (f", {flag.concerns_group}" if flag.concerns_group else "")
    text = f"{flag.id} [{flag.status.value}; {flag.raised_by_stage.value} on {about}] {flag.content}"
    if flag.resolution_description:
        text += f" -> {flag.resolution_description}"
    return text


class ProfileStageTask(GraspTask):
    """
    One stage of the KG profile. Input: a dict with kg (name of a configured
    KG; may be omitted if exactly one KG is configured), profile (output
    profile of the stage before; omitted for the first stage) and, for the
    group stage, group.
    """
    stage: Stage
    prompt_file: str
    kg_manager: KgManager
    input_profile: KgProfile   # as received
    default: KgProfile         # with the stage's default; output without finish
    profile: KgProfile         # with the stage's result after finish
    finished: bool
    rationale: str | None
    stats: dict[str, Any]

    # --- stage interface ---

    @abstractmethod
    def prepare(self, input: dict) -> str:
        """
        Computes default and statistics, sets self.default, 
        returns the stage's input text.
        """

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return []

    def finish_properties(self) -> dict[str, dict]:
        return {}

    def finish_required(self) -> list[str]:
        return []

    @abstractmethod
    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        """(profile with the stage's result, summary); raises FunctionCallException on invalid arguments."""

    def stage_rules(self) -> list[str]:
        return []

    # --- GraspTask ---

    def setup(self, input: Any) -> str:
        if not isinstance(input, dict):
            input = {"kg": input} if input else {}
        kg = input.get("kg")
        if kg:
            self.kg_manager, _ = find_manager(self.managers, kg)
        elif len(self.managers) == 1:
            self.kg_manager = self.managers[0]
        else:
            raise ValueError(
                "Input must name the kg unless exactly one KG is configured"
                )
        profile = input.get("profile")
        self.input_profile = (
            KgProfile.model_validate(profile) if profile
            else KgProfile(kg_name=self.kg_manager.kg)
            )
        if self.input_profile.kg_name != self.kg_manager.kg:
            raise ValueError(
                f"Profile of {self.input_profile.kg_name} given for {self.kg_manager.kg}"
                )
        self.finished, self.rationale, self.stats = False, None, {}
        # a large KG is profiled on samples of the partitions
        # and groups (large_kg_sampling); the space stage
        # decides whether the KG is large
        large = self.input_profile.large and self.stage != Stage.SPACE
        use_sample(
            self.kg_manager,
            KgSample(
                self.kg_manager, tuple(self.input_profile.entity_space)
                ) if large else None
            )
        body = self.prepare(input)
        self.profile = self.default.model_copy(deep=True)
        head = f"Knowledge graph: {self.kg_manager.kg}" + (
            f" ({self.kg_manager.description})" if self.kg_manager.description else "")
        if large:
            head += "\n\n" + LARGE_KG_NOTE
        return "\n\n".join(
            part for part in (
                head, self.format_earlier_results(), body
                ) if part
            )

    def format_earlier_results(self) -> str:
        profile = self.input_profile
        parts = []
        notes = describe_profile(profile, findings=False)
        if notes:
            parts.append(
                format_section("Results of earlier stages",
                               format_enumerate(notes))
                )
        if profile.findings:
            findings = [f"[{f.stage.value}{', ' + f.group if f.group else ''}] {f.text}"
                        for f in profile.findings]
            parts.append(format_section(
                "Observations of earlier stages",
                "Unverified notes of earlier stages. Rely on them only as far as you can confirm "
                "them; if one contradicts what you see, report it with a flag in finish.\n"
                + format_enumerate(findings),
            ))
        if profile.flags:
            parts.append(format_section("Flags raised so far", format_enumerate(
                [format_flag(flag) for flag in profile.flags])))
        return "\n\n".join(parts)

    def system_information(self) -> str:
        info = load_prompt("profile_common.txt") + "\n\n" + load_prompt(self.prompt_file)
        task_instructions = load_task_instructions(self.config)
        if task_instructions:
            info += "\n\n" + format_section("Task-specific instructions", task_instructions)
        return info

    def rules(self) -> list[str]:
        return self.stage_rules() + [
            "Inspect the knowledge graph with the SPARQL functions before you change anything.",
            "You can execute multiple tool calls in one step. Prefer several tool calls and "
            "SPARQL queries per step over single calls.",
            "Call finish exactly once, at the end.",
            f"You have at most {self.config.max_steps} steps; one step is one response of yours, "
            "with all tool calls in it. If you run out of steps without calling finish, the "
            "default of this stage is used.",
        ]

    def earlier_stages(self) -> list[Stage]:
        return STAGES[:STAGES.index(self.stage)]

    def about_description(self) -> str:
        return "Group (or partition) it is about, if one."

    def refers_to_groups(self) -> bool:
        """Whether findings and flags of the stage can name a group or partition
        (not before there are any)."""
        return True

    def finish_definition(self) -> dict:
        group_property = ({"group": {"type": "string", "description": self.about_description()}}
                          if self.refers_to_groups() else {})
        properties = {
            "rationale": {"type": "string",
                          "description": "What you changed and why, and what remains uncertain."},
            "findings": {
                "type": "array",
                "description": (
                    f"At most {MAX_FINDINGS} short observations for later stages and for matching, the "
                    "most useful ones (one fact each, e.g. where synonyms are stored or which entities "
                    "only annotate others); omit what the profile itself already says."
                ),
                "maxItems": MAX_FINDINGS,
                "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}} | group_property,
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
        if self.earlier_stages():
            properties["flags"] = {
                "type": "array",
                "description": "Objections to results or observations of earlier stages that you found wrong.",
                "items": {
                    "type": "object",
                    "properties": {
                        "concerns": {"type": "string", "enum": [s.value for s in self.earlier_stages()]},
                        "text": {"type": "string"},
                    } | group_property,
                    "required": ["concerns", "text"],
                    "additionalProperties": False,
                },
            }
        properties |= self.finish_properties()
        return {
            "name": FINISH,
            "description": "Submits the result of this stage.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["rationale"] + self.finish_required(),
                "additionalProperties": False,
            },
        }

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.stage_functions() + [(self.finish_definition(), self.finish)]

    def function_definitions(self) -> list[dict]:
        return [definition for definition, _ in self.functions()]

    def final_step_functions(self) -> list[dict] | None:
        return [self.finish_definition()]

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        handlers = {definition["name"]: handler for definition, handler in self.functions()}
        if fn_name not in handlers:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return handlers[fn_name](fn_args)

    def finish(self, fn_args: dict) -> str:
        if len(fn_args.get("findings") or []) > MAX_FINDINGS:
            raise FunctionCallException(f"At most {MAX_FINDINGS} findings; keep the most useful ones for matching")
        profile, summary = self.apply_finish(fn_args)
        try:
            findings = [Finding(stage=self.stage, text=f.get("text"), group=group_reference(f.get("group")))
                        for f in fn_args.get("findings") or []]
            flags = [
                Flag(id=next_flag_id(self.stage, len(profile.flags) + index), raised_by_stage=self.stage,
                     concerns_stage=f.get("concerns"), content=f.get("text"),
                     concerns_group=group_reference(f.get("group")))
                for index, f in enumerate(fn_args.get("flags") or [])
                ]
        except (ValidationError, TypeError) as e:
            raise FunctionCallException(f"Invalid findings or flags: {e}") from e
        known = set(profile.partition_names()) | set(profile.group_names())
        about = [finding.group for finding in findings] + [flag.concerns_group for flag in flags]
        unknown = {name for name in about if name and name not in known}
        if unknown:
            raise FunctionCallException(
                f"Unknown groups or partitions {sorted(unknown)}; known: {', '.join(known) or 'none'}")
        if any(flag.concerns_stage not in self.earlier_stages() for flag in flags):
            raise FunctionCallException("A flag must concern an earlier stage")
        profile.findings += findings
        profile.flags += flags
        self.profile, self.finished, self.rationale = profile, True, fn_args.get("rationale")
        return "Submitted. " + summary

    def done(self, fn_name: str) -> bool:
        return fn_name == FINISH and self.finished

    def output(self, messages: list[Message]) -> dict[str, Any]:
        return ProfileStageOutput(
            kg_name=self.kg_manager.kg,
            stage=self.stage,
            finished=self.finished,
            rationale=self.rationale,
            profile=(self.profile if self.finished else self.default),
            default=self.default,
            stats=self.stats
        ).model_dump(mode="json")
