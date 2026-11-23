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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Callable, ClassVar, Literal, Self, Sequence, TypedDict

from pydantic import ValidationError

from grasp.configs import GraspConfig
from grasp.functions import ExecutionResult, execute_sparql, find_manager
from grasp.manager import KgManager
from grasp.model import Message
from grasp.sparql.types import Binding
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.profile import (
    PredicatePattern,
    RoleElement,
    RoleElementKind,
    RoleElementUsage,
    StructureRelation,
    ROLE_DESCRIPTIONS,
    ROLE_FIELDS,
    STAGES,
    Finding,
    Flag,
    GroupDefinition,
    GroupOrigin,
    GroupProfile,
    KgProfile,
    Partition,
    ProfileStageOutput,
    PredicateRoles,
    Stage,
    WithinKgNonCompetingPartitionPair,
    check_structure,
    derived_group_roles,
    describe_predicate_pattern,
    describe_profile,
    describe_structure_relation_heading,
    diff_roles,
    effective_roles,
    role_elements_of,
    role_element_order,
    role_elements_with_added,
    roles_from_elements,
    short_iri,
)
from grasp.tasks.om.foundation.configuration import (
    load_task_instructions,
)
from grasp.tasks.om.foundation.rendering import Item, Section, SectionContent, Table, format_display_content
from grasp.tasks.om.foundation.types import LocalNameType
from grasp.tasks.om.foundation.rdf_vocabulary import (
    DEFAULT_LABEL_PREDICATES,
    RDF_TYPE,
    UNCATEGORIZED,
    VOCABULARY,
)
from grasp.tasks.om.kg_operations.entity_rendering import clip
from grasp.tasks.om.kg_operations.grouping import (
    Grouping,
    default_non_competing_pairs,
    get_assigned_default_partitions,
    ensure_groups,
    ensure_partitions,
    retrieve_group_sizes,
    grouping_to_sparql_query,
    groups_to_grouping,
    get_default_partitions,
    partitions_to_grouping,
    single_group_grouping,
)
from grasp.tasks.om.kg_operations.profile_statistics import (
    PredicateCoverage,
    DirectionalUsage,
    canon_roles,
    complement,
    default_roles,
    get_predicate_usage,
    element_usage_in_roles,
    ensure_space,
    member_coverage,
    members_query,
    membership_overlap,
    predicate_coverage,
    rest,
    type_coverage,
    unassigned_predicates,
)
from grasp.tasks.om.sparql_access import (
    query_failure_message,
    select_rows,
    union_of,
)
from grasp.tasks.om.tasks.prompting import BLANK_NODE_RULE, format_validation_error, inline_schema, load_prompt
from grasp.utils import FunctionCallException, format_section


class ProfileFunctionName(StrEnum):
    """The names of the LLM functions of the profile stages."""
    # entity space (stage a)
    INSPECT_NAMESPACE = auto()
    SET_ENTITY_SPACE = auto()
    # partitions and groups (stages b, d)
    TEST_PARTITION = auto()
    TEST_GROUP = auto()
    SET_PARTITION = auto()
    REMOVE_PARTITION = auto()
    SET_COMPETITION = auto()
    SET_GROUP = auto()
    EXCLUDE_GROUP = auto()
    REMOVE_GROUP = auto()
    SET_LOCAL_NAME_TYPE = auto()
    # roles (stages c, e)
    TRY_PATTERN = auto()
    EDIT_ROLE = auto()
    SET_STRUCTURE_RELATION = auto()
    REMOVE_STRUCTURE_RELATION = auto()
    SHOW_ROLES = auto()
    # findings and flags
    ADD_FINDING = auto()
    REMOVE_FINDING = auto()
    RAISE_FLAG = auto()
    REMOVE_FLAG = auto()
    RESOLVE_FLAG = auto()
    FINISH = auto()


MAX_FINDINGS_PER_STAGE = 5
MAX_FINDINGS_AFTER_REVISION = 10
NUM_EXAMPLE_MATCHES_TRY_PATTERN = 8
MAX_VALUE_CHARS = 80
# defaults of the profile parts
NUM_NAMESPACES = 15
NUM_MEMBERS_LISTED_PER_PREFIX = 8
NUM_SHOWN_PREDICATES_PER_PARTITION = 10
NUM_PREDICATES = 25
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# stage a) counts entities per prefix by kind,
# without the subjects outside all kinds
DEFAULT_SPACE_PARTITION_GROUPING = partitions_to_grouping(
    get_default_partitions(), mark_uncategorized_as_rest=False
    )
# name of all entities of a grouping, to compare a group with the rest
ALL = "_all"
# exclude_group warns if the excluded groups cover more of the entity space
LARGE_EXCLUDED_GROUP_SHARE = 0.5
# stage e runs in full mode for a group with at least MIN_UNASSIGNED predicates
# without a role, each covering MIN_UNASSIGNED_SHARE of its entities as subject
# (vocabulary predicates such as rdf:type not counted), or for a group formed in
# stage d; otherwise in light mode
MIN_UNASSIGNED_PREDICATES = 3
MIN_UNASSIGNED_PREDICATE_SHARE = 0.05
MIN_GROUP_SIZE_FOR_GROUP_STAGE = 50


def format_share(n: int, size: int) -> str:
    return f"{n}/{size} ({n / size:.0%})" if size else str(n)


def run_retrieve_rows(
        manager: KgManager, sparql: str, llm_written_patterns: Sequence[str] = ()
        ) -> list[dict[str, Binding]]:
    try:
        return select_rows(manager, sparql)
    except Exception as e:
        raise FunctionCallException(
            query_failure_message(e, llm_written_patterns)
            ) from e


def get_entity_examples(
        manager: KgManager,
        grouping: Grouping,
        space: list[str],
        name_pattern: str,
        where: str,
        k: int
        ) -> list[Item]:
    """
    Up to k entities with a name (name_pattern binds it to ?o; see
    Editor.name_sparql) and their partition or group.
    """
    rows = run_retrieve_rows(
        manager,
        grouping_to_sparql_query(
            manager,
            grouping,
            space,
            "?s ?group ?o",
            f"""{where}\nOPTIONAL {{ {name_pattern} }}""",
            f"LIMIT {k * 4}"
            )
        )
    seen: set[str] = set()
    items: list[Item] = []
    for row in rows:
        entity = row["s"].value
        if entity in seen:
            continue
        seen.add(entity)
        name = f' "{clip(row["o"].value, 50)}"' if "o" in row else ""
        items.append(Item(f"{entity}{name} ({row['group'].value})"))
        if len(items) == k:
            break
    return items


def size_text(
        sizes: dict[str, int],
        name: str
        ) -> str:
    """
    Size of a partition or group.
    """
    return f"{sizes.get(name, 0)} entities"


def run_retrieve_group_sizes(
        manager: KgManager, grouping: Grouping, space: list[str]
        ) -> dict[str, int]:
    try:
        return retrieve_group_sizes(manager, grouping, space)
    except Exception as e:
        raise FunctionCallException(query_failure_message(e)) from e


def format_predicate_usage(
        usage_inside: dict[str, int],
        size_inside: int,
        usage_outside: dict[str, int],
        size_outside: int,
        other: str
        ) -> list[Item]:
    used_often, used_seldom = get_predicate_usage(
        usage_inside,
        size_inside,
        usage_outside,
        size_outside,
        NUM_SHOWN_PREDICATES_PER_PARTITION
        )
    return [
        Item(
            "Common predicates:",
            [
                Item(f"{short_iri(pu.key)}: {pu.share_inside:.0%} used inside, {pu.share_outside:.0%} in {other}")
                for pu in used_often
                ]
        ),
        Item("Seldom predicates:",
             [
                 Item(f"{short_iri(pu.key)}: {pu.share_inside:.0%} used inside, {pu.share_outside:.0%} in {other}")
                 for pu in used_seldom
                 ]
             )
        ]


def validate_membership_pattern(fn_args: dict) -> str:
    membership = (fn_args.get("membership") or "").strip()
    if not re.search(r"\?s\b", membership):
        raise FunctionCallException("The membership pattern must use ?s")
    return membership


def validate_name(name: str, unit: str) -> str:
    if not NAME_RE.match(name or "") or name in (UNCATEGORIZED, ALL):
        raise FunctionCallException(
            f"Invalid {unit} name {name!r}: only lower case letters, digits and _ allowed"
            )
    return name


MEMBERSHIP_ARG_DESCRIPTION = {
    "type": "string",
    "description": (
        "SPARQL group pattern over ?s, e.g. ?s <http://example.org/p> ?o"
        )
    }


def test_grouping_function(function_name: ProfileFunctionName, unit: str, description: str) -> dict:
    return {
        "name": function_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "membership": MEMBERSHIP_ARG_DESCRIPTION,
                "compare_with": {
                    "type": "string", "description": f"The {unit} to compare with."
                    }
                },
            "required": ["membership"],
            "additionalProperties": False,
        },
    }


def format_value_type_shares(values: int, literals: int, blanks: int) -> str:
    """
    Kinds of the values (objects) of a predicate.
    """
    if not values:
        return ""
    iris = values - literals - blanks
    parts = [f"{iris / values:.0%} IRI"]
    if blanks:
        parts.append(f"{blanks / values:.0%} blank node")
    parts.append(f"{literals / values:.0%} literal")
    return ", ".join(parts)


def format_predicate_overview(
        predicate_coverage: dict[str, dict[str, PredicateCoverage]],
        partition_sizes: dict[str, int],
        partition_names: list[str],
        roles: PredicateRoles,
        k: int
        ) -> Table:
    """
    The predicates with the highest coverage in some partition: the share of
    each partition's entities that are its subject (as ?s), the kinds of its
    values (?o) over these partitions, and the role it has. One column per
    partition; '-' where no entity of the partition has the predicate.
    """
    # a predicate may be in several roles or relations
    role_names_by_predicate: dict[str, list[str]] = defaultdict(list)
    for element in sorted(role_elements_of(roles), key=role_element_order):
        if element.kind == RoleElementKind.PREDICATE:
            role_names_by_predicate[str(element.definition)].append(element.role_name)
    most_used_in_partition: dict[str, float] = {}
    for p in partition_names:
        for predicate, c in predicate_coverage.get(p, {}).items():
            if partition_sizes.get(p):
                most_used_in_partition[predicate] = max(
                    most_used_in_partition.get(predicate, 0.0),
                    c.entities / partition_sizes[p]
                    )
    rows: list[tuple[str, ...]] = []
    for predicate in sorted(
            most_used_in_partition, key=lambda x: -most_used_in_partition[x]
            )[:k]:
        num_subjetcs_per_partition: list[str] = []
        values = literals = blanks = 0
        for p in partition_names:
            c = predicate_coverage.get(p, {}).get(predicate)
            if c is None or not partition_sizes.get(p):
                num_subjetcs_per_partition.append("-")
                continue
            num_subjetcs_per_partition.append(f"{c.entities / partition_sizes[p]:.0%}")
            values, literals, blanks = values + c.values, literals + c.literals, blanks + c.blanks
        rows.append((
            short_iri(predicate),
            ", ".join(role_names_by_predicate.get(predicate, [])),
            *num_subjetcs_per_partition,
            format_value_type_shares(values, literals, blanks),
            ))
    return Table(
        ("predicate", "roles", *(f"{p} as ?s" for p in partition_names), "values (?o)"),
        rows,
        )


class GroupEditMode(StrEnum):
    FULL = "full"    # roles of the group, findings
    LIGHT = "light"  # mainly matching pitfalls of the group as findings


MAX_GROUP_TASK_STEPS_BY_MODE = {GroupEditMode.FULL: 30, GroupEditMode.LIGHT: 12}


def group_mode(
        manager: KgManager, profile: KgProfile, group: str
        ) -> tuple[GroupEditMode, list[str]]:
    """
    (mode of stage e) for a group, the unassigned predicates that count).
    """
    grouping = single_group_grouping(profile, group)
    current = profile.get_group(group)
    usage = element_usage_in_roles(manager, grouping, profile.entity_space, profile.roles).get(group, {})
    effective = effective_roles(profile.roles, current, usage)
    coverage = predicate_coverage(manager, grouping, profile.entity_space).get(group, {})
    size = retrieve_group_sizes(manager, grouping, profile.entity_space).get(group, 0)
    counted = [p for p, c in unassigned_predicates(effective, coverage)
               if not p.startswith(VOCABULARY) and size and c.entities / size >= MIN_UNASSIGNED_PREDICATE_SHARE]
    formed = current.definition.origin == GroupOrigin.FORMED
    return (GroupEditMode.FULL if formed or len(counted) >= MIN_UNASSIGNED_PREDICATES else GroupEditMode.LIGHT), counted


class EditorBaseFields(TypedDict):
    kg_manager: KgManager
    entity_space: list[str]
    name_predicate_pattern: PredicatePattern | None


def names_to_sparql(name_predicate_pattern: PredicatePattern | None) -> str:
    """
    The names of ?s (bound to ?o), as a SPARQL union: the names role, or the
    default label predicates before stage c sets it.
    """
    if name_predicate_pattern:
        return union_of(name_predicate_pattern.branches_for_union())
    return union_of([f"?s <{predicate}> ?o" for predicate in DEFAULT_LABEL_PREDICATES])


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
    name_predicate_pattern: PredicatePattern | None  # the KG-wide names role of the profile

    def name_sparql(self) -> str:
        """The names of ?s for examples (bound to ?o)."""
        return names_to_sparql(self.name_predicate_pattern)

    @staticmethod
    def base_fields(
            kg_manager: KgManager,
            kg_profile: KgProfile,
            entity_space: list[str] | None = None
            ) -> EditorBaseFields:
        """
        The fields every editor takes from the profile: the manager, the entity
        space (ensure_space, unless given) and the KG-wide names role.
        """
        return EditorBaseFields(
            kg_manager=kg_manager,
            entity_space=entity_space if entity_space is not None else ensure_space(kg_manager, kg_profile),
            name_predicate_pattern=kg_profile.roles.names,
            )

    @abstractmethod
    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        """The functions for the LLM: (definition, handler) pairs."""

    @abstractmethod
    def input_for_llm(self) -> list[Section]:
        """The overview of the current state, for the input of the task."""


@dataclass(eq=False)
class EntitySpaceEditor(Editor):
    """
    The entity space: IRI prefixes of the entities to align.
    """
    partition_grouping: Grouping

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
            ) -> Self:
        """
        The entity space of the profile. create_missing: without one, the
        default space (configured namespaces or the prefix with the most named
        entities; needs config); otherwise the editor starts empty.
        """
        entity_space = (
            ensure_space(kg_manager, kg_profile, config) if create_missing
            else list(kg_profile.entity_space)
            )
        partition_grouping = (
            partitions_to_grouping(kg_profile.partitions)
            if kg_profile.partitions else DEFAULT_SPACE_PARTITION_GROUPING
            )
        return cls(
            **cls.base_fields(kg_manager, kg_profile, entity_space),
            partition_grouping=partition_grouping,
            )

    def input_for_llm(self) -> list[Section]:
        return [Section("Entity space", [", ".join(self.entity_space) or "none"])]

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            ({
                "name": ProfileFunctionName.INSPECT_NAMESPACE,
                "description": (
                    "Shows the entities under an IRI prefix: their number per kind, how many have a "
                    "canon name, and examples with names and types."
                    ),
                "parameters": {
                    "type": "object",
                    "properties": {"prefix": {"type": "string", "description": "IRI prefix, e.g. http://example.org/onto#"}},
                    "required": ["prefix"],
                    "additionalProperties": False,
                },
            }, self.inspect_namespace),
            ({
                "name": ProfileFunctionName.SET_ENTITY_SPACE,
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

        group_sizes = run_retrieve_group_sizes(
            self.kg_manager, self.partition_grouping, [prefix]
            )
        if not group_sizes:
            return format_display_content(
                Section(
                    f"Namespace {prefix}",
                    ["No entities with IRIs starting with it found."]
                    )
                )
        named_entities = run_retrieve_rows(
            self.kg_manager,
            grouping_to_sparql_query(
                self.kg_manager,
                self.partition_grouping,
                [prefix],
                "(COUNT(DISTINCT ?s) AS ?n)",
                "FILTER EXISTS {" + self.name_sparql() + "}"
                )
        )
        named_count = int(
            named_entities[0]["n"].value
            ) if named_entities and "n" in named_entities[0] else 0
        types = run_retrieve_rows(
            self.kg_manager,
            grouping_to_sparql_query(
                self.kg_manager,
                self.partition_grouping,
                [prefix],
                "?t (COUNT(DISTINCT ?s) AS ?n)",
                f"?s <{RDF_TYPE}> ?t",
                "GROUP BY ?t ORDER BY DESC(?n) LIMIT 8",
                )
            )
        return format_display_content(
            Section(
                f"Namespace {prefix}", [
                    "entities by kind:",
                    *(Item(f"{group}: {size}") for group, size in group_sizes.items()),
                    Item(f"{named_count} with a canon name"),
                    "most frequent types:",
                    *(
                        [
                            Item(f"{short_iri(row['t'].value)}: {row['n'].value}")
                            for row in types if "t" in row
                            ] if types else [Item("none")]
                        ),
                    "examples:", *get_entity_examples(
                        self.kg_manager,
                        self.partition_grouping,
                        [prefix],
                        self.name_sparql(),
                        "",
                        8
                        ),
                    ]
                )
            )

    def prefix_has_entities(self, prefix: str) -> bool:
        return bool(
            run_retrieve_group_sizes(self.kg_manager, self.partition_grouping, [prefix])
            )

    def set_entity_space(self, fn_args: dict) -> str:
        prefixes = [prefix for prefix in fn_args.get("prefixes") or [] if prefix]
        if not prefixes:
            raise FunctionCallException("Name at least one IRI prefix")
        empty_prefixes = [
            prefix for prefix in prefixes if not self.prefix_has_entities(prefix)
            ]
        if empty_prefixes:
            raise FunctionCallException(f"No entities under {empty_prefixes}")
        self.entity_space = prefixes
        return format_display_content(
            Section("Entity space set", ["Entity space: " + ", ".join(prefixes)])
            )

    def checked_entity_space(self) -> list[str]:
        """
        The result: the entity space, if it names a prefix.
        """
        if not self.entity_space:
            raise FunctionCallException(
                f"The entity space is empty; set it with {ProfileFunctionName.SET_ENTITY_SPACE}"
                )
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
    test_function_name: ClassVar[ProfileFunctionName]
    definitions: dict[str, T]  # name -> definition, in order
    tested_patterns: set[str] = field(init=False)  # membership patterns tested with test_<unit>
    group_sizes: dict[str, int] = field(init=False)
    sizes_by_predicates: dict[str, dict[str, int]] = field(init=False)  # member -> predicate -> num entities
    sizes_by_types: dict[str, dict[str, int]] = field(init=False)  # member -> type -> num entities


    def __post_init__(self) -> None:
        self.definitions = dict(self.definitions)
        self.tested_patterns = set()
        self.update_statistics()

    @abstractmethod
    def grouping(self) -> Grouping:
        """The definitions as a Grouping, for the statistics queries."""

    @abstractmethod
    def sizes_outside_of_member(
            self, kind: Literal["predicates", "types"], name: str
            ) -> tuple[dict[str, int], int]:
        """
        Counts of predicates or types and size of the entities outside one member.
        """

    def member_names(self) -> list[str]:
        return list(self.definitions)

    def size_stats(self) -> dict:
        return {"sizes": self.group_sizes}

    def size_text(self, name: str) -> str:
        return size_text(self.group_sizes, name)

    def update_statistics(self) -> None:
        grouping = self.grouping()
        try:
            self.group_sizes = retrieve_group_sizes(
                self.kg_manager, grouping, self.entity_space
                )
            self.sizes_by_predicates = {
                member: {predicate: c.entities for predicate, c in counts.items()}
                for member, counts in predicate_coverage(
                    self.kg_manager, grouping, self.entity_space
                    ).items()
                }
            self.sizes_by_types = type_coverage(
                self.kg_manager, grouping, self.entity_space
                )
        except Exception as e:
            raise FunctionCallException(query_failure_message(e)) from e

    def describe_member(self, name: str, title: str) -> Section:
        """
        A member with examples and the predicates and types that distinguish it.
        """
        size = self.group_sizes.get(name, 0)
        outside, size_outside = self.sizes_outside_of_member("predicates", name)
        content: list[SectionContent] = [
            f"{self.size_text(name)}.",
            Item("examples:", get_entity_examples(
                self.kg_manager,
                self.grouping(),
                self.entity_space,
                self.name_sparql(),
                f'FILTER(?group = "{name}")', 3)
                ),
            Item("predicates (share of entities with the predicate):", [
                item for item in format_predicate_usage(
                    self.sizes_by_predicates.get(name, {}), size, outside, size_outside, f"other {self.unit}s")
                if item.children
                ]),
            ]
        if self.sizes_by_types.get(name):
            outside, size_outside = self.sizes_outside_of_member("types", name)
            content.append(Item("types (share of entities with the type):", [
                item for item in format_predicate_usage(
                    self.sizes_by_types[name], size, outside, size_outside, f"other {self.unit}s")
                if item.children
                ]))
        return Section(title, content)

    def update_definitions(self, definitions: dict[str, T]) -> None:
        """
        Sets the definitions and their statistics;
        on a failed sparql query, nothing changes.
        """
        old = self.definitions
        self.definitions = definitions
        try:
            self.update_statistics()
        except FunctionCallException:
            self.definitions = old
            raise

    def name_known(self, name: str | None) -> str:
        if name is None or name not in self.definitions:
            raise FunctionCallException(
                f"Unknown {self.unit} {name}; {self.unit}s: {', '.join(self.member_names())}"
                )
        return name

    def test_definition(self, fn_args: dict) -> str:
        membership_pattern = validate_membership_pattern(fn_args)
        try:
            size, overlap = membership_overlap(
                self.kg_manager, self.grouping(), self.entity_space, membership_pattern
                )
            inside, inside_types = member_coverage(
                self.kg_manager, self.entity_space, membership_pattern
                )
        except Exception as e:
            raise FunctionCallException(
                query_failure_message(e, [membership_pattern])
                ) from e
        self.tested_patterns.add(membership_pattern)
        title = f"Test of the membership pattern {{ {membership_pattern} }}"
        if not size:
            return format_display_content(
                Section(title, ["No entity of the entity space matches."])
                )
        size_line = f"{size} entities match; "
        compare = fn_args.get("compare_with")
        if compare is not None:
            self.name_known(compare)
        else:
            compare = max((member for member in overlap if member in self.group_sizes),
                          key=lambda member: overlap[member], default=None)
        where = f"by current {self.unit} (share of the {self.unit} that matches)"
        content: list[SectionContent] = [
            size_line + where + ": "
            + ", ".join(f"{p} {format_share(n, self.group_sizes.get(p, 0))}"
                        for p, n in sorted(overlap.items(), key=lambda x: -x[1]))
            ]
        if compare is not None:
            rest_size = self.group_sizes.get(compare, 0) - overlap.get(compare, 0)
            for kind, counts, ins in (("predicates", self.sizes_by_predicates, inside), ("types", self.sizes_by_types, inside_types)):
                outside = {k: max(0, n - ins.get(k, 0)) for k, n in counts.get(compare, {}).items()}
                usage_items = [item for item in format_predicate_usage(
                    ins, size, outside, rest_size, f"rest of {compare}") if item.children]
                if usage_items:
                    content.append(
                        Item(f"{kind} (share of the matches vs. the rest of {compare}):",
                             usage_items)
                        )
        rows = run_retrieve_rows(
            self.kg_manager,
            # the membership pattern is in a subquery that returns only ?s, so
            # its own ?o does not meet the ?o of the names
            "SELECT DISTINCT ?s ?o WHERE {" + members_query(
                membership_pattern, self.entity_space
                ) + f"\nOPTIONAL {{ {self.name_sparql()} }} "
            + "} LIMIT 6",
            [membership_pattern]
        )
        content.append(Item("examples:", [
            Item(r["s"].value + (f' "{clip(r["o"].value, 50)}"' if "o" in r else ""))
            for r in rows if "s" in r
            ]))
        return format_display_content(Section(title, content))

    def checked_membership(self, fn_args: dict) -> str:
        membership = validate_membership_pattern(fn_args)
        if membership not in self.tested_patterns:
            raise FunctionCallException(
                f"Test the membership pattern with {self.test_function_name} first (exactly this pattern)"
                )
        return membership


@dataclass(eq=False)
class PartitionEditor(GroupingEditor[Partition]):
    """
    The exclusive partitions (an entity belongs to the first that matches)
    and the pairs of partitions whose entities do not compete.
    """
    unit = "partition"
    test_function_name = ProfileFunctionName.TEST_PARTITION
    non_competing_pairs: list[WithinKgNonCompetingPartitionPair]

    def __post_init__(self) -> None:
        self.non_competing_pairs = list(self.non_competing_pairs)
        super().__post_init__()

    @classmethod
    def from_kg_profile(
            cls, kg_manager: KgManager, kg_profile: KgProfile, *, create_missing: bool
            ) -> Self:
        """
        The partitions and non-competing pairs of the profile. create_missing:
        without partitions, the kind partitions with entities and the default
        pairs (properties do not compete); otherwise a ValueError.
        """
        if not kg_profile.partitions and not create_missing:
            raise ValueError("The profile has no partitions")
        base = cls.base_fields(kg_manager, kg_profile)
        definitions = ensure_partitions(kg_manager, kg_profile, base["entity_space"])
        non_competing_pairs = (
            list(kg_profile.non_competing_pairs)
            if kg_profile.partitions
            else default_non_competing_pairs(definitions)
            )
        return cls(
            **base,
            definitions=definitions,
            non_competing_pairs=non_competing_pairs,
        )

    def grouping(self) -> Grouping:
        # the subjects outside all partitions as a member
        # of their own (UNCATEGORIZED)
        return partitions_to_grouping(
            list(self.definitions.values()), mark_uncategorized_as_rest=True
            )

    def sizes_outside_of_member(
            self, kind: Literal["predicates", "types"], name: str
            ) -> tuple[dict[str, int], int]:
        counts = self.sizes_by_predicates if kind == "predicates" else self.sizes_by_types
        return complement(counts, self.group_sizes, name)

    def input_for_llm(self) -> list[Section]:
        return [
            Section("Partitions in order (an entity belongs to the first that matches)", self.format_sizes()),
            Section("Partitions that do not compete with each other", self.format_non_competing_pairs()),
            ]

    def format_sizes(self) -> list[SectionContent]:
        items: list[SectionContent] = [
            Item(f"{name}: {self.size_text(name)}" + (f"; {partition.features}" if partition.features else ""),
                 [Item(f"membership: {{ {partition.membership_sparql} }}")])
            for name, partition in self.definitions.items()
            ]
        if self.group_sizes.get(UNCATEGORIZED):
            items.append(Item(f"{UNCATEGORIZED}: {self.group_sizes[UNCATEGORIZED]} subjects match no partition"))
        return items

    def format_non_competing_pairs(self) -> list[SectionContent]:
        if not self.non_competing_pairs:
            return ["None: all partitions compete."]
        return [
            Item(f"{pair.partition_a} - {pair.partition_b}" + (f": {pair.reason}" if pair.reason else ""))
            for pair in self.non_competing_pairs
            ]

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            (test_grouping_function(self.test_function_name, self.unit, (
                "Tests a membership pattern: how many entities of the entity space match, in which "
                "current partitions they are, which predicates and types distinguish them from the "
                "rest of the partition they overlap most (or compare_with), and examples. A pattern "
                "must be tested before it can be set.")), self.test_definition),
            ({
                "name": ProfileFunctionName.SET_PARTITION,
                "description": (
                    "Defines a partition by a tested membership pattern, or redefines an existing one "
                    "(it keeps its place). A new partition is placed before the partition named in "
                    "before, otherwise first. Shows the partition sizes afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A short lower case noun for the kind."},
                        "membership": MEMBERSHIP_ARG_DESCRIPTION,
                        "features": {"type": "string", "description": "What distinguishes its entities."},
                        "before": {"type": "string"},
                    },
                    "required": ["name", "membership", "features"],
                    "additionalProperties": False,
                },
            }, self.set_partition),
            ({
                "name": ProfileFunctionName.REMOVE_PARTITION,
                "description": "Removes a partition; its entities fall to the next matching partition.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }, self.remove_partition),
            ({
                "name": ProfileFunctionName.SET_COMPETITION,
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
        name = validate_name(fn_args.get("name") or "", self.unit)
        membership = self.checked_membership(fn_args)
        before = fn_args.get("before")
        if before is not None and self.name_known(before) == name:
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
        return format_display_content(Section(f"Partition {name} set", subsections=self.input_for_llm()))

    def remove_partition(self, fn_args: dict) -> str:
        name = self.name_known(fn_args.get("name"))
        if len(self.definitions) == 1:
            raise FunctionCallException("The last partition cannot be removed")
        self.update_definitions({other: partition for other, partition in self.definitions.items() if other != name})
        self.non_competing_pairs = [pair for pair in self.non_competing_pairs if name not in pair.partitions()]
        return format_display_content(Section(f"Partition {name} removed", subsections=self.input_for_llm()))

    def set_competition(self, fn_args: dict) -> str:
        partition_a = self.name_known(fn_args.get("partition_a"))
        partition_b = self.name_known(fn_args.get("partition_b"))
        if partition_a == partition_b:
            raise FunctionCallException(
                "Entities of one partition always compete with each other"
                )
        self.non_competing_pairs = [
            pair for pair in self.non_competing_pairs
            if pair.partitions() != {partition_a, partition_b}
            ]
        if not fn_args.get("compete"):
            self.non_competing_pairs.append(
                WithinKgNonCompetingPartitionPair(
                    partition_a=partition_a, partition_b=partition_b, reason=fn_args.get("reason") or ""
                )
                )
        return format_display_content(
            Section(
                "Partitions that do not compete with each other", self.format_non_competing_pairs()
                )
            )

    def with_partitions(self, profile: KgProfile) -> KgProfile:
        """
        The profile with the entity space, the partitions that have
        entities and their non-competing pairs.
        """
        kept = [partition for name, partition in self.definitions.items() if self.group_sizes.get(name)]
        kept_names = {partition.name for partition in kept}
        non_competing_pairs = [pair for pair in self.non_competing_pairs if pair.partitions() <= kept_names]
        return profile.model_copy(
            update={
                "entity_space": self.entity_space,
                "partitions": kept,
                "non_competing_pairs": non_competing_pairs
            }, deep=True
            )

    def check_partitions(self) -> None:
        if not any(self.group_sizes.get(name) for name in self.definitions):
            raise FunctionCallException("No partition has entities")

    def empty_partition_names(self) -> list[str]:
        """
        Partitions without entities: with_partitions drops them.
        """
        return [name for name in self.definitions if not self.group_sizes.get(name)]


@dataclass(eq=False)
class GroupEditor(GroupingEditor[GroupDefinition]):
    """
    The overlapping groups and their exclusion from the alignment (an entity
    in an excluded group is not aligned), and the local name type of a group.
    """
    unit = "group"
    test_function_name = ProfileFunctionName.TEST_GROUP
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
        without groups, one per partition; otherwise a ValueError.
        """
        if not kg_profile.groups and not create_missing:
            raise ValueError("The profile has no groups")
        base = cls.base_fields(kg_manager, kg_profile)
        return cls(
            **base,
            definitions=ensure_groups(kg_manager, kg_profile, base["entity_space"]),
            exclusions={group.name: group.exclusion_reason for group in kg_profile.groups if group.excluded},
        )

    def grouping(self) -> Grouping:
        return groups_to_grouping(list(self.definitions.values()))

    def update_statistics(self) -> None:
        super().update_statistics()
        merged = self.grouping().merged(ALL)
        try:
            self.all_size = retrieve_group_sizes(self.kg_manager, merged, self.entity_space).get(ALL, 0)
            self.all_predicates = {predicate: c.entities for predicate, c in
                                   predicate_coverage(self.kg_manager, merged, self.entity_space).get(ALL, {}).items()}
            self.all_types = type_coverage(self.kg_manager, merged, self.entity_space).get(ALL, {})
        except Exception as e:
            raise FunctionCallException(query_failure_message(e)) from e

    def sizes_outside_of_member(self, kind: Literal["predicates", "types"], name: str) -> tuple[dict[str, int], int]:
        if kind == "predicates":
            return rest(self.all_predicates, self.all_size, self.sizes_by_predicates.get(name, {}), self.group_sizes.get(name, 0))
        return rest(self.all_types, self.all_size, self.sizes_by_types.get(name, {}), self.group_sizes.get(name, 0))

    def input_for_llm(self) -> list[Section]:
        return [Section("Groups", [
            f"{self.all_size} entities in any group; groups may overlap.", *self.format_sizes()])]

    def format_sizes(self) -> list[SectionContent]:
        items: list[SectionContent] = []
        for name, definition in self.definitions.items():
            notes = [definition.origin.value] + ([f"part of {definition.parent}"] if definition.parent else [])
            excluded = f" [excluded: {self.exclusions[name]}]" if name in self.exclusions else ""
            items.append(Item(
                f"{name}: {self.size_text(name)} ({', '.join(notes)})"
                + (f"; {definition.features}" if definition.features else "") + excluded,
                [Item(f"membership: {{ {definition.membership_sparql} }}")]))
        return items

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [
            (test_grouping_function(self.test_function_name, self.unit, (
                "Tests a membership pattern: how many entities of the entity space match, how much of "
                "each current group they cover, which predicates and types distinguish them from the "
                "rest of the group they overlap most (or compare_with), and examples. A pattern must "
                "be tested before it can be set.")), self.test_definition),
            ({
                "name": ProfileFunctionName.SET_GROUP,
                "description": (
                    "Defines a group by a tested membership pattern, or redefines an existing one. "
                    "Groups may overlap. Shows the group sizes afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A short lower case noun for the group."},
                        "membership": MEMBERSHIP_ARG_DESCRIPTION,
                        "features": {"type": "string",
                                     "description": "What distinguishes its entities, so that they can be recognized."},
                        "parent": {"type": "string", "description": "A group it is part of, if one."},
                    },
                    "required": ["name", "membership", "features"],
                    "additionalProperties": False,
                },
            }, self.set_group),
            ({
                "name": ProfileFunctionName.EXCLUDE_GROUP,
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
                "name": ProfileFunctionName.REMOVE_GROUP,
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
            "name": ProfileFunctionName.SET_LOCAL_NAME_TYPE,
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
        name = validate_name(fn_args.get("name") or "", self.unit)
        membership = self.checked_membership(fn_args)
        parent = fn_args.get("parent")
        if parent is not None and self.name_known(parent) == name:
            raise FunctionCallException("A group cannot be part of itself")
        definition = GroupDefinition(name=name, membership_sparql=membership, parent=parent,
                                     features=fn_args.get("features") or "")
        self.update_definitions({**self.definitions, name: definition})
        return format_display_content(Section(f"Group {name} set", subsections=self.input_for_llm()))

    def excluded_share(self, exclusions: dict[str, str]) -> float:
        """Share of the entities in any group that are in an excluded group."""
        if not exclusions or not self.all_size:
            return 0.0
        grouping = Grouping(tuple(("excluded", self.definitions[name].membership_sparql) for name in exclusions),
                            exclusive=True)
        return retrieve_group_sizes(self.kg_manager, grouping, self.entity_space).get("excluded", 0) / self.all_size

    def exclude_group(self, fn_args: dict) -> str:
        name = self.name_known(fn_args.get("name"))
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
            raise FunctionCallException(query_failure_message(e)) from e
        result = f"Group {name} {'excluded' if excluded else 'included'}; {excluded_share:.0%} of the entities in groups are excluded now."
        if excluded and excluded_share > LARGE_EXCLUDED_GROUP_SHARE:
            result += (" This is a large part. Exclude only entities that would not be matched for this task; "
                       "to exclude part of a group, define that part as a group of its own and exclude it.")
        return format_display_content(Section(
            f"Group {name} {'excluded' if excluded else 'included'}", [result], self.input_for_llm()))

    def remove_group(self, fn_args: dict) -> str:
        name = self.name_known(fn_args.get("name"))
        if len(self.definitions) == 1:
            raise FunctionCallException("The last group cannot be removed")
        membership_pattern = self.definitions[name].membership_sparql
        remaining = {other: definition for other, definition in self.definitions.items() if other != name}
        try:
            size, overlap = membership_overlap(
                self.kg_manager, groups_to_grouping(list(remaining.values())),
                self.entity_space, membership_pattern
                )
        except Exception as e:
            raise FunctionCallException(
                query_failure_message(e, [membership_pattern])
                ) from e
        self.update_definitions(remaining)
        self.exclusions.pop(name, None)
        self.local_name_types.pop(name, None)
        alone = overlap.get(UNCATEGORIZED, 0)
        return format_display_content(Section(f"Group {name} removed", [
            f"{format_share(alone, size)} of its entities are in no other group and are no longer aligned."
            ], self.input_for_llm()))

    def set_local_name_type(self, fn_args: dict) -> str:
        name = self.name_known(fn_args.get("name"))
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
            if not self.group_sizes.get(name):
                continue
            update: dict[str, Any] = {"definition": definition, "excluded": name in self.exclusions,
                                      "exclusion_reason": self.exclusions.get(name, "")}
            if name in self.local_name_types:
                update["local_name_type"] = self.local_name_types[name]
            groups.append(old[name].model_copy(update=update) if name in old else GroupProfile(**update))
        return profile.model_copy(update={"entity_space": self.entity_space, "groups": groups}, deep=True)

    def check_groups(self) -> None:
        kept = [name for name in self.definitions if self.group_sizes.get(name)]
        if not kept:
            raise FunctionCallException("No group has entities")
        if all(name in self.exclusions for name in kept):
            raise FunctionCallException("Every group is excluded: nothing would be aligned")

    def empty_group_names(self) -> list[str]:
        """Groups without entities: with_groups drops them."""
        return [name for name in self.definitions if not self.group_sizes.get(name)]


@dataclass(eq=False)
class RoleEditor(Editor):
    """
    Editing of roles (KG-wide in stage c, of one group in stage e), with the
    coverage of each element over the partitions or groups in scope. The
    coverage is counted once per element and cached; the task may hand in
    counts it has already made (usage, usage_by_direction, counted), the rest is counted
    on creation. Created by from_kg_profile (KG-wide) or from_group_profile
    (one group: its effective roles, and its deviation as the result).
    """
    unit: Literal["partition", "group"]  # in texts for the model
    roles: PredicateRoles  # the working roles, changed by the functions
    grouping: Grouping
    unit_sizes: dict[str, int]
    scope: list[str]  # partitions or groups whose coverage is shown
    # where an element comes from, shown as a mark (stage e: [kg] or [group])
    origin: Callable[[RoleElement], str] | None = None
    # partition or group -> element -> num entities
    role_element_usage: dict[str, RoleElementUsage] = field(default_factory=dict)
    # partition or group -> relation -> (as ?s, as ?o)
    usage_by_direction: dict[str, DirectionalUsage] = field(default_factory=dict)
    # elements counted, also those used nowhere
    counted_role_elements: set[RoleElement] = field(default_factory=set)
    # one group: its roles derived from the KG-wide
    # ones (the base of its deviation); KG-wide: None
    derived_roles: PredicateRoles | None = None

    def __post_init__(self) -> None:
        # own copies: the editor changes them,
        # the caller's objects (e.g. the default) stay as they are
        self.roles = self.roles.model_copy(deep=True)
        self.role_element_usage = {
            member: dict(counts) for member, counts
            in self.role_element_usage.items()
            }
        self.usage_by_direction = {
            member: dict(counts) for member, counts
            in self.usage_by_direction.items()
            }
        self.counted_role_elements = set(self.counted_role_elements)
        self.update_role_element_usage(self.roles)

    @classmethod
    def from_kg_profile(cls, kg_manager: KgManager, kg_profile: KgProfile, *, create_missing: bool) -> Self:
        """
        The KG-wide roles of the profile, with their coverage per partition.
        create_missing: without partitions, the kind partitions; without
        roles, the canon elements with coverage; otherwise a ValueError.
        """
        base = cls.base_fields(kg_manager, kg_profile)
        entity_space = base["entity_space"]
        partitions = list(kg_profile.partitions)
        if not partitions:
            if not create_missing:
                raise ValueError("The profile has no partitions")
            partitions = list(get_assigned_default_partitions(kg_manager, entity_space).values())
        grouping = partitions_to_grouping(partitions, mark_uncategorized_as_rest=False)
        partition_names = [partition.name for partition in partitions]
        group_sizes = retrieve_group_sizes(kg_manager, grouping, entity_space)

        roles = kg_profile.roles
        role_element_usage: dict[str, RoleElementUsage] = {}
        role_element_usage_by_direction: dict[str, DirectionalUsage] = {}
        counted_role_elements: set[RoleElement] = set()
        if not role_elements_of(roles):
            if not create_missing:
                raise ValueError("The profile has no roles")
            # the canon is counted for the default: the
            # editor counts only what is added later
            role_element_usage = element_usage_in_roles(
                kg_manager, grouping, entity_space,
                canon_roles(), role_element_usage_by_direction
                )
            roles = default_roles(role_element_usage, group_sizes, partition_names)
            counted_role_elements = set(role_elements_of(canon_roles()))
        return cls(
            **base,
            unit="partition",
            roles=roles,
            grouping=grouping,
            unit_sizes=group_sizes,
            scope=partition_names,
            role_element_usage=role_element_usage,
            usage_by_direction=role_element_usage_by_direction,
            counted_role_elements=counted_role_elements,
        )

    @classmethod
    def from_group_profile(
            cls, kg_manager: KgManager, kg_profile: KgProfile, group_name: str
            ) -> Self:
        """
        The effective roles of one group (the KG-wide roles used in it, minus
        its removed, plus its added), with the coverage in the group only
        (without the entities of excluded groups); marks show whether an
        element is the group's own or KG-wide.
        """
        if group_name not in kg_profile.group_names():
            raise ValueError(
                f"Unknown group {group_name}; groups: {', '.join(kg_profile.group_names())}"
                )
        base = cls.base_fields(kg_manager, kg_profile)
        entity_space = base["entity_space"]
        grouping = single_group_grouping(kg_profile, group_name)
        group_profile = kg_profile.get_group(group_name)
        usage_by_direction: dict[str, DirectionalUsage] = {}
        usage = element_usage_in_roles(
            kg_manager, grouping, entity_space,
            roles_from_elements(
                role_elements_with_added(role_elements_of(kg_profile.roles), group_profile.added)
                ),
            usage_by_direction,
            )
        group_role_element_usage = usage.get(group_name, {})
        effective_predicate_roles = effective_roles(kg_profile.roles, group_profile, group_role_element_usage)
        added_role_elements = set(group_profile.added)

        def origin(element: RoleElement) -> str:
            return "group" if element in added_role_elements else "kg"

        return cls(
            **base,
            unit="group",
            roles=effective_predicate_roles,
            grouping=grouping,
            unit_sizes=retrieve_group_sizes(kg_manager, grouping, entity_space),
            scope=[group_name],
            origin=origin,
            role_element_usage=usage,
            usage_by_direction=usage_by_direction,
            counted_role_elements=set(role_elements_of(effective_predicate_roles)),
            derived_roles=derived_group_roles(kg_profile.roles, group_role_element_usage),
        )

    def input_for_llm(self) -> list[Section]:
        return self.format_roles(self.roles, self.origin)

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        pattern_schema = inline_schema(PredicatePattern)
        member = {"type": "string", "enum": self.scope}
        return [
            ({
                "name": ProfileFunctionName.TRY_PATTERN,
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
                "name": ProfileFunctionName.EDIT_ROLE,
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
                "name": ProfileFunctionName.SET_STRUCTURE_RELATION,
                "description": (
                    "Adds a structure relation or replaces the one of the same name. Shows its coverage."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"relation": inline_schema(StructureRelation, drop=frozenset({"source"}))},
                    "required": ["relation"],
                    "additionalProperties": False,
                },
            }, self.set_structure_relation),
            ({
                "name": ProfileFunctionName.REMOVE_STRUCTURE_RELATION,
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
    def update_role_element_usage(self, roles: PredicateRoles) -> None:
        """
        Usage of the elements of roles not counted yet.
        """
        known_role_elements = {
            key for counts in self.role_element_usage.values() for key in counts
            } | self.counted_role_elements
        new_elements = {
            e for e in role_elements_of(roles) if e not in known_role_elements
            }
        if not new_elements:
            return
        # roles_from_elements drops a relation part without its relation's properties
        relation_names = {e.role_name for e in new_elements if e.is_structure_relation}
        new_elements |= {
            e for e in role_elements_of(roles)
            if e.kind == RoleElementKind.RELATION_PROPERTIES and e.role_name in relation_names
            }
        usage_by_direction: dict[str, DirectionalUsage] = {}
        try:
            usage = element_usage_in_roles(
                self.kg_manager, self.grouping,
                self.entity_space,
                roles_from_elements(new_elements),
                usage_by_direction
                )
        except Exception as e:
            own_patterns = [
                element.definition for element in new_elements
                if element.kind == RoleElementKind.PATTERN and isinstance(element.definition, str)
                ]
            raise FunctionCallException(query_failure_message(e, own_patterns)) from e
        for member, counts in usage.items():
            self.role_element_usage.setdefault(member, {}).update(counts)
        for member, counts in usage_by_direction.items():
            self.usage_by_direction.setdefault(member, {}).update(counts)
        self.counted_role_elements |= set(new_elements)

    def format_coverage(self, element: RoleElement) -> str:
        counts = [
            f"""{p}: {format_share(
                self.role_element_usage.get(p, {}).get(element, 0),
                self.unit_sizes.get(p, 0)
                )}"""
            for p in self.scope
            ]
        return "; ".join(counts)

    def format_non_structure_role(
            self,
            field: str,
            pattern: PredicatePattern | None,
            origin: Callable[[RoleElement], str] | None = None
            ) -> Item:
        if pattern is None:
            return Item(f"{field}: none")
        mark = lambda element: f" [{origin(element)}]" if origin else ""
        return Item(f"{field}:", [
            Item(f"{element.kind.value} {element.definition}{mark(element)}: {self.format_coverage(element)}")
            for element in sorted(role_elements_of(PredicateRoles(**{field: pattern})), key=role_element_order)
            ])

    def format_structure_relation(
            self,
            relation: StructureRelation,
            origin: Callable[[RoleElement], str] | None = None
            ) -> Item:
        """
        The relation (its properties, with their origin), then each of its
        predicates and patterns with its coverage as ?s and as ?o.
        """
        mark = lambda element: f" [{origin(element)}]" if origin else ""
        elements = sorted(role_elements_of(PredicateRoles(structure=[relation])), key=role_element_order)
        properties = next(e for e in elements if e.kind == RoleElementKind.RELATION_PROPERTIES)
        children = []
        for element in elements:
            if element.kind == RoleElementKind.RELATION_PROPERTIES:
                continue
            direction = lambda i: "; ".join(
                f"{p} {format_share(self.usage_by_direction.get(p, {}).get(element, (0, 0))[i], self.unit_sizes.get(p, 0))}"
                for p in self.scope
                )
            children.append(Item(f"{element.kind.value} {element.definition}{mark(element)}: "
                                 f"as ?s: {direction(0)}; as ?o: {direction(1)}"))
        return Item(f"{describe_structure_relation_heading(relation)}{mark(properties)}", children)

    def format_roles(
            self, roles: PredicateRoles, origin: Callable[[RoleElement], str] | None = None
            ) -> list[Section]:
        """Each element with its coverage per partition or group in scope."""
        self.update_role_element_usage(roles)
        marks = (" [kg]: a KG-wide element used here, [group]: the group's own." if origin else "")
        return [
            Section("Roles", [
                f"Coverage as ?s per {self.unit}.{marks}",
                *(self.format_non_structure_role(field, getattr(roles, field), origin) for field in ROLE_FIELDS),
                ]),
            Section("Structure relations", [
                f"Coverage of each predicate or pattern as ?s and as ?o per {self.unit}.",
                *([self.format_structure_relation(relation, origin) for relation in roles.structure] or ["none"]),
                ]),
            ]

    # --- functions ---
    def try_pattern(self, fn_args: dict) -> str:
        try:
            predicate_pattern = PredicatePattern.model_validate(
                fn_args.get("predicate_pattern") or {}
                )
        except ValidationError as e:
            raise FunctionCallException(
                f"Invalid predicate_pattern: {format_validation_error(e)}"
                ) from e
        member = fn_args.get("show_examples_from")
        if member is not None and member not in self.scope:
            raise FunctionCallException(
                f"Unknown {self.unit} {member}; known: {', '.join(self.scope)}"
                )
        rows = run_retrieve_rows(
            self.kg_manager,
            grouping_to_sparql_query(
                self.kg_manager,
                self.grouping,
                self.entity_space,
                "?group (COUNT(DISTINCT ?s) AS ?subjects) (COUNT(DISTINCT ?o) AS ?objects) "
                "(SUM(IF(isLiteral(?o), 1, 0)) AS ?literals) (COUNT(*) AS ?n)",
                "",
                "GROUP BY ?group",
                predicate_pattern.branches_for_union(),
                ),
            predicate_pattern.branches_for_union()
            )
        counts = {row["group"].value: row for row in rows if "group" in row}
        value = lambda row, key: int(row[key].value) if key in row else 0
        content: list[SectionContent] = []
        for partition in self.scope:
            row = counts.get(partition, {})
            content.append(
                Item(f"{self.unit} {partition}: {format_share(value(row, 'subjects'),
                                                              self.unit_sizes.get(partition, 0))} entities are ?s; "
                     f"{value(row, 'n')} matches, {value(row, 'objects')} distinct ?o, "
                     f"{value(row, 'literals')} literal matches"))
        if member is None:
            in_scope = [p for p in self.scope if value(counts.get(p, {}), "subjects")]
            member = max(in_scope, key=lambda p: value(counts[p], "subjects"), default=None)
        if member is not None and value(counts.get(member, {}), "subjects"):
            content.append(
                Item(f"examples in {member} (?s -> ?o, with names in quotes):",
                     self.format_examples(member, predicate_pattern.branches_for_union()))
                )
        return format_display_content(
            Section(
                f"Test of the pattern: {describe_predicate_pattern(predicate_pattern)}",
                content)
            )

    def name_sparql(self) -> str:
        """The names of ?s by the working names role (the functions may change it)."""
        return names_to_sparql(self.roles.names)

    def format_examples(self, member: str, predicate_patterns: list[str]) -> list[Item]:
        """
        Example matches with a name of ?s and of ?o
        (local names alone might say little).
        """
        # not name_sparql(): the tested pattern itself binds ?o, which the
        # names (bound to ?o) would join; so the name predicates with own
        # variables, the role's predicates (not its patterns) and the defaults
        # (?o may be an entity of another part, where the role need not hold)
        names = self.roles.names.predicates if self.roles.names else []
        name_predicates = list(dict.fromkeys(names + DEFAULT_LABEL_PREDICATES))
        rows = run_retrieve_rows(
            self.kg_manager,
            grouping_to_sparql_query(
                self.kg_manager,
                self.grouping,
                self.entity_space,
                "?s ?o ?s_name ?o_name",
                f'FILTER(?group = "{member}")\n'
                f"OPTIONAL {{ {union_of([f'?s <{p}> ?s_name' for p in name_predicates])} }}\n"
                f"OPTIONAL {{ {union_of([f'?o <{p}> ?o_name' for p in name_predicates])} }}",
                f"LIMIT {NUM_EXAMPLE_MATCHES_TRY_PATTERN * 4}", predicate_patterns,
                ),
            predicate_patterns
            )
        seen: set[tuple[str, str]] = set()
        items: list[Item] = []
        for row in rows:
            if "s" not in row or "o" not in row:
                continue
            s, o = row["s"].value, row["o"].value
            if (s, o) in seen:
                continue
            seen.add((s, o))
            named = lambda var: row[var].value + (
                f' "{clip(row[var + "_name"].value, 40)}"'
                if var + "_name" in row else ""
                )
            items.append(
                Item(f"{named('s')} -> {clip(named('o'), MAX_VALUE_CHARS)}")
                )
            if len(items) == NUM_EXAMPLE_MATCHES_TRY_PATTERN:
                break
        return items

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
                raise FunctionCallException(f"Invalid add: {format_validation_error(e)}") from e
            predicates += [p for p in add.predicates if p not in predicates]
            patterns += [p for p in add.patterns if p not in patterns]
        new = PredicatePattern(predicates=predicates, patterns=patterns) if predicates or patterns else None
        self.update_role_element_usage(PredicateRoles(**{role: new}))  # may raise on SPARQL errors; then nothing changes
        self.roles = self.roles.model_copy(update={role: new})
        return format_display_content(Section(
            f"Role {role} changed", [f"Coverage as ?s per {self.unit}.", self.format_non_structure_role(role, new)]))

    def set_structure_relation(self, fn_args: dict) -> str:
        try:
            relation = StructureRelation.model_validate(
                fn_args.get("relation") or {}
                )
        except ValidationError as e:
            raise FunctionCallException(f"Invalid relation: {format_validation_error(e)}") from e
        relation.source = "llm"
        self.update_role_element_usage(PredicateRoles(structure=[relation]))
        relation_item = self.format_structure_relation(relation)
        structure = [r for r in self.roles.structure if r.name != relation.name]
        replaced = len(structure) < len(self.roles.structure)
        self.roles = self.roles.model_copy(
            update={"structure": structure + [relation]}
            )
        problems = check_structure(self.roles.structure)
        content: list[SectionContent] = [
            f"Coverage of each predicate or pattern as ?s and as ?o per {self.unit}.",
            relation_item
            ]
        if problems:
            content.append("To resolve before finish: " + "; ".join(problems))
        return format_display_content(
            Section(
                f"Structure relation {relation.name} {'replaced' if replaced else 'added'}",
                content
                )
            )

    def remove_structure_relation(self, fn_args: dict) -> str:
        name = fn_args.get("name")
        names = [r.name for r in self.roles.structure]
        if name not in names:
            raise FunctionCallException(f"No structure relation {name}; relations: {', '.join(names) or 'none'}")
        self.roles = self.roles.model_copy(update={"structure": [r for r in self.roles.structure if r.name != name]})
        return format_display_content(Section(f"Structure relation {name} removed", [
            "Structure relations: " + (", ".join(r.name for r in self.roles.structure) or "none")]))

    def check_roles(self) -> None:
        """Raises if the structure relations of the working roles are inconsistent."""
        problems = check_structure(self.roles.structure)
        if problems:
            raise FunctionCallException("Structure relations: " + "; ".join(problems))

    def checked_roles(self) -> PredicateRoles:
        """The result: the working roles, if their structure relations are consistent."""
        self.check_roles()
        return self.roles.model_copy(deep=True)

    def group_deviation(self) -> tuple[list[RoleElement], list[RoleElement]]:
        """
        One group: (added, removed) of the working roles against the roles
        derived from the KG-wide ones. Unchecked (also for the working state):
        call check_roles before taking it as a result.
        """
        if self.derived_roles is None:
            raise ValueError("Only the roles of a group (from_group_profile) have a deviation")
        return diff_roles(self.derived_roles, self.roles)


def format_flag(flag: Flag, show_evidence_queries: bool = False) -> str:
    """
    A flag for the input: id, status, who raised it about what, and its resolution.
    """
    text = (
        f"[{flag.id}] ({flag.status.value}; {flag.raised_by_stage.value} "
        f"on {flag.concerns}): {flag.content}")
    if show_evidence_queries:
        text += (("." if not text.endswith((".", ";", "?", "!")) else "")
                 + f" Evidence query: {flag.evidence_query}")
    if flag.resolution_description:
        text += f" -> {flag.resolution_description}"
    return text


def get_required_argument(fn_args: dict, argument_name: str) -> Any:
    value = fn_args.get(argument_name)
    if value is None or value == "":
        raise FunctionCallException(f"Argument '{argument_name}' missing")
    return value


def remove_own_entry(entries: dict[str, Any], kind: str, entry_id: str) -> str:
    """Removes a finding or flag of the current stage; those of earlier stages cannot be removed."""
    if entries.pop(entry_id, None) is None:
        raise FunctionCallException(
            f"No {kind} {entry_id} of this stage (only your own can be removed); "
            f"yours: {', '.join(entries) or 'none'}"
            )
    return f"{kind.capitalize()} {entry_id} removed. Yours: {', '.join(entries) or 'none'}"


EVIDENCE_QUERY_PARAMETER = {
    "type": "string",
    "description": (
        "SPARQL query whose result supports the statement; it is run and its result shown to you. "
        "Prefer counting queries (COUNT, shares) over lists of examples."
        ),
    }

ID_PARAMETER = {"type": "string"}

ADD_FINDING_SCHEMA = {
    "name": ProfileFunctionName.ADD_FINDING,
    "description": (
        f"Adds an observation about the knowledge graph that matters for matching, with a query "
        f"that supports it. At most {MAX_FINDINGS_PER_STAGE} per stage. The finding gets an id."
        ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "evidence_query": EVIDENCE_QUERY_PARAMETER,
            },
        "required": ["text", "evidence_query"],
        "additionalProperties": False,
        },
    }

REMOVE_FINDING_SCHEMA = {
    "name": ProfileFunctionName.REMOVE_FINDING,
    "description": "Removes a finding you added in this stage.",
    "parameters": {
        "type": "object",
        "properties": {"id": ID_PARAMETER},
        "required": ["id"],
        "additionalProperties": False,
        },
    }

RAISE_FLAG_SCHEMA = {
    "name": ProfileFunctionName.RAISE_FLAG,
    "description": (
        "Raises an objection, with a query that supports it, against a finding or flag of an "
        "earlier stage, or against the result of an earlier stage. The flag gets an id."
        ),
    "parameters": {
        "type": "object",
        "properties": {
            "concerns": {
                "type": "string",
                "description": "Id of a finding or flag of an earlier stage, or the name of an earlier stage.",
                },
            "text": {"type": "string"},
            "evidence_query": EVIDENCE_QUERY_PARAMETER,
            },
        "required": ["concerns", "text", "evidence_query"],
        "additionalProperties": False,
        },
    }

REMOVE_FLAG_SCHEMA = {
    "name": ProfileFunctionName.REMOVE_FLAG,
    "description": "Removes a flag you raised in this stage.",
    "parameters": {
        "type": "object",
        "properties": {"id": ID_PARAMETER},
        "required": ["id"],
        "additionalProperties": False,
        },
    }


class ProfileStageTask(GraspTask):
    """
    One stage of the KG profile. Input: a dict with kg (name of a configured
    KG; may be omitted if exactly one KG is configured), profile (output
    profile of the stage before; omitted for the first stage) and, for the
    group stage, group.
    """
    stage: Stage
    prompt_file: str
    is_revision_task: ClassVar[bool] = False
    kg_manager: KgManager
    input_profile: KgProfile   # as received
    default_profile: KgProfile  # with the stage's default; output without finish
    working_profile: KgProfile  # with the stage's result after finish
    finished: bool
    rationale: str | None
    stats: dict[str, Any]
    new_findings: dict[str, Finding]  # of this stage; set up in setup
    new_flags: dict[str, Flag]  # dito
    id_counter: Counter[Literal["finding", "flag"]]

    # --- stage interface ---

    @abstractmethod
    def prepare(self, input: dict) -> Section:
        """
        Computes default and statistics, sets self.default,
        returns the stage's part of the input.
        """

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return []

    def finish_properties(self) -> dict[str, dict]:
        return {}

    def finish_required(self) -> list[str]:
        return []

    @abstractmethod
    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        """
        (profile with the stage's result, summary);
        raises FunctionCallException on invalid arguments.
        """

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
        self.new_findings, self.new_flags, self.id_counter = {}, {}, Counter()
        stage_section = self.prepare(input)
        self.working_profile = self.default_profile.model_copy(deep=True)
        knowledge_graph = Section(
            f"Knowledge graph: {self.kg_manager.kg}",
            [self.kg_manager.description or ""],
            )
        sections = [knowledge_graph, *self.format_earlier_results(), stage_section]
        return "\n\n".join(format_display_content(section) for section in sections)

    def format_earlier_results(self) -> list[Section]:
        """
        The profile so far (not in the revision: it shows the whole profile
        itself), the findings and the flags of earlier stages.
        """
        profile = self.input_profile
        sections = []
        profile_notes = describe_profile(profile, findings=False)
        if ((profile_notes.content or profile_notes.subsections)
                and not self.is_revision_task):
            sections.append(
                Section("Results of earlier stages",
                        profile_notes.content, profile_notes.subsections)
                )
        if profile.findings and not self.is_revision_task:
            sections.append(
                Section(
                    "Observations of earlier stages", [
                        "Unverified notes of earlier stages. Rely on them only as far as you can confirm "
                        f"them; if one contradicts what you see, report it with a flag in {ProfileFunctionName.RAISE_FLAG}.",
                        *(
                            Item(
                                f"[{f_id}] {f_obj.content} (Evidence query: {f_obj.evidence_query})")
                            for f_id, f_obj in profile.findings.items()
                            ),
                        ]
                    )
                )
        if profile.flags and not self.is_revision_task:
            sections.append(
                Section(
                    "Flags raised so far",
                    [Item(format_flag(flag, True)) for flag in profile.flags.values()]
                    )
                )
        return sections

    def system_information(self) -> str:
        info = load_prompt("profile_common.txt") + "\n\n" + load_prompt(self.prompt_file)
        task_instructions = load_task_instructions(self.config)
        if task_instructions:
            info += "\n\n" + format_section("Task-specific instructions", task_instructions)
        return info

    def rules(self) -> list[str]:
        return self.stage_rules() + [
            BLANK_NODE_RULE,
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

    def id_prefix(self) -> str:
        """Prefix of the ids of this stage's findings and flags; GroupTask adds the group."""
        return str(self.stage)

    def next_id(self, kind: Literal["finding", "flag"]) -> str:
        self.id_counter[kind] += 1
        return f"{self.id_prefix()}-{kind}{self.id_counter[kind]}"

    def run_evidence_query(self, fn_args: dict) -> ExecutionResult:
        evidence = execute_sparql(
            self.managers, self.kg_manager.kg, get_required_argument(fn_args, "evidence_query"),
            self.config.result_max_rows, self.config.result_max_columns,
            request_timeout=self.config.sparql_request_timeout,
            read_timeout=self.config.sparql_read_timeout,
            sparql_result_max_rows=self.config.sparql_result_max_rows,
            )
        if evidence.result is None:
            raise FunctionCallException(evidence.formatted)
        return evidence

    def add_finding(self, fn_args: dict) -> str:
        if not self.is_revision_task and len(self.new_findings) >= MAX_FINDINGS_PER_STAGE:
            raise FunctionCallException(
                f"At most {MAX_FINDINGS_PER_STAGE} findings per stage. Remove one of yours "
                f"({', '.join(self.new_findings)}) if this one is more important."
                )
        text = get_required_argument(fn_args, "text")
        evidence = self.run_evidence_query(fn_args)
        finding_id = self.next_id("finding")
        self.new_findings[finding_id] = Finding(
            id=finding_id, stage=self.stage, content=text, evidence_query=evidence.sparql
            )
        return f"Finding {finding_id} added. Evidence:\n\n{evidence.formatted}"

    def raise_flag(self, fn_args: dict) -> str:
        text = get_required_argument(fn_args, "text")
        concerns = get_required_argument(fn_args, "concerns")
        earlier_stage_names = [str(stage) for stage in self.earlier_stages()]
        if (concerns not in self.input_profile.findings
                and concerns not in self.input_profile.flags
                and concerns not in earlier_stage_names):
            raise FunctionCallException(
                f"{concerns} is no finding, flag or stage before this one. "
                f"Findings: {', '.join(self.input_profile.findings) or 'none'}. "
                f"Flags: {', '.join(self.input_profile.flags) or 'none'}. "
                f"Earlier stages: {', '.join(earlier_stage_names) or 'none'}."
                )
        evidence = self.run_evidence_query(fn_args)
        flag_id = self.next_id("flag")
        self.new_flags[flag_id] = Flag(
            id=flag_id, raised_by_stage=self.stage, concerns=concerns,
            content=text, evidence_query=evidence.sparql
            )
        return f"Flag {flag_id} raised on {concerns}. Evidence:\n\n{evidence.formatted}"

    def remove_finding(self, fn_args: dict) -> str:
        return remove_own_entry(self.new_findings, "finding", get_required_argument(fn_args, "id"))

    def remove_flag(self, fn_args: dict) -> str:
        flag_id = get_required_argument(fn_args, "id")
        flag = self.new_flags.get(flag_id)
        if flag is not None and flag.raised_by_stage != self.stage:
            raise FunctionCallException(
                f"{flag_id} was raised by an earlier stage; close "
                f"it with {ProfileFunctionName.RESOLVE_FLAG}"
            )
        return remove_own_entry(self.new_flags, "flag", flag_id)

    def finish_definition(self) -> dict:
        properties = {
            "rationale": {"type": "string",
                          "description": "What you changed and why, and what remains uncertain."},
            }
        properties |= self.finish_properties()
        return {
            "name": ProfileFunctionName.FINISH,
            "description": "Submits the result of this stage.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["rationale"] + self.finish_required(),
                "additionalProperties": False,
            },
        }

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.stage_functions() + [
            (ADD_FINDING_SCHEMA, self.add_finding),
            (REMOVE_FINDING_SCHEMA, self.remove_finding),
            (RAISE_FLAG_SCHEMA, self.raise_flag),
            (REMOVE_FLAG_SCHEMA, self.remove_flag),
            (self.finish_definition(), self.finish),
            ]

    def function_definitions(self) -> list[dict]:
        return [definition for definition, _ in self.functions()]

    def final_step_functions(self) -> list[dict] | None:
        return [self.finish_definition()]

    def call_function(
            self,
            fn_name: str,
            fn_args: dict,
            known: set[str],
            example_indices: dict | None
            ) -> str:
        handlers = {definition["name"]: handler for definition, handler in self.functions()}
        if fn_name not in handlers:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return handlers[fn_name](fn_args)

    def finish(self, fn_args: dict) -> str:
        profile, summary = self.apply_finish(fn_args)
        earlier_findings = {} if self.is_revision_task else self.input_profile.findings
        earlier_flags = {} if self.is_revision_task else self.input_profile.flags
        profile.findings = earlier_findings | self.new_findings
        profile.flags = earlier_flags | self.new_flags
        self.working_profile, self.finished, self.rationale = profile, True, fn_args.get("rationale")
        return "Submitted. " + summary

    def done(self, fn_name: str) -> bool:
        return fn_name == ProfileFunctionName.FINISH and self.finished

    def output(self, messages: list[Message]) -> dict[str, Any]:
        return ProfileStageOutput(
            kg_name=self.kg_manager.kg,
            stage=self.stage,
            finished=self.finished,
            rationale=self.rationale,
            profile=(self.working_profile if self.finished else self.default_profile),
            default=self.default_profile,
            stats=self.stats
        ).model_dump(mode="json")
