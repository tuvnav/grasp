"""
Data model of the staged KG profile.

Stages, each a GRASP task on one KG; each reads the profile of the stages
before and adds to it:
    a.  space       entity space (namespaces of the entities to align)
    b.  partitions  exclusive partitions: the hard boundaries between kinds of
                    entities, from the KG alone, and which partitions of the KG
                    do not compete with each other
    c.  roles       KG-wide roles: names, synonyms, ..., structure relations
    d.  groups      overlapping groups of entities recognizable by their own
                    features, with the task in view; a group may be excluded
                    from the alignment
    e.  group       one small task per group: local name type and the group's
                    deviations from the KG-wide roles
    f.  revision    open flags resolved or dismissed, findings condensed

The entities to align are those in at least one group and in no excluded group.
Partitions and groups are independent: a group is defined by its own
membership pattern, never by a partition (stage d starts with one group per
partition, as a copy). Partitions are read only by stage b, the roles stage
(coverage per partition) and the competition (alignment_state), so
they can be dropped without touching groups.

A group stores only its deviations (added, removed); its effective roles are
derived: KG-wide roles used in the group, minus removed, plus added. The group
task shows the LLM the effective roles and computes the delta when it finishes
(diff_roles), so the LLM never edits a delta itself. An entity in several groups
gets the KG-wide roles minus the removed of all its groups, plus their added
(entity_roles).

Findings are short notes of a stage for the stages after it and for later
prompts (they replace the structural notes when these are missing). Later
stages read them as unverified observations and report a contradiction as a
flag instead of changing an earlier stage's result.
"""
from __future__ import annotations

from collections import defaultdict
from enum import StrEnum, auto
import re
from typing import Any, Iterable, Literal, Mapping, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from grasp.tasks.om.foundation.rendering import Item, Section, SectionContent
from grasp.tasks.om.foundation.types import IRI, LocalNameType
from grasp.tasks.om.foundation.rdf_vocabulary import RDF_TYPE, VOCABULARY, VOCABULARY_PREFIXES

INVERSE_PREFIX = "^"
PATH_SEPARATOR = "/"


class Stage(StrEnum):
    SPACE = "space"
    PARTITIONS = "partitions"
    ROLES = "roles"
    GROUPS = "groups"
    GROUP = "group"
    REVISION = "revision"


STAGES = list(Stage)  # in pipeline order


class RelationAlgebra(BaseModel):
    symmetric: bool = False
    transitive: bool = False
    inverse_of: str | None = None


def without_enclosing_braces(pattern: str) -> str:
    """
    The pattern without braces that enclose all of it: a pattern is the
    inside of a group, and in a group of its own ?s would be unbound
    ({ BIND(STR(?s) AS ?o) } binds no ?o). Kept: { A } UNION { B } (two
    groups) and a subquery ({ SELECT ... }). Braces in string literals
    (e.g. a regex "\\d{4}") do not count.
    """
    stripped = pattern.strip()
    while stripped.startswith("{") and stripped.endswith("}"):
        depth, quote, escaped = 0, None, False
        for index, character in enumerate(stripped):
            if quote:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in "\"'":
                quote = character
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0 and index < len(stripped) - 1:
                    return stripped  # the first group closes before the end
        inside = stripped[1:-1].strip()
        if inside.upper().startswith("SELECT"):
            return stripped
        stripped = inside
    return stripped


class PredicatePattern(BaseModel):
    # unknown fields (e.g. a misspelled "pattern") are errors, not silently dropped
    model_config = ConfigDict(extra="forbid")
    patterns: list[str] = Field(default=[], description="SPARQL patterns over ?s (the entity) and ?o (its value or a related entity), each an alternative; a pattern may compute ?o, e.g. BIND(REPLACE(STR(?s), \".*#\", \"\") AS ?o). Write a pattern as the inside of a group, without enclosing { } (they are removed): inside a group of its own, ?s is unbound.")
    predicates: list[IRI] = Field(default=[], description="Predicate IRIs: ?s <predicate> ?o, each an alternative.")

    @field_validator("patterns")
    @classmethod
    def patterns_without_enclosing_braces(cls, patterns: list[str]) -> list[str]:
        return [without_enclosing_braces(pattern) for pattern in patterns]

    @model_validator(mode='after')
    def not_empty(self) -> "PredicatePattern":
        if not self.patterns and not self.predicates:
            raise ValueError("SPARQL pattern or predicates required")
        return self

    def sparql_branches(self) -> list[str]:
        """
        The alternatives for querying: each a pattern over ?s and ?o: the
        predicates as one path (<a>|<b>), then each pattern. Not joined by
        UNION: a pattern may need ?s bound before it (BIND(... STR(?s) ...)),
        and an alternative of a UNION is evaluated on its own; so each
        alternative is queried in the block that binds ?s.
        """
        predicate_path = "|".join(f"<{p}>" for p in self.predicates)
        return ([f"?s {predicate_path} ?o"] if predicate_path else []) + list(self.patterns)

    def inverted(self) -> PredicatePattern:
        """
        The pattern in the other direction: ?s and ?o
        swapped in every alternative (e.g. "the parts
        of ?s" from "?s is part of ?o").
        """
        return PredicatePattern(
            patterns=[
                swap(alternative) for alternative in self.sparql_branches()
                ]
            )


def swap(pattern: str) -> str:
    return re.sub(
        r"\?(s|o)\b",
        lambda match: "?o" if match.group(1) == "s" else "?s", pattern
        )


def compose(
        first: PredicatePattern, second: PredicatePattern, step: int
        ) -> PredicatePattern:
    """
    A path of the form 'first/second': ?o of first and ?s
    of second joined in ?path{step}; the other variables of each
    get a suffix so that they do not clash.
    """
    link = f"?path{step}"

    def rename(branch: str, subject: str, value: str, suffix: str) -> str:
        return re.sub(
            r"\?(\w+)\b",
            lambda match: subject if match.group(1) == "s" else value if match.group(1) == "o"
            else f"?{match.group(1)}_{suffix}",
            branch)

    return PredicatePattern(patterns=[
        f"{rename(first_branch, '?s', link, f'{step}a')} . {rename(second_branch, link, '?o', f'{step}b')}"
        for first_branch in first.sparql_branches()
        for second_branch in second.sparql_branches()
        ])


class StructureRelationKind(StrEnum):
    SUBSUMPTION = auto()
    MEMBERSHIP = auto()
    PARTONOMY = auto()
    DISJOINTNESS = auto()
    OTHER = auto()


DEFAULT_KIND_ALGEBRA = {
    StructureRelationKind.SUBSUMPTION: {"transitive": True},
    StructureRelationKind.DISJOINTNESS: {"symmetric": True},
}


class StructureRelation(RelationAlgebra):
    name: str
    kind: StructureRelationKind
    predicate_pattern: PredicatePattern
    source: Literal["default", "declared", "llm"] = "default"

    @model_validator(mode="after")
    def kind_defaults(self) -> "StructureRelation":
        manual_args = self.model_fields_set
        for name, value in DEFAULT_KIND_ALGEBRA.get(self.kind, {}).items():
            if name not in manual_args:
                setattr(self, name, value)
        return self


def _structure(
    name: str, kind: StructureRelationKind, predicates: list[str] = [],
        patterns: list[str] = [], **algebra
        ) -> StructureRelation:
    return StructureRelation(
        name=name, kind=kind,
        predicate_pattern=PredicatePattern(predicates=[IRI(p) for p in predicates], patterns=patterns),
        **algebra,
    )


_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_OWL = "http://www.w3.org/2002/07/owl#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
_PART_OF = "http://purl.obolibrary.org/obo/BFO_0000050"
_NOT_VOCABULARY = " && ".join(f'!STRSTARTS(STR(?o), "{ns}")' for ns in VOCABULARY)


# default structure relations, read from ?s (specific) to ?o (general); the
# default profile keeps those used in a partition
DEFAULT_STRUCTURE_RELATIONS = [
    # without the blank nodes of OWL restrictions (part_of, existential_restriction)
    _structure("subclass_of", StructureRelationKind.SUBSUMPTION,
               patterns=[f"?s <{_RDFS}subClassOf> ?o . FILTER(isIRI(?o))"]),
    _structure("subproperty_of", StructureRelationKind.SUBSUMPTION, [_RDFS + "subPropertyOf"]),
    # skos:broader is not transitive by the SKOS specification, only skos:broaderTransitive
    _structure("broader", StructureRelationKind.SUBSUMPTION, [_SKOS + "broader"], transitive=False),
    _structure("broader_transitive", StructureRelationKind.SUBSUMPTION, [_SKOS + "broaderTransitive"]),
    # types from the vocabulary (owl:Class, owl:NamedIndividual) declare the kind, not a membership
    _structure("instance_of", StructureRelationKind.MEMBERSHIP,
               patterns=[f"?s <{RDF_TYPE}> ?o . FILTER(isIRI(?o) && {_NOT_VOCABULARY})"]),
    _structure("in_scheme", StructureRelationKind.MEMBERSHIP, [_SKOS + "inScheme"]),
    _structure("part_of", StructureRelationKind.PARTONOMY, [_PART_OF],
               patterns=[f"?s <{_RDFS}subClassOf> [ <{_OWL}onProperty> <{_PART_OF}> ; <{_OWL}someValuesFrom> ?o ]"]),
    # any existential restriction; often a partonomy under a KG-specific property
    # (e.g. UNDEFINED_part_of in the OAEI Anatomy track), which the profile task can tell
    _structure("disjoint_with", StructureRelationKind.DISJOINTNESS,
               [_OWL + "disjointWith", _OWL + "propertyDisjointWith"]),
]


class PredicateRoles(BaseModel):
    """
    Roles of predicates and patterns, KG-wide or as a group's delta.
    Patterns read from ?s, the entity, to ?o, the value; a pattern may compute
    ?o (e.g. a code from the IRI with BIND), which replaces code_sparql and
    definition_sparql of the old profile.
    """
    model_config = ConfigDict(extra="forbid")
    names: PredicatePattern | None = None
    synonyms: PredicatePattern | None = None
    definitions: PredicatePattern | None = None
    codes: PredicatePattern | None = None
    cross_references: PredicatePattern | None = None
    noise: PredicatePattern | None = None
    structure: list[StructureRelation] = []

    @model_validator(mode="after")
    def validate_structure_relation_names(self) -> Self:
        clashing_structure_relation_names = [
            name for name in self.structure_relation_names()
            if name in ROLE_FIELDS
            or name in set(StructureRelationKind)
            or name.startswith(INVERSE_PREFIX)
        ]
        if (
            clashing_structure_relation_names
            or len(set(self.structure_relation_names())) != len(self.structure)
                ):
            raise ValueError(
                f"structure relation names must be unique, "
                "no role name, no structure relation kind name, "
                f"and not start with {INVERSE_PREFIX!r}: "
                + str(
                    clashing_structure_relation_names
                    or self.structure_relation_names()
                )
            )

        return self

    def structure_relation_names(self) -> list[str]:
        return [relation.name for relation in self.structure]

    def predicate_pattern_by_name(self, name: str) -> PredicatePattern | None:
        """
        The pattern of a value source by name: a role ("names"),
        a structure relation ("part_of") or its inverse ("^part_of"),
        a relation kind (subsumption, membership, …) or its inverse,
        or a path of them joined by '/' ("subclass_of/names":
        the names of the superclasses).
        'None' means a role is not set yet.
        Raises ValueError for an unknown name.
        """
        if PATH_SEPARATOR in name:
            step_names = name.split(PATH_SEPARATOR)
            steps = [
                self.predicate_pattern_by_name(step)
                for step in step_names
                ]
            if any(step is None for step in steps):
                return None  # a role of the path is not set
            path = steps[0]
            for step_number, step in enumerate(steps[1:], start=1):
                path = compose(path, step, step_number)
            return path
        if name in ROLE_FIELDS:
            return getattr(self, name)  # may be 'None'
        relation_name = name.removeprefix(INVERSE_PREFIX)
        if relation_name in set(StructureRelationKind):
            of_kind = [
                relation.predicate_pattern for relation in self.structure
                if relation.kind == relation_name
            ]
            if not of_kind:
                return None
            pattern = PredicatePattern(
                predicates=list(
                    dict.fromkeys(
                        p for pattern in of_kind for p in pattern.predicates
                        )
                    ),
                patterns=list(
                    dict.fromkeys(
                        p for pattern in of_kind for p in pattern.patterns
                        )
                )
            )
            return pattern.inverted() if name.startswith(INVERSE_PREFIX) else pattern

        relation = next(
            (relation for relation in self.structure if relation.name == relation_name),
            None,
            )
        if relation is None:
            raise ValueError(
                f"{name!r} is neither a role {ROLE_FIELDS}, a structure relation "
                f"{self.structure_relation_names()} nor a relation kind "
                f"{[str(kind) for kind in StructureRelationKind]} (prefix {INVERSE_PREFIX!r}: the "
                f"other direction; join names with {PATH_SEPARATOR!r} for a path)"
            )
        return (
            relation.predicate_pattern.inverted()
            if name.startswith(INVERSE_PREFIX)
            else relation.predicate_pattern
            )


# the role fields of Roles
type RoleName = Literal["names", "synonyms", "definitions", "codes", "cross_references", "noise"]
ROLE_FIELDS: tuple[RoleName, ...] = get_args(RoleName.__value__)  # a type alias keeps its Literal in __value__
assert set(PredicateRoles.model_fields) == {*ROLE_FIELDS, "structure"}, "RoleName and the fields of Roles differ"


class RoleElementKind(StrEnum):
    PREDICATE = auto()            # one predicate of a role or relation
    PATTERN = auto()               # one SPARQL pattern of a role or relation
    RELATION_PROPERTIES = auto()  # transitive, symmetric etc.


class RelationProperties(BaseModel):
    """
    A structure relation without its predicate pattern.
    """
    model_config = ConfigDict(frozen=True)
    kind: StructureRelationKind
    symmetric: bool = False
    transitive: bool = False
    inverse_of: str | None = None
    # provenance, not part of the definition: equal
    # properties set by different stages are one element
    source: Literal["default", "declared", "llm"] = Field(
        default="default", exclude=False, repr=False
        )


class RoleElement(BaseModel):
    """
    The smallest unit of Roles: one predicate or pattern of a role field or a
    structure relation, or the properties of a structure relation.
    """
    model_config = ConfigDict(frozen=True)
    role_name: str  # a role field ("names") or a relation name ("subclass_of")
    is_structure_relation: bool
    kind: RoleElementKind
    definition: IRI | str | RelationProperties  # predicate, SPARQL pattern or properties

    def __hash__(self) -> int:
        return hash(
            (self.role_name,
             self.is_structure_relation,
             self.kind,
             # source of the properties does not count (see eq)
             hashable_definition(self.definition))
            )

    def __eq__(self, other: "RoleElement") -> bool:
        return (
            isinstance(other, RoleElement) and hash(self) == hash(other)
            and hashable_definition(self.definition) == hashable_definition(other.definition)
            )

    def sparql(self) -> str | None:
        """
        The pattern over ?s and ?o of this element.
        None for relation properties.
        """
        match self.kind:
            case RoleElementKind.PREDICATE:
                return f"?s <{self.definition}> ?o"
            case RoleElementKind.PATTERN:
                if isinstance(self.definition, RelationProperties):
                    raise ValueError(
                        "RoleElement has kind 'RELATION_PROPERTIES' "
                        "but its definition is a RelationProperties."
                        )
                return self.definition
            case _:
                return None


def hashable_definition(definition: IRI | str | RelationProperties) -> str | tuple:
    """
    The definition as a hashable value for equality: relation properties
    without their source (who set them does not change what they mean).
    """
    if isinstance(definition, RelationProperties):
        return tuple(sorted(definition.model_dump(exclude={"source"}).items()))
    return definition


type RoleElements = frozenset[RoleElement]
# number of entities of a group (or partition) that an element covers (as ?s)
type RoleElementUsage = dict[RoleElement, int]


def role_elements_of(roles: PredicateRoles) -> RoleElements:
    role_elements: set[RoleElement] = set()
    for field in ROLE_FIELDS:
        if (pattern := getattr(roles, field)) is not None:
            add_predicate_pattern(role_elements, field, False, pattern)
        for structure_relation in roles.structure:
            add_predicate_pattern(
                role_elements,
                structure_relation.name,
                True,
                structure_relation.predicate_pattern
                )
            role_elements.add(
                RoleElement(
                    role_name=structure_relation.name,
                    is_structure_relation=True,
                    kind=RoleElementKind.RELATION_PROPERTIES,
                    definition=RelationProperties(
                        **structure_relation.model_dump(
                            exclude={"name", "predicate_pattern"}
                            )
                        )
                    )
                )
    return frozenset(role_elements)


def add_predicate_pattern(
        role_elements: set[RoleElement],
        role_name: str,
        is_structure_relation: bool,
        pattern: PredicatePattern
        ):
    role_elements.update(
        RoleElement(
            role_name=role_name,
            is_structure_relation=is_structure_relation,
            kind=RoleElementKind.PREDICATE,
            definition=p
            )
        for p in pattern.predicates
        )
    role_elements.update(
        RoleElement(
            role_name=role_name,
            is_structure_relation=is_structure_relation,
            kind=RoleElementKind.PATTERN,
            definition=p
            )
        for p in pattern.patterns
        )


def roles_from_elements(elements: RoleElements) -> PredicateRoles:
    """
    A structure relation needs its properties and at least one
    predicate or pattern, otherwise it is left out (e.g. not used in a group).
    """
    # (is_structure_relation, field_name) -> predicate / pattern
    predicates: dict[tuple[bool, str], list[IRI]] = defaultdict(list)
    patterns: dict[tuple[bool, str], list[str]] = defaultdict(list)
    # sturture field name -> properties
    properties: dict[str, RelationProperties] = {}

    for element in sorted(elements, key=role_element_order):
        key = (element.is_structure_relation, element.role_name)
        match element.kind:
            case RoleElementKind.PREDICATE:
                predicates[key].append(element.definition)
            case RoleElementKind.PATTERN:
                patterns[key].append(element.definition)
            case RoleElementKind.RELATION_PROPERTIES:
                properties[element.role_name] = element.definition

    def predicate_pattern_of(key: tuple[bool, str]) -> PredicatePattern | None:
        if not predicates[key] and not patterns[key]:
            return None
        return PredicatePattern(predicates=predicates[key], patterns=patterns[key])

    non_structure_roles = {
        field: p for field in ROLE_FIELDS
        if (p := predicate_pattern_of((False, field))) is not None
        }
    structure_roles = [
        StructureRelation(name=name, predicate_pattern=p, **props.model_dump())
        for name, props in properties.items()
        if (p := predicate_pattern_of((True, name))) is not None
    ]

    return PredicateRoles(structure=structure_roles, **non_structure_roles)


def role_element_order(element: RoleElement) -> tuple[bool, int, str, int, str]:
    """
    Sort key of role elements, so that sets of them give the same Roles in every
    run: role fields first (in the order of ROLE_FIELDS), then structure
    relations by name; within one role, predicates, patterns, properties
    (the order of RoleElementKind), then by definition.
    """
    field_position = 0 if element.is_structure_relation else ROLE_FIELDS.index(element.role_name)
    return (
        element.is_structure_relation,
        field_position,
        element.role_name,
        list(RoleElementKind).index(element.kind),
        str(hashable_definition(element.definition)),
        )


def role_elements_with_added(
        elements: RoleElements, added: Iterable[RoleElement]
        ) -> RoleElements:
    """
    The elements plus the added ones; added relation
    properties replace the relation's old ones.
    """
    added = set(added)
    replaced = {
        element.role_name for element in added
        if element.kind == RoleElementKind.RELATION_PROPERTIES
    }
    kept = {
        element for element in elements
        if not (
            element.kind == RoleElementKind.RELATION_PROPERTIES
            and element.role_name in replaced
            )
    }
    return frozenset(kept | added)


def diff_roles(
        derived: PredicateRoles, final: PredicateRoles
        ) -> tuple[list[RoleElement], list[RoleElement]]:
    before, after = role_elements_of(derived), role_elements_of(final)
    return (
        sorted(after - before, key=role_element_order),
        sorted(before - after, key=role_element_order)
        )


def derived_group_roles(
    kg_roles: PredicateRoles, element_usage: RoleElementUsage
        ) -> PredicateRoles:
    return roles_from_elements(
        frozenset(
            element for element in role_elements_of(kg_roles)
            if element.kind == RoleElementKind.RELATION_PROPERTIES
            or element_usage.get(element, 0) > 0
        )
    )


def used_group_roles(
        kg_roles: PredicateRoles, group: GroupProfile, usage: RoleElementUsage
        ) -> PredicateRoles:
    elements = role_elements_of(derived_group_roles(kg_roles, usage)) - set(group.removed)
    return roles_from_elements(role_elements_with_added(elements, group.added))


def effective_roles_of_entity(
        kg_roles: PredicateRoles, groups: Iterable[GroupProfile]
        ) -> PredicateRoles:
    """
    Roles of an entity in these groups: the KG-wide roles minus the removed
    of all groups, plus their added (an addition wins over a removal by
    another group. Of two groups that change the same relation's properties,
    the later defined one wins).
    """
    groups = list(groups)
    elements = role_elements_of(kg_roles)
    for group in groups:
        elements -= set(group.removed)
    for group in groups:
        elements = role_elements_with_added(elements, group.added)
    return roles_from_elements(elements)


def entities_by_effective_roles(
        kg_profile: KgProfile, group_names_by_entity: Mapping[IRI, Iterable[str]]
) -> dict[frozenset[str], tuple[PredicateRoles, set[IRI]]]:
    """
    The entities divided by their combination of groups of the profile, each
    part with its effective roles (effective_roles_of_entity: the KG-wide roles with the
    deviations of its groups). Names that are no group of the profile (inline
    groups of a population) are left out. Entities in the same groups have the
    same roles, so each part can be retrieved in one batch in later steps.
    """
    profile_group_names = set(kg_profile.group_names())
    entities_by_group_combination: dict[frozenset[str], set[IRI]] = {}
    for entity, group_names in group_names_by_entity.items():
        entities_by_group_combination.setdefault(
            frozenset(group_names) & profile_group_names, set()
            ).add(entity)
    return {
        group_names: (
            effective_roles_of_entity(
                kg_profile.roles, map(kg_profile.get_group, group_names)
                ),
            members
            ) for group_names, members in entities_by_group_combination.items()
    }


def check_structure_relations(relations: list[StructureRelation]) -> list[str]:
    """
    Problems of a list of structure relations: duplicate names, inverse_of
    naming no relation of the list, or inverses that do not point back.
    """
    problems = []
    by_name: dict[str, StructureRelation] = {}
    for relation in relations:
        if relation.name in by_name:
            problems.append(f"duplicate structure relation name {relation.name}")
        by_name[relation.name] = relation
    for relation in relations:
        if relation.inverse_of is None:
            continue
        inverse = by_name.get(relation.inverse_of)
        if inverse is None:
            problems.append(f"{relation.name}: inverse_of names no structure relation ({relation.inverse_of})")
        elif inverse.inverse_of not in (None, relation.name):
            problems.append(f"{relation.name} is inverse_of {inverse.name}, but {inverse.name} is inverse_of {inverse.inverse_of}")
    return problems


class Partition(BaseModel):
    """A partition of stage b. Partitions are exclusive: in their order, an
    entity belongs to the first whose membership matches."""
    model_config = ConfigDict(extra="forbid")
    name: str
    membership_sparql: str  # SPARQL pattern over ?s, the inside of a group
    features: str = ""      # what distinguishes its entities from the rest


class WithinKgNonCompetingPartitionPair(BaseModel):
    """Two partitions of one KG whose entities do not compete with each other
    (e.g. properties and classes): neither is a candidate or a rival for the
    other. Symmetric: the order of the two partitions does not matter."""
    model_config = ConfigDict(extra="forbid")
    partition_a: str
    partition_b: str
    reason: str = ""
    set_by: str = "partitions"  # the stage that set it (stage b, or a stage of the pair)

    def partitions(self) -> frozenset[str]:
        return frozenset((self.partition_a, self.partition_b))


class GroupOrigin(StrEnum):
    PARTITION = "partition"  # copy of a partition (the start of stage d)
    FORMED = "formed"        # formed in stage d


class GroupDefinition(BaseModel):
    """
    Which entities form a group: its name and membership. Also the form in
    which a strategy defines a group inline (candidates.Population).
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    membership_sparql: str  # SPARQL pattern over ?s, the inside of a group
    origin: GroupOrigin = GroupOrigin.FORMED
    parent: str | None = None  # a group it is part of (for reading; membership is its own)
    features: str = ""         # what distinguishes its entities, to recognize them


class GroupProfile(BaseModel):
    """A group of the profile: its definition and what the stages decided about it."""
    model_config = ConfigDict(extra="forbid")
    definition: GroupDefinition
    # not aligned (e.g. image pages, helper nodes), set by stage d; an entity in an
    # excluded group is not aligned even if it is in other groups
    excluded: bool = False
    exclusion_reason: str = ""
    local_name_type: LocalNameType | None = None  # set by stage e (or its default)
    added: list[RoleElement] = []   # group-specific changes on roles
    removed: list[RoleElement] = []  # roles used in the group, but uninformative there

    @property
    def name(self) -> str:
        return self.definition.name


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    stage: Stage
    content: str
    evidence_query: str


class FlagStatus(StrEnum):
    OPEN = auto()
    RESOLVED = auto()
    DISMISSED = auto()


class Flag(BaseModel):
    """A later stage's objection to an earlier stage's result."""
    model_config = ConfigDict(extra="forbid")
    id: str
    raised_by_stage: Stage  # stage that raises the flag
    concerns: str  # finding id, flag id or stage it concerns
    content: str
    evidence_query: str
    status: FlagStatus = FlagStatus.OPEN
    resolution_description: str | None = None


def unknown_name_message(kind: str, name: str, existing_names: list[str]) -> str:
    """Error message for a partition or group (kind) name the profile does not have."""
    message = f"{name} is not an existing {kind} name in this knowledge graph."
    if not existing_names:
        return message + f" No {kind}s defined in this knowledge graph yet."
    return message + f" Choose between: {existing_names}"


class KgProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kg_name: str
    triples: int | None = None
    entity_space: list[str] = []
    partitions: list[Partition] = []  # in order: the first match wins
    non_competing_pairs: list[WithinKgNonCompetingPartitionPair] = []
    roles: PredicateRoles = PredicateRoles()
    groups: list[GroupProfile] = []
    findings: dict[str, Finding] = Field(default_factory=dict)
    flags: dict[str, Flag] = Field(default_factory=dict)

    @model_validator(mode="after")
    def non_competing_pairs_name_partitions(self) -> Self:
        for pair in self.non_competing_pairs:
            unknown = pair.partitions() - set(self.partition_names())
            if unknown or pair.partition_a == pair.partition_b:
                raise ValueError(f"non-competing pair {pair.partition_a} - {pair.partition_b}: "
                                 + (f"unknown partitions {sorted(unknown)}" if unknown else "a partition with itself"))
        return self

    @model_validator(mode="after")
    def partition_group_names_unique(self) -> Self:
        for kind, names in (
                ("partition", self.partition_names()), ("group", self.group_names())
                ):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"Duplicate {kind} names: {duplicates}")
        return self

    def partition_names(self) -> list[str]:
        return [partition.name for partition in self.partitions]

    def group_names(self) -> list[str]:
        return [group.name for group in self.groups]

    def get_partition(self, partition_name: str) -> Partition:
        partition = next((partition for partition in self.partitions if partition.name == partition_name), None)
        if partition is None:
            raise ValueError(unknown_name_message("partition", partition_name, self.partition_names()))
        return partition

    def get_group(self, group_name: str) -> GroupProfile:
        group = next((group for group in self.groups if group.name == group_name), None)
        if group is None:
            raise ValueError(unknown_name_message("group", group_name, self.group_names()))
        return group

    def get_effective_roles_of_all_groups(self) -> list[PredicateRoles]:
        """
        The KG-wide roles and the effective roles of each group that is not excluded
        """
        return [
            self.roles,
            *(
                effective_roles_of_entity(self.roles, [group]) 
                for group in self.groups if not group.excluded
                )
            ]

    def is_role_set(self, role_name: str) -> bool:
        """Whether a role or relation kind has a pattern KG-wide or in a group: it can give values."""
        return any(
            roles.predicate_pattern_by_name(role_name) is not None
            for roles in self.get_effective_roles_of_all_groups()
            )

    def competes(self, partition_a: str, partition_b: str) -> bool:
        """Whether entities of the two partitions of this KG compete with each other."""
        return all(pair.partitions() != {partition_a, partition_b} for pair in self.non_competing_pairs)

    def excluded_group_names(self) -> list[str]:
        return [group.name for group in self.groups if group.excluded]

    def excluded_group_membership_sparqls(self) -> list[str]:
        return [group.definition.membership_sparql for group in self.groups if group.excluded]

    def with_group(self, group: GroupProfile | GroupDefinition) -> "KgProfile":
        """A copy with the group appended."""
        copied = self.model_copy(deep=True)
        copied.append_group(group)
        return copied

    def with_replaced_group(self, group: GroupProfile) -> "KgProfile":
        """A copy with the group of the same name replaced by this one."""
        self.get_group(group.name)
        groups = [group if existing.name == group.name else existing for existing in self.groups]
        return self.model_copy(update={"groups": groups}, deep=True)

    def has_group(self, group_name: str) -> bool:
        return group_name in self.group_names()

    def append_group(self, group: GroupDefinition | GroupProfile):
        match group:
            case GroupDefinition():
                group_profile = GroupProfile(definition=group)
            case GroupProfile():
                group_profile = group
            case _:
                raise TypeError(f"Expected GroupDefinition or GroupProfile, got {type(group)}")
        if self.has_group(group_profile.name):
            raise ValueError(
                f"{group_profile.name} already exists in this knowledge graph."
                f" Please choose a new name. Already existing: {self.group_names()}"
                )
        self.groups.append(group_profile)


class ProfileStageOutput(BaseModel):
    kg_name: str
    stage: Stage
    finished: bool
    rationale: str | None
    profile: KgProfile
    default: KgProfile
    stats: dict[str, Any]


# --- description for prompts -------------------------------------------------------

# TODO: Why not use GRASPs prefixed version?
def short_iri(iri: str) -> str:
    """
    Vocabulary IRIs by their prefixed name (rdfs:label), others as they are.
    """
    for namespace, prefix in VOCABULARY_PREFIXES.items():
        if iri.startswith(namespace):
            return f"{prefix}:{iri[len(namespace):]}"
    return iri


def describe_predicate_pattern(pattern: PredicatePattern) -> str:
    parts = []
    if pattern.predicates:
        parts.append("predicates: " + ", ".join(
            short_iri(predicate) for predicate in pattern.predicates
        ) + ".")
    if pattern.patterns:
        parts.append("sparql patterns: " + ", ".join(
            f"`{sparql}`" for sparql in pattern.patterns
            ) + ".")
    return " ".join(parts)


def describe_predicate_pattern_items(pattern: PredicatePattern) -> list[Item]:
    """The predicates and the SPARQL patterns, one sub-item each."""
    items = []
    if pattern.predicates:
        items.append(Item("predicates:", [Item(short_iri(predicate)) for predicate in pattern.predicates]))
    if pattern.patterns:
        items.append(Item("sparql patterns:", [Item(f"`{sparql}`") for sparql in pattern.patterns]))
    return items


ROLE_DESCRIPTIONS: dict[RoleName, str] = {
    "names": "preferred names",
    "synonyms": "alternative names",
    "definitions": "textual definitions or descriptions",
    "codes": "codes of the entity in its own system (notations, IDs)",
    "cross_references": "codes or IRIs of the entity in other systems",
    "noise": "values that say nothing about the meaning of an entity (e.g. formatting, provenance, media files)",
}


# how a structure relation of a kind reads from ?s to ?o
RELATION_READINGS = {
    StructureRelationKind.SUBSUMPTION: "?s is narrower than ?o",
    StructureRelationKind.MEMBERSHIP: "?s is a member of ?o",
    StructureRelationKind.PARTONOMY: "?s is part of ?o",
    StructureRelationKind.DISJOINTNESS: "?s and ?o share no members",
    StructureRelationKind.OTHER: "?s is related to ?o",
}


def describe_structure_relation_heading(relation: StructureRelation) -> str:
    """
    E.g. "'subclass_of' (subsumption: ?s is narrower than ?o; transitive)".
    """
    algebra_description = [
        property_name for property_name in ("transitive", "symmetric")
        if getattr(relation, property_name)
        ]
    if relation.inverse_of:
        algebra_description.append(f"inverse of {relation.inverse_of}")
    algebra = "; " + ", ".join(algebra_description) if algebra_description else ""
    return f"'{relation.name}' ({relation.kind.value}: {RELATION_READINGS[relation.kind]}{algebra})"


def describe_structure_relation(relation: StructureRelation) -> str:
    """The relation in one line (e.g. in the role overview of the profile stages)."""
    return (f"{describe_structure_relation_heading(relation)}: "
            f"{describe_predicate_pattern(relation.predicate_pattern)}")


def describe_roles(roles: PredicateRoles) -> list[Item]:
    """Each role and structure relation with its predicates and patterns below it."""
    items = [
        Item(f"'{field.replace('_', ' ')}' ({ROLE_DESCRIPTIONS[field]}):", describe_predicate_pattern_items(pattern))
        for field in ROLE_FIELDS
        if (pattern := getattr(roles, field)) is not None
        ]
    items += [
        Item(f"structure relation {describe_structure_relation_heading(relation)}:",
             describe_predicate_pattern_items(relation.predicate_pattern))
        for relation in roles.structure
        ]
    return items


def describe_profile(profile: KgProfile, findings: bool = True) -> Section:
    """
    The profile as short notes for prompts.
    Patterns read from ?s, the entity, to ?o; structure relations from the
    specific to the general entity.
    """
    title = "Profile"
    content = []
    subsections: list[Section] = []

    if len(profile.entity_space) == 1:
        content.append(f"Entities to align have IRIs starting with <{profile.entity_space[0]}>.")
    elif profile.entity_space:
        content += ["Entities to align have IRIs starting with:"]
        content += [Item(f"<{space}>") for space in profile.entity_space]

    if profile.partitions:
        partition_subsection = Section(
            "Partitions (exclusive ontological meta-categories)"
            )
        partition_content = ["Partition names and features:"]
        partition_content += [
            Item(f"{partition.name}: {partition.features}")
            if partition.features
            else Item(partition.name)
            for partition in profile.partitions
            ]
        if profile.non_competing_pairs:
            partition_content += ["Partitions which do not compete with each other:"]
            # pairs of one partition with the same reason in one item
            partners_by_reason: dict[tuple[str, str], list[str]] = {}
            for pair in profile.non_competing_pairs:
                partners_by_reason.setdefault((pair.partition_a, pair.reason), []).append(pair.partition_b)
            partition_content += [
                Item(f"{partition} and {', '.join(partners)}" + (f": {reason}" if reason else ""))
                for (partition, reason), partners in partners_by_reason.items()
            ]
        partition_subsection.content = partition_content
        subsections.append(partition_subsection)

    if profile.groups:
        groups_subsection = Section("Groups (thematic categories, may overlap)")
        groups_content: list[SectionContent] = []
        # groups that only copy a partition are named once, not described
        partition_copies = [group.name for group in profile.groups
                            if is_partition_copy(group) and not group.excluded]
        if partition_copies:
            groups_content.append(f"Each of these partitions is also a group: {', '.join(partition_copies)}.")
        included_groups = ["Group names and features:"]
        excluded_groups = ["Excluded groups:"]
        for group in profile.groups:
            if is_partition_copy(group) and not group.excluded:
                continue
            if group.excluded:
                excluded_groups += [Item(
                    f"{group.name}" + (f": {group.exclusion_reason}"
                                       if group.exclusion_reason else "")
                )]
            else:
                included_groups += [Item(
                    f"{group.name}"
                    + (f" (part of '{group.definition.parent}')"
                       if group.definition.parent else "")
                    + (f": {group.definition.features}"
                       if group.definition.features else "")
                )]
        for group_descriptions in (included_groups, excluded_groups):
            groups_content += group_descriptions if len(group_descriptions) > 1 else []

        groups_subsection.content = groups_content
        subsections.append(groups_subsection)

    roles_items = describe_roles(profile.roles)
    if roles_items:
        roles_subsection = Section(
            "Important Predicates",
            ["Predicates map from ?s (entity) to ?o (value):", *roles_items]
            )
        subsections.append(roles_subsection)

    if findings and profile.findings:
        subsections.append(
            Section(
                "Findings", [
                    Item(finding.content)
                    for finding in profile.findings.values()
                    ]
                )
            )

    return Section(title, content, subsections)


def is_partition_copy(group: GroupProfile) -> bool:
    """A group that is a partition, unchanged and without own features."""
    return group.definition.origin == GroupOrigin.PARTITION and not group.definition.features
