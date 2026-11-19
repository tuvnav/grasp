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
from enum import StrEnum, auto
import re
from typing import Any, Iterable, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, model_validator

from grasp.tasks.om.foundation.rendering import Item, Section, SectionContent
from grasp.tasks.om.foundation.types import IRI, LocalNameType
from grasp.tasks.om.foundation.rdf_vocabulary import RDF_TYPE, VOCABULARY, VOCABULARY_PREFIXES

INVERSE_PREFIX = "^"


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


class PredicatePattern(BaseModel):
    # unknown fields (e.g. a misspelled "pattern") are errors, not silently dropped
    model_config = ConfigDict(extra="forbid")
    patterns: list[str] = []  # SPARQL patterns over ?s and ?o
    predicates: list[IRI] = []

    @model_validator(mode='after')
    def not_empty(self) -> "PredicatePattern":
        if not self.patterns and not self.predicates:
            raise ValueError("SPARQL pattern or predicates required")
        return self

    def branches_for_union(self) -> list[str]:
        """
        The alternatives for querying: each pattern, and each predicate on
        its own (not a path <a>|<b>, because Qlever resolves single predicates
        much faster than predicate paths).
        """
        return list(self.patterns) + [f"?s <{p}> ?o" for p in self.predicates]

    def inverted(self) -> "PredicatePattern":
        """
        The pattern in the other direction: ?s and ?o 
        swapped in every alternative (e.g. "the parts 
        of ?s" from "?s is part of ?o").
        """
        return PredicatePattern(
            patterns=[
                swap(alternative) for alternative in self.branches_for_union()
                ]
            )


def swap(pattern: str) -> str:
    return re.sub(
        r"\?(s|o)\b",
        lambda match: "?o" if match.group(1) == "s" else "?s", pattern
        )


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
    corresponds_to: str | None = None
    source: Literal["default", "declared", "llm"] = "default"

    @model_validator(mode="after")
    def kind_defaults(self) -> "StructureRelation":
        manual_args = self.model_fields_set
        for name, value in DEFAULT_KIND_ALGEBRA.get(self.kind, {}).items():
            if name not in manual_args:
                setattr(self, name, value)
        return self


def _structure(name: str, kind: StructureRelationKind, predicates: list[str] = [],
               patterns: list[str] = [], **algebra) -> StructureRelation:
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


class Roles(BaseModel):
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
            if name in ROLE_FIELDS or name.startswith(INVERSE_PREFIX)
        ]
        if (
            clashing_structure_relation_names
            or len(set(self.structure_relation_names())) != len(self.structure)
                ):
            raise ValueError(
                f"structure relation names must be unique, no role name and not start with "
                f"{INVERSE_PREFIX!r}: {
                    clashing_structure_relation_names or self.structure_relation_names()
                    }"
            )

        return self

    def structure_relation_names(self) -> list[str]:
        return [relation.name for relation in self.structure]

    def predicate_pattern_by_name(self, name: str) -> PredicatePattern | None:
        """
        The pattern of a value source by name: a role ("names"), a structure
        relation ("part_of") or its inverse ("^part_of"). 'None' means
        a role is not set yet.
        Raises ValueError for an unknown name.
        """
        if name in ROLE_FIELDS:
            return getattr(self, name)  # may be 'None'
        relation_name = name.removeprefix(INVERSE_PREFIX)
        relation = next(
            (relation for relation in self.structure if relation.name == relation_name),
            None,
            )
        if relation is None:
            raise ValueError(
                f"{name!r} is neither a role {ROLE_FIELDS} nor a structure relation "
                f"{self.structure_relation_names()} (prefix {INVERSE_PREFIX!r}: the "
                "other direction)"
            )
        return (
            relation.predicate_pattern.inverted()
            if name.startswith(INVERSE_PREFIX)
            else relation.predicate_pattern
            )


# the role fields of Roles; also used to refer to a role elsewhere (scope.ValuePatterns)
type RoleName = Literal["names", "synonyms", "definitions", "codes", "cross_references", "noise"]
ROLE_FIELDS: tuple[RoleName, ...] = get_args(RoleName.__value__)  # a type alias keeps its Literal in __value__
assert set(Roles.model_fields) == {*ROLE_FIELDS, "structure"}, "RoleName and the fields of Roles differ"


class Partition(BaseModel):
    """A partition of stage b. Partitions are exclusive: in their order, an
    entity belongs to the first whose membership matches."""
    model_config = ConfigDict(extra="forbid")
    name: str
    membership_sparql: str  # SPARQL group pattern over ?s
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
    CLASS = "class"          # a large class (rdf:type) of a large KG
    FORMED = "formed"        # formed in stage d


class GroupDefinition(BaseModel):
    """
    Which entities form a group: its name and membership. Also the form in
    which a stage defines a group inline (scope.Population).
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    membership_sparql: str  # SPARQL group pattern over ?s
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
    added: Roles = Roles()    # group-specific; a relation overrides the KG-wide one of its name
    removed: Roles = Roles()  # used in the group, but uninformative there

    @property
    def name(self) -> str:
        return self.definition.name




class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Stage
    text: str
    group: str | None = None  # the group (or in stage b the partition) it is about


class FlagStatus(StrEnum):
    OPEN = auto()
    RESOLVED = auto()
    DISMISSED = auto()


class Flag(BaseModel):
    """A later stage's objection to an earlier stage's result."""
    model_config = ConfigDict(extra="forbid")
    id: str
    raised_by_stage: Stage        # stage that raises the flag
    concerns_stage: Stage     # stage whose result it questions
    content: str
    concerns_group: str | None = None
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
    # triples of the KG (None: not countable); a large KG is profiled on samples (large_kg_sampling)
    triples: int | None = None
    large: bool = False
    entity_space: list[str] = []
    partitions: list[Partition] = []  # in order: the first match wins
    non_competing_pairs: list[WithinKgNonCompetingPartitionPair] = []
    roles: Roles = Roles()
    groups: list[GroupProfile] = []
    findings: list[Finding] = []
    flags: list[Flag] = []

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


# --- elements and deltas ------------------------------------------------------


# an element of a Roles: (field, "predicate" | "pattern", IRI or pattern) for the
# role fields, ("structure", "relation", name) for a structure relation
type ElementKey = tuple[str, str, str]
type Elements = dict[ElementKey, str | StructureRelation]


def role_elements(roles: Roles) -> Elements:
    elements: Elements = {}
    for field in ROLE_FIELDS:
        pattern: PredicatePattern | None = getattr(roles, field)
        if pattern is None:
            continue
        for predicate in pattern.predicates:
            elements[(field, "predicate", predicate)] = predicate
        for sparql in pattern.patterns:
            elements[(field, "pattern", sparql)] = sparql
    for relation in roles.structure:
        elements[("structure", "relation", relation.name)] = relation
    return elements


def roles_from_elements(elements: Elements) -> Roles:
    predicates: dict[str, list[IRI]] = {field: [] for field in ROLE_FIELDS}
    patterns: dict[str, list[str]] = {field: [] for field in ROLE_FIELDS}
    structure = []
    for (field, kind, _), value in elements.items():
        if kind == "relation":
            structure.append(value)
        elif kind == "predicate":
            predicates[field].append(IRI(value))
        else:
            patterns[field].append(value)
    fields = {
        field: PredicatePattern(predicates=predicates[field], patterns=patterns[field])
        for field in ROLE_FIELDS
        if predicates[field] or patterns[field]
    }
    return Roles(structure=structure, **fields)


def same_element(element_a: str | StructureRelation, element_b: str | StructureRelation) -> bool:
    # a relation edited without change stays the same, whoever set it
    if isinstance(element_a, StructureRelation) and isinstance(element_b, StructureRelation):
        return element_a.model_dump(exclude={"source"}) == element_b.model_dump(exclude={"source"})
    return element_a == element_b


# number of entities of a group (or partition) that an element covers (as ?s)
type Usage = dict[ElementKey, int]


def derived_roles(kg_roles: Roles, usage: Usage) -> Roles:
    """KG-wide roles restricted to the elements used in a group."""
    return roles_from_elements({
        key: value for key, value in role_elements(kg_roles).items() if usage.get(key, 0) > 0
    })


def effective_roles(kg_roles: Roles, group: GroupProfile, usage: Usage) -> Roles:
    """Roles of a group: KG-wide roles used in it, minus removed, plus added."""
    elements = role_elements(derived_roles(kg_roles, usage))
    for key in role_elements(group.removed):
        elements.pop(key, None)
    elements.update(role_elements(group.added))
    return roles_from_elements(elements)


def entity_roles(kg_roles: Roles, groups: Iterable[GroupProfile]) -> Roles:
    """Roles of an entity in these groups: the KG-wide roles minus the removed
    of all groups, plus their added (an addition wins over a removal by
    another group)."""
    groups = list(groups)
    elements = role_elements(kg_roles)
    for group in groups:
        for key in role_elements(group.removed):
            elements.pop(key, None)
    for group in groups:
        elements.update(role_elements(group.added))
    return roles_from_elements(elements)


def diff_roles(derived: Roles, final: Roles) -> tuple[Roles, Roles]:
    """(added, removed) that turn derived into final. A relation changed
    under its name is added (overriding), not removed and added."""
    before, after = role_elements(derived), role_elements(final)
    added = {key: value for key, value in after.items() if key not in before or not same_element(before[key], value)}
    removed = {key: value for key, value in before.items() if key not in after}
    return roles_from_elements(added), roles_from_elements(removed)


def check_structure(relations: list[StructureRelation]) -> list[str]:
    """Problems of a list of structure relations: duplicate names, inverse_of
    naming no relation of the list, or inverses that do not point back."""
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
            f"{{{sparql}}}" for sparql in pattern.patterns
            ) + ".")
    return " ".join(parts)


def describe_predicate_pattern_items(pattern: PredicatePattern) -> list[Item]:
    """The predicates and the SPARQL patterns, one sub-item each."""
    items = []
    if pattern.predicates:
        items.append(Item("predicates:", [Item(short_iri(predicate)) for predicate in pattern.predicates]))
    if pattern.patterns:
        items.append(Item("sparql patterns:", [Item(f"{{{sparql}}}") for sparql in pattern.patterns]))
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


def describe_roles(roles: Roles) -> list[Item]:
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


def describe_profile(profile: KgProfile,
                     findings: bool = True,
                     groups: Iterable[str] | None = None
                     ) -> Section:
    """
    The profile as short notes for prompts.
    Patterns read from ?s, the entity, to ?o; structure relations from the
    specific to the general entity. With groups, only the findings about the
    whole KG and about these groups (or partitions).
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
        shown_findings = [finding for finding in profile.findings
                          if groups is None or finding.group is None or finding.group in groups]
        if shown_findings:
            subsections.append(Section("Findings", [
                Item(finding.text if finding.group is None else f"{finding.group}: {finding.text}")
                for finding in shown_findings
            ]))

    return Section(title, content, subsections)


def is_partition_copy(group: GroupProfile) -> bool:
    """A group that is a partition, unchanged and without own features."""
    return group.definition.origin == GroupOrigin.PARTITION and not group.definition.features
