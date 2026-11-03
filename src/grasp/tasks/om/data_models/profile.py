"""
Data model of the staged KG profile.

Stages, each a GRASP task on one KG; each reads the profile of the stages
before and adds to it:
    a.  space       entity space (namespaces of the entities to align)
    b1. partitions  exclusive partitions: the hard boundaries between kinds of
                    entities, from the KG alone, and which partitions of the KG
                    do not compete with each other
    c.  roles       KG-wide roles: names, synonyms, ..., structure relations
    b2. groups      overlapping groups of entities recognizable by their own
                    features, with the task in view; a group may be excluded
                    from the alignment
    d.  group       one small task per group: local name type and the group's
                    deviations from the KG-wide roles

The entities to align are those in at least one group and in no excluded group.
Partitions and groups are independent: a group is defined by its own
membership pattern, never by a partition (stage b2 starts with one group per
partition, as a copy). Partitions are read only by stage b1, the roles stage
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
from enum import StrEnum
from typing import Iterable, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, model_validator

from grasp.tasks.om.data_models.patterns import PredicatePattern, StructureRelation
from grasp.tasks.om.foundation.types import IRI, NameType
from grasp.tasks.om.foundation.vocabulary import VOCABULARY_PREFIXES


class Stage(StrEnum):
    SPACE = "space"
    PARTITIONS = "partitions"
    ROLES = "roles"
    GROUPS = "groups"
    GROUP = "group"


STAGES = list(Stage)  # in pipeline order


class Roles(BaseModel):
    """Roles of predicates and patterns, KG-wide or as a group's delta.
    Patterns read from ?s, the entity, to ?o, the value; a pattern may compute
    ?o (e.g. a code from the IRI with BIND), which replaces code_sparql and
    definition_sparql of the old profile."""
    model_config = ConfigDict(extra="forbid")
    names: PredicatePattern | None = None
    synonyms: PredicatePattern | None = None
    definitions: PredicatePattern | None = None
    codes: PredicatePattern | None = None
    cross_references: PredicatePattern | None = None
    noise: PredicatePattern | None = None
    structure: list[StructureRelation] = []


# the role fields of Roles; also used to refer to a role elsewhere (scope.ValuePatterns)
type RoleName = Literal["names", "synonyms", "definitions", "codes", "cross_references", "noise"]
ROLE_FIELDS: tuple[RoleName, ...] = get_args(RoleName.__value__)  # a type alias keeps its Literal in __value__
assert set(Roles.model_fields) == {*ROLE_FIELDS, "structure"}, "RoleName and the fields of Roles differ"


class Partition(BaseModel):
    """A partition of stage b1. Partitions are exclusive: in their order, an
    entity belongs to the first whose membership matches."""
    model_config = ConfigDict(extra="forbid")
    name: str
    membership_sparql: str  # SPARQL group pattern over ?entity
    features: str = ""      # what distinguishes its entities from the rest


class WithinKgNonCompetingPair(BaseModel):
    """Two partitions of one KG whose entities do not compete with each other
    (e.g. properties and classes): neither is a candidate or a rival for the
    other. Symmetric: the order of the two partitions does not matter."""
    model_config = ConfigDict(extra="forbid")
    partition_a: str
    partition_b: str
    reason: str = ""
    set_by: str = "partitions"  # the stage that set it (stage b1, or a stage of the pair)

    def partitions(self) -> frozenset[str]:
        return frozenset((self.partition_a, self.partition_b))


class GroupOrigin(StrEnum):
    PARTITION = "partition"  # copy of a partition (the start of stage b2)
    CLASS = "class"          # a large class (rdf:type) of a large KG
    FORMED = "formed"        # formed in stage b2


class GroupDefinition(BaseModel):
    """Which entities form a group: its name and membership. Also the form in
    which a stage defines a group inline (scope.Population)."""
    model_config = ConfigDict(extra="forbid")
    name: str
    membership_sparql: str  # SPARQL group pattern over ?entity
    origin: GroupOrigin = GroupOrigin.FORMED
    parent: str | None = None  # a group it is part of (for reading; membership is its own)
    features: str = ""         # what distinguishes its entities, to recognize them


class GroupProfile(BaseModel):
    """A group of the profile: its definition and what the stages decided about it."""
    model_config = ConfigDict(extra="forbid")
    definition: GroupDefinition
    # not aligned (e.g. image pages, helper nodes), set by stage b2; an entity in an
    # excluded group is not aligned even if it is in other groups
    excluded: bool = False
    exclusion_reason: str = ""
    local_name_type: NameType | None = None  # set by stage d (or its default)
    added: Roles = Roles()    # group-specific; a relation overrides the KG-wide one of its name
    removed: Roles = Roles()  # used in the group, but uninformative there

    @property
    def name(self) -> str:
        return self.definition.name


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Stage
    text: str
    group: str | None = None  # the group (or in stage b1 the partition) it is about


class Flag(BaseModel):
    """A later stage's objection to an earlier stage's result."""
    model_config = ConfigDict(extra="forbid")
    stage: Stage        # stage that raises the flag
    concerns: Stage     # stage whose result it questions
    text: str
    group: str | None = None


class KgProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kg_name: str
    # triples of the KG (None: not countable); a large KG is profiled on samples (large_kg_sampling)
    triples: int | None = None
    large: bool = False
    entity_space: list[str] = []
    partitions: list[Partition] = []  # in order: the first match wins
    non_competing_pairs: list[WithinKgNonCompetingPair] = []
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
        partition_names = self.partition_names()
        if partition_name not in partition_names:
            error_message = (
                f"{partition_name} is not an existing partition name "
                f"in this knowledge graph."
                )
            if len(partition_names) == 0:
                error_message += " No partitions defined in this knowledge graph yet."
            else:
                error_message += f" Choose between: {partition_names}"
            raise ValueError(error_message)
        try:
            return next(p for p in self.partitions if p.name == partition_name)
        except Exception as e:
            raise ValueError from e

    def get_group(self, group_name: str) -> GroupProfile:
        group_names = self.group_names()
        if group_name not in group_names:
            error_message = (
                f"{group_name} is not an existing group name "
                f"in this knowledge graph."
                )
            if len(group_names) == 0:
                error_message += " No groups defined in this knowledge graph yet."
            else:
                error_message += f" Choose between: {group_names}"
            raise ValueError(error_message)
        try:
            return next(g for g in self.groups if g.name == group_name)
        except Exception as e:
            raise ValueError from e

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

    def group_name_already_exists(self, name: str) -> bool:
        return name in self.group_names()

    def append_group(self, group: GroupDefinition | GroupProfile):
        match group:
            case GroupDefinition():
                group_profile = GroupProfile(definition=group)
            case GroupProfile():
                group_profile = group
            case _:
                raise TypeError(f"Expected GroupDefinition or GroupProfile, got {type(group)}")
        if self.group_name_already_exists(group_profile.name):
            raise ValueError(
                f"{group_profile.name} already exists in this knowledge graph."
                f" Please choose a new name. Already existing: {self.group_names()}"
                )
        self.groups.append(group_profile)

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

def short_iri(iri: str) -> str:
    """Vocabulary IRIs by their prefixed name (rdfs:label), others as they are."""
    for namespace, prefix in VOCABULARY_PREFIXES.items():
        if iri.startswith(namespace):
            return f"{prefix}:{iri[len(namespace):]}"
    return iri


def describe_pattern(pattern: PredicatePattern) -> str:
    return ", ".join([short_iri(predicate) for predicate in pattern.predicates]
                     + [f"pattern {{ {sparql} }}" for sparql in pattern.patterns])


def describe_relation(relation: StructureRelation) -> str:
    algebra = [property_name for property_name in ("transitive", "symmetric") if getattr(relation, property_name)]
    if relation.inverse_of:
        algebra.append(f"inverse of {relation.inverse_of}")
    return (f"{relation.name} ({relation.kind.value}{', ' + ', '.join(algebra) if algebra else ''}): "
            f"{describe_pattern(relation.predicate_pattern)}")


def describe_roles(roles: Roles) -> list[str]:
    lines = [f"{field.replace('_', ' ')}: {describe_pattern(pattern)}"
             for field in ROLE_FIELDS if (pattern := getattr(roles, field)) is not None]
    lines += [f"structure relation {describe_relation(relation)}" for relation in roles.structure]
    return lines


def describe_profile(profile: KgProfile, findings: bool = True, groups: Iterable[str] | None = None) -> list[str]:
    """The profile as short notes for prompts.
    Patterns read from ?s, the entity, to ?o; structure relations from the
    specific to the general entity. With groups, only the findings about the
    whole KG and about these groups (or partitions)."""
    notes = []
    if profile.entity_space:
        notes.append("Entities to align have IRIs starting with " + " or ".join(profile.entity_space) + ".")
    if profile.partitions:
        parts = [f"{partition.name}: {partition.features}" if partition.features else partition.name
                 for partition in profile.partitions]
        notes.append("Partitions (exclusive kinds of entities): " + "; ".join(parts) + ".")
    if profile.non_competing_pairs:
        notes.append("Do not compete with each other: "
                     + "; ".join(f"{pair.partition_a} and {pair.partition_b}" + (f" ({pair.reason})" if pair.reason else "")
                                 for pair in profile.non_competing_pairs)
                     + ".")
    if profile.groups:
        parts = []
        for group in profile.groups:
            definition = group.definition
            text = group.name
            if definition.parent:
                text += f" (part of {definition.parent})"
            if definition.features:
                text += f": {definition.features}"
            if group.excluded:
                text += " [excluded from the alignment" + (f": {group.exclusion_reason}" if group.exclusion_reason else "") + "]"
            parts.append(text)
        notes.append("Groups (may overlap): " + "; ".join(parts) + ".")
    role_lines = describe_roles(profile.roles)
    if role_lines:
        notes.append("Roles of predicates (?s entity, ?o value): " + "; ".join(role_lines) + ".")
    for group in profile.groups:
        details = []
        if group.local_name_type is not None:
            details.append(f"local names of IRIs are {group.local_name_type.value}s"
                           if group.local_name_type != NameType.OTHER else "local names of IRIs are mixed")
        added, removed = describe_roles(group.added), describe_roles(group.removed)
        if added:
            details.append("in addition " + "; ".join(added))
        if removed:
            details.append("not informative here: " + "; ".join(removed))
        if details:
            notes.append(f"Group {group.name}: " + "; ".join(details) + ".")
    if findings:
        shown = None if groups is None else set(groups)
        for finding in profile.findings:
            if shown is not None and finding.group is not None and finding.group not in shown:
                continue
            scope = f" ({finding.group})" if finding.group else ""
            notes.append(f"Observation{scope}: {finding.text}")
    return notes
