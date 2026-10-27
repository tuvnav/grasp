"""
Data model of the staged KG profile.

Stages, each a GRASP task on one KG; each reads the profile of the stages
before and adds to it:
    a. space       entity space (namespaces of the entities to align)
    b. partitions  check the default partitions by kind; split, redefine, exclude
    c. roles       KG-wide roles: names, synonyms, ..., structure relations
    d. partition   one small task per partition: local name type and the
                   partition's deviations from the KG-wide roles

A partition stores only its deviations (added, removed); its effective roles
are derived: KG-wide roles used in the partition, minus removed, plus added.
The partition task shows the LLM the effective roles and computes the delta
when it finishes (diff_roles), so the LLM never edits a delta itself.

Findings are short notes of a stage for the stages after it and for later
prompts (they replace the structural notes when these are missing). Later
stages read them as unverified observations and report a contradiction as a
flag instead of changing an earlier stage's result.
"""
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from grasp.tasks.om.model import IRI, NameType, PredicatePattern, StructureRelation


class Stage(StrEnum):
    SPACE = "space"
    PARTITIONS = "partitions"
    ROLES = "roles"
    PARTITION = "partition"


class Roles(BaseModel):
    """Roles of predicates and patterns, KG-wide or as a partition's delta.
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


ROLE_FIELDS = ("names", "synonyms", "definitions", "codes", "cross_references", "noise")


class PartitionOrigin(StrEnum):
    DEFAULT = "default"      # a kind of the default partitioning (class, property, concept, instance)
    SPLIT = "split"          # a group carved out of another partition
    REDEFINED = "redefined"  # a partition with its own membership instead of a kind


class PartitionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # SPARQL pattern over ?entity; None for a default partition (membership by
    # KIND_PARTITION_SPARQL). Partitions are checked in order, the first match wins,
    # so a split comes before the partition it is carved from.
    membership: str | None = None
    origin: PartitionOrigin = PartitionOrigin.DEFAULT
    parent: str | None = None  # the partition a split is carved from
    features: str = ""         # what distinguishes its entities from the rest
    excluded: bool = False     # not aligned (e.g. image pages, helper nodes)


class PartitionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition: PartitionDefinition = PartitionDefinition()
    local_name_type: NameType | None = None  # set by stage d (or its default)
    added: Roles = Roles()    # partition-specific; a relation overrides the KG-wide one of its name
    removed: Roles = Roles()  # used in the partition, but uninformative there


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Stage
    text: str
    partition: str | None = None


class Flag(BaseModel):
    """A later stage's objection to an earlier stage's result."""
    model_config = ConfigDict(extra="forbid")
    stage: Stage        # stage that raises the flag
    concerns: Stage     # stage whose result it questions
    text: str
    partition: str | None = None


class KgProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kg_name: str
    entity_space: list[str] = []
    roles: Roles = Roles()
    partitions: dict[str, PartitionProfile] = {}
    findings: list[Finding] = []
    flags: list[Flag] = []


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


def same_element(a: str | StructureRelation, b: str | StructureRelation) -> bool:
    # a relation edited without change stays the same, whoever set it
    if isinstance(a, StructureRelation) and isinstance(b, StructureRelation):
        return a.model_dump(exclude={"source"}) == b.model_dump(exclude={"source"})
    return a == b


# number of entities of a partition that an element covers (as ?s)
type Usage = dict[ElementKey, int]


def derived_roles(kg_roles: Roles, usage: Usage) -> Roles:
    """KG-wide roles restricted to the elements used in a partition."""
    return roles_from_elements({
        key: value for key, value in role_elements(kg_roles).items() if usage.get(key, 0) > 0
    })


def effective_roles(kg_roles: Roles, partition: PartitionProfile, usage: Usage) -> Roles:
    """Roles of a partition: KG-wide roles used in it, minus removed, plus added."""
    elements = role_elements(derived_roles(kg_roles, usage))
    for key in role_elements(partition.removed):
        elements.pop(key, None)
    elements.update(role_elements(partition.added))
    return roles_from_elements(elements)


def diff_roles(derived: Roles, final: Roles) -> tuple[Roles, Roles]:
    """(added, removed) that turn derived into final. A relation changed
    under its name is added (overriding), not removed and added."""
    before, after = role_elements(derived), role_elements(final)
    added = {k: v for k, v in after.items() if k not in before or not same_element(before[k], v)}
    removed = {k: v for k, v in before.items() if k not in after}
    return roles_from_elements(added), roles_from_elements(removed)
