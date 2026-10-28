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
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from grasp.tasks.om.model import IRI, VOCABULARY_PREFIXES, NameType, PredicatePattern, StructureRelation


class Stage(StrEnum):
    SPACE = "space"
    PARTITIONS = "partitions"
    ROLES = "roles"
    PARTITION = "partition"


STAGES = list(Stage)  # in pipeline order


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
    return ", ".join([short_iri(p) for p in pattern.predicates] + [f"pattern {{ {p} }}" for p in pattern.patterns])


def describe_relation(relation: StructureRelation) -> str:
    algebra = [a for a in ("transitive", "symmetric") if getattr(relation, a)]
    if relation.inverse_of:
        algebra.append(f"inverse of {relation.inverse_of}")
    return (f"{relation.name} ({relation.kind.value}{', ' + ', '.join(algebra) if algebra else ''}): "
            f"{describe_pattern(relation.predicate_pattern)}")


def describe_roles(roles: Roles) -> list[str]:
    lines = [f"{field.replace('_', ' ')}: {describe_pattern(pattern)}"
             for field in ROLE_FIELDS if (pattern := getattr(roles, field)) is not None]
    lines += [f"structure relation {describe_relation(r)}" for r in roles.structure]
    return lines


def describe_profile(profile: KgProfile, findings: bool = True, partitions: Iterable[str] | None = None) -> list[str]:
    """The profile as short notes for prompts (in place of structural notes).
    Patterns read from ?s, the entity, to ?o; structure relations from the
    specific to the general entity. With partitions, only the findings about
    the whole KG and about these partitions."""
    notes = []
    if profile.entity_space:
        notes.append("Entities to align have IRIs starting with " + " or ".join(profile.entity_space) + ".")
    if profile.partitions:
        parts = []
        for name, partition in profile.partitions.items():
            definition = partition.definition
            text = name
            if definition.origin == PartitionOrigin.SPLIT:
                text += f" (part of {definition.parent})"
            if definition.features:
                text += f": {definition.features}"
            if definition.excluded:
                text += " [excluded from the alignment]"
            parts.append(text)
        notes.append("Partitions: " + "; ".join(parts) + ".")
    role_lines = describe_roles(profile.roles)
    if role_lines:
        notes.append("Roles of predicates (?s entity, ?o value): " + "; ".join(role_lines) + ".")
    for name, partition in profile.partitions.items():
        details = []
        if partition.local_name_type is not None:
            details.append(f"local names of IRIs are {partition.local_name_type.value}s"
                           if partition.local_name_type != NameType.OTHER else "local names of IRIs are mixed")
        added, removed = describe_roles(partition.added), describe_roles(partition.removed)
        if added:
            details.append("in addition " + "; ".join(added))
        if removed:
            details.append("not informative here: " + "; ".join(removed))
        if details:
            notes.append(f"Partition {name}: " + "; ".join(details) + ".")
    if findings:
        shown = None if partitions is None else set(partitions)
        for finding in profile.findings:
            if shown is not None and finding.partition is not None and finding.partition not in shown:
                continue
            scope = f" ({finding.partition})" if finding.partition else ""
            notes.append(f"Observation{scope}: {finding.text}")
    return notes
