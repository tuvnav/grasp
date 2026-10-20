from dataclasses import dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Any, Callable
from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.manager.utils import merge_prefixes
from grasp.sparql.types import Binding
from grasp.tasks.entities import Entity
from grasp.utils import derive_label_from_iri
from pydantic import BaseModel, model_validator
from rdflib import Literal
import yaml


@dataclass
class FunctionRegistry:
    Registry: dict[StrEnum, Callable] = field(default_factory=dict)

    def register(self, function_name: StrEnum):
        def decorator(func: Callable):
            self.Registry[function_name] = func
            return func
        return decorator


PREFIX_RE = re.compile(r"^\s*PREFIX\s+([^:\s]*):\s*<([^>]*)>\s*$", re.IGNORECASE)


def split_prologue(manager: KgManager, sparql: str) -> tuple[dict[str, str], str]:
    """Normalizes the prefixes of a query and splits it into (prefixes, body)."""
    lines = manager.fix_prefixes(sparql).splitlines()
    prefixes = {}
    while lines and (m := PREFIX_RE.match(lines[0])):
        prefixes[m.group(1)] = m.group(2)
        lines.pop(0)
    return prefixes, "\n".join(lines)


@dataclass(frozen=True)
class QueryPart:
    """A self-contained SELECT query as part of a composed query. Optional parts
    are left-joined: rows without a match keep their other variables."""
    sparql: str
    optional: bool = False


def compose(manager: KgManager, parts: list[str | QueryPart], projection: list[str]) -> str:
    """Joins self-contained SELECT queries on their shared projected variables. Also moves all PREFIX-declarations of the individual
    queries to a single PREFIX-block at the top of the aggregate query."""
    prefixes, bodies = {}, []
    for part in parts:
        if isinstance(part, str):
            part = QueryPart(part)
        part_prefixes, body = split_prologue(manager, part.sparql)
        prefixes, _, _ = merge_prefixes(prefixes, part_prefixes, do_raise=True)
        bodies.append(("OPTIONAL {\n" if part.optional else "{\n") + body + "\n}")
    prologue = "\n".join(f"PREFIX {s}: <{l}>" for s, l in prefixes.items())
    return (f"{prologue}\nSELECT DISTINCT {' '.join(projection)} WHERE {{\n"
            + "\n".join(bodies) + "\n}")


# Key contract: ?entity ?key, optional ?role = "secondary" (key only blocks
# ambiguous anchors; default primary). Labels are primary, synonyms secondary;
# entities without a label get the local name of their IRI as primary key.
NAME_ANCHORS_SPARQL = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT ?entity ?key ?role WHERE {
  {
    SELECT DISTINCT ?entity WHERE {
      ?entity ?p ?o .
      FILTER(isIRI(?entity))
      FILTER(!STRSTARTS(STR(?entity), STR(rdf:)) && !STRSTARTS(STR(?entity), STR(rdfs:))
          && !STRSTARTS(STR(?entity), STR(owl:)) && !STRSTARTS(STR(?entity), STR(skos:)))
    }
  }
  {
    # labels: primary keys (?role unbound)
    ?entity rdfs:label|skos:prefLabel ?name .
  } UNION {
    # synonyms: secondary keys, only block ambiguous anchors
    ?entity skos:altLabel|oboInOwl:hasExactSynonym|oboInOwl:hasRelatedSynonym|oboInOwl:hasSynonym ?name .
    FILTER(isLiteral(?name))
    BIND("secondary" AS ?role)
  } UNION {
    # synonym as a resource with its own label
    ?entity skos:altLabel|oboInOwl:hasExactSynonym|oboInOwl:hasRelatedSynonym|oboInOwl:hasSynonym ?synonym .
    ?synonym rdfs:label ?name .
    BIND("secondary" AS ?role)
  } UNION {
    # fallback for entities without a label (?name unbound)
    BIND(true AS ?fallback)
  }
  FILTER(!BOUND(?fallback) || NOT EXISTS { ?entity rdfs:label|skos:prefLabel ?label })
  # outside the UNION, where ?entity is bound: local name of the IRI as primary key
  BIND(COALESCE(?name, REPLACE(STR(?entity), "^.*[/#]", "")) AS ?key)
}"""


# Scope contract: ?entity. Restricts the entities of the key query; no scope
# query means all entities of the key query.

# Partition contract: ?entity ?partition. Joined optionally, entities without a
# partition get UNCATEGORIZED. An entity may be in several partitions (e.g.
# punning). ?evidence is for inspection only.
UNCATEGORIZED = "uncategorized"
# partition of all entities when a side has no partition query
ALL = "all"

KIND_PARTITION_SPARQL = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?entity ?partition ?evidence WHERE {
  {
    ?entity rdf:type ?type .
    VALUES (?type ?partition) {
      (owl:Class "class") (rdfs:Class "class")
      (owl:ObjectProperty "property") (owl:DatatypeProperty "property")
      (owl:AnnotationProperty "property") (rdf:Property "property")
      (owl:NamedIndividual "instance") (skos:Concept "concept")
    }
    BIND("declared" AS ?evidence)
  } UNION {
    ?x rdf:type ?entity .
    BIND("class" AS ?partition) BIND("used" AS ?evidence)
  } UNION {
    { ?entity rdfs:subClassOf ?y } UNION { ?y rdfs:subClassOf ?entity }
    BIND("class" AS ?partition) BIND("used" AS ?evidence)
  } UNION {
    ?s ?entity ?o .
    BIND("property" AS ?partition) BIND("used" AS ?evidence)
  } UNION {
    ?entity rdf:type ?t .
    FILTER(isIRI(?t) && !STRSTARTS(STR(?t), STR(owl:)) && !STRSTARTS(STR(?t), STR(rdfs:))
           && !STRSTARTS(STR(?t), STR(rdf:)) && !STRSTARTS(STR(?t), STR(skos:)))
    BIND("instance" AS ?partition) BIND("used" AS ?evidence)
  }
  FILTER(isIRI(?entity))
  FILTER(!STRSTARTS(STR(?entity), STR(rdf:)) && !STRSTARTS(STR(?entity), STR(rdfs:))
      && !STRSTARTS(STR(?entity), STR(owl:)) && !STRSTARTS(STR(?entity), STR(skos:)))
}"""


# class KeyRole(StrEnum):
#     LOCAL = "local"
#     LABEL = "label"
#     SYNONYM = "synonym"
#     UNKNOWN = "unknown"

#     @classmethod
#     def _missing_(cls, value: Any) -> "KeyRole":
#         return cls.UNKNOWN

class AnchorStrategyType(StrEnum):
    KEY_ANCHORS = "key_anchors"
    NO_ANCHORS = "no_anchors"


class CandidateStrategyType(StrEnum):
    EMBEDDING_CANDIDATES = "embedding_candidates"


# For raw KG retrieval results
@dataclass(frozen=True)
class KeyEntityRow:
    entity: str
    key: str
    partition: str
    primary: bool

    @staticmethod
    def from_raw_row(row: dict, prefixes: dict[str, str]) -> "KeyEntityRow":
        entity = Binding.from_dict(row["entity"])
        key = Binding.from_dict(row["key"])
        # unbound optional variables are missing from the row
        partition = Binding.from_dict(row["partition"]).value if "partition" in row else UNCATEGORIZED
        role = Binding.from_dict(row["role"]).value if "role" in row else None
        primary = role != "secondary"

        return KeyEntityRow(entity.value, key.value, partition, primary)


@dataclass(frozen=True)
class PartitionGroup:
    """Source partitions compared with target partitions in one run of the anchor
    rule; uniqueness holds within the union of the partitions of each side."""
    source: frozenset[str]
    target: frozenset[str]

    @property
    def label(self) -> str:
        return f"{'+'.join(sorted(self.source))} <-> {'+'.join(sorted(self.target))}"


@dataclass(frozen=True)
class KeyAnchor:
    source_iri: str
    target_iri: str
    group: PartitionGroup
    string_normalization_level: str  # finest normalization level on which the pair is unique
    evidence: str  # shared normalized key on that level
    # original (unnormalized) primary keys behind the shared key, for display
    source_key: str = ""
    target_key: str = ""
    head: bool = False  # key is extended by another name in one of the ontologies
    information_content: float = 1.0  # lexical information content, min over both sides


@dataclass
class NormalizationLevelStats:
    new_anchors: int = 0
    # entities whose key first collides with another entity within the same ontology on this level
    new_collisions_source: int = 0
    new_collisions_target: int = 0
    # entities dropped because they have unique keys towards several partners
    ambiguous_source: int = 0
    ambiguous_target: int = 0


@dataclass
class GroupStats:
    n_source: int
    n_target: int
    anchors: int = 0
    specific: int = 0
    head: int = 0
    extended: int = 0
    anchor_rate: float = 0.0
    token_overlap: float = 0.0
    levels: dict[str, NormalizationLevelStats] = field(default_factory=dict)


@dataclass
class KeyAnchorResult:
    anchors: list[KeyAnchor]
    group_statistics: dict[PartitionGroup, GroupStats]
    # partitions that occur in the data but in no group
    unpaired_source: list[str]
    unpaired_target: list[str]
    # partitions named in the pairing that do not occur in the data
    missing_source: list[str]
    missing_target: list[str]


class SideQueries(BaseModel):
    """Queries of one side, joined on ?entity (see the contracts above)."""
    key: str = NAME_ANCHORS_SPARQL
    scope: str | None = None
    partition: str | None = KIND_PARTITION_SPARQL


class PairingGroup(BaseModel):
    """Source partitions compared with target partitions, e.g. OWL classes with
    SKOS concepts: source ["class"], target ["class", "concept"]."""
    source: list[str]
    target: list[str]


class KeyAnchorParams(BaseModel):
    source: SideQueries = SideQueries()
    target: SideQueries = SideQueries()
    # also run the extended normalization levels (word order, plural, inflection)
    extended: bool = False
    # None = each partition with the partition of the same name
    pairing: list[PairingGroup] | None = None

    @model_validator(mode="before")
    @classmethod
    def target_defaults_to_source(cls, data: Any) -> Any:
        # the target side starts as a copy of the source side, fields given for
        # the target (also explicit nulls) override it; runs on the raw input,
        # the only place where omitted and default values differ
        if not isinstance(data, dict) or "source" not in data:
            return data
        source = data["source"]
        if isinstance(source, BaseModel):
            source = source.model_dump(exclude_unset=True)
        target = data.get("target") or {}
        if isinstance(target, BaseModel):
            target = target.model_dump(exclude_unset=True)
        return {**data, "target": {**source, **target}}


type AnchorStrategyParams = KeyAnchorParams  # add other param classes


class AnchorStrategyConfig(BaseModel):
    type: AnchorStrategyType
    params: AnchorStrategyParams


@dataclass
class AnchorStats:
    n_source: int
    n_target: int
    per_stratum: dict[str, int]
    details: dict[str, int | float]


@dataclass
class AnchorResult:
    config: AnchorStrategyConfig
    stats: AnchorStats
    anchors: list[KeyAnchor]


def format_entity(entity: Entity) -> str:
    output = f"Full IRI: {entity.identifier}"
    if entity.entity != entity.identifier:
        output += f", shortened IRI: {entity.entity}"
    if entity.label is not None:
        output += f", label: {entity.label}"
    if entity.aliases and entity.aliases != [entity.label]:
        output += f", aliases: {entity.aliases}"
    if entity.infos is not None and entity.infos != []:
        output += f", infos: {entity.infos}"
    return output


class RelationSpec(BaseModel):
    """
    A relation that may hold between a source and a target entity, together
    with the mathematical properties the system uses for consistency checks.
    `iri` is the relation's identifier; for OAEI relations the Alignment API
    symbol (e.g. "=") is used so that exported alignments stay unchanged.
    `description` is shown to the LLM and should state the functional
    criterion of the relation, read from the source to the target entity:
    what a query for one entity should return in the other system.
    """
    iri: str
    description: str
    symmetric: bool = False
    transitive: bool = False
    inverse_of: str | None = None
    # relations that cannot hold for the same entity pair simultaneously
    excludes: list[str] = []
    # weaker relations that hold for an entity pair whenever this one holds
    # (e.g. skos:exactMatch implies skos:closeMatch)
    implies: list[str] = []

    @model_validator(mode="after")
    def check_symmetry(self) -> "RelationSpec":
        # a relation is symmetric iff it is its own inverse
        if self.symmetric:
            if self.inverse_of is None:
                self.inverse_of = self.iri
            elif self.inverse_of != self.iri:
                raise ValueError(
                    f"Relation {self.iri} is symmetric, so it must be its "
                    f"own inverse, but inverse_of is {self.inverse_of}"
                )
        elif self.inverse_of == self.iri:
            self.symmetric = True
        return self


class RelationProfile(BaseModel):
    """
    The set of relations the LLM may choose from within one alignment task.
    """
    name: str
    relations: list[RelationSpec]
    # alignment-level cardinality in Alignment API notation (11, 1?, ??, **);
    # informational for now, not enforced
    cardinality: str = "**"
    # whether an ontology may contain duplicate entities, so that several
    # entities related by a symmetric and transitive relation to the same
    # entity of the other ontology can all be correct
    duplicates_within: bool = False
    # optional explanation of how to decide between the relations, shown to
    # the LLM right after the relation descriptions (e.g. an order in which
    # the relations are to be checked)
    description: str | None = None

    @model_validator(mode="after")
    def check_references(self) -> "RelationProfile":
        iris = [r.iri for r in self.relations]
        if not iris:
            raise ValueError(f"Relation profile {self.name} has no relations")
        if len(iris) != len(set(iris)):
            raise ValueError(f"Relation profile {self.name} has duplicate relations")
        for r in self.relations:
            for ref in [r.inverse_of, *r.excludes, *r.implies]:
                if ref is not None and ref not in iris:
                    raise ValueError(
                        f"Relation {r.iri} in profile {self.name} references "
                        f"unknown relation {ref}"
                    )
        return self

    def get(self, iri: str) -> RelationSpec | None:
        return next((r for r in self.relations if r.iri == iri), None)


EQUIVALENCE = RelationSpec(
    iri="=",
    description="The source and the target entity refer to the same real-world things: "
    "a query for either entity should also return the entries of the other.",
    symmetric=True,
    transitive=True,
)

# relation types of the STROMA/TaSeR datasets, as used in the OAEI Beyond
# Equivalence track (MELT: TrackRepository.GeneralRelation)
STROMA_RELATIONS = [
    EQUIVALENCE.model_copy(update={"excludes": ["<", ">"]}),
    RelationSpec(
        iri="<",
        description="The source entity is narrower than the target entity: a query for the "
        "target entity should also return all entries of the source entity, but not vice "
        "versa (e.g. 'Jeans' < 'Clothing').",
        transitive=True,
        inverse_of=">",
        excludes=["=", ">"],
    ),
    RelationSpec(
        iri=">",
        description="The source entity is broader than the target entity: a query for the "
        "source entity should also return all entries of the target entity, but not vice "
        "versa (e.g. 'Clothing' > 'Jeans').",
        transitive=True,
        inverse_of="<",
        excludes=["=", "<"],
    ),
    RelationSpec(
        iri="HasA",
        description="Entries of the target entity are parts or components of entries of the "
        "source entity (e.g. 'Suit' HasA 'Trousers').",
        inverse_of="PartOf",
    ),
    RelationSpec(
        iri="PartOf",
        description="Entries of the source entity are parts or components of entries of the "
        "target entity (e.g. 'Trousers' PartOf 'Suit').",
        inverse_of="HasA",
    ),
    RelationSpec(
        iri="Related",
        description="The entities are closely associated, so a system using one of them "
        "should link to the other, but none of the other relations holds.",
        symmetric=True,
    ),
]

RELATION_PROFILES: dict[str, RelationProfile] = {
    "equivalence": RelationProfile(
        name="equivalence", relations=[EQUIVALENCE], cardinality="11"
    ),
    "stroma": RelationProfile(
        name="stroma", relations=STROMA_RELATIONS, cardinality="**"
    ),
}


def allows_many(profile: RelationProfile) -> bool:
    """
    Whether entities may take part in several correspondences, i.e. the
    profile's cardinality (Alignment API notation) contains '*' or '+'.
    """
    return any(c in profile.cardinality for c in "*+")


def load_relation_profile(spec: str | dict) -> RelationProfile:
    """
    `spec` is the name of a built-in profile, the path to a YAML or JSON
    file with a profile, or a profile given inline as a mapping.
    """
    if isinstance(spec, dict):
        return RelationProfile(**spec)
    if spec in RELATION_PROFILES:
        return RELATION_PROFILES[spec]
    path = Path(spec).expanduser()
    if path.suffix not in (".yaml", ".yml", ".json"):
        raise ValueError(
            f"Unknown relation profile {spec}, must be one of "
            f"{list(RELATION_PROFILES)} or a .yaml/.yml/.json file"
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    return RelationProfile(**data)


def get_relation_profile(config: GraspConfig) -> RelationProfile:
    return load_relation_profile(
        config.task_kwargs.get("om", {}).get("relation_profile", "equivalence")
    )


class CorrespondenceSide(StrEnum):
    SOURCE = "source"
    TARGET = "target"


class Correspondence(BaseModel):
    # TODO: Enxtendable to handle complex patterns
    # besides atomic components?
    """
    A singe correspondence between two entities
    from different ontologies.
    """
    entity_source: Entity
    entity_target: Entity

    # OAEI standard
    measure: float = 1.0
    # iri of a RelationSpec from the task's relation profile
    relation: str = EQUIVALENCE.iri

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.entity_source.identifier, self.entity_target.identifier, self.relation)


class PotentialCorrespondences(BaseModel):
    source_entity: Entity
    candidates: list[Entity]
