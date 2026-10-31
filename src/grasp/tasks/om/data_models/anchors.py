"""
Data model of the anchor stage: key rows, key anchors and their statistics,
the queries of a side (key, scope, partition) with their defaults, and the
stage result.
"""
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from grasp.sparql.types import Binding
from grasp.tasks.om.foundation.types import IRI
from grasp.tasks.om.foundation.vocabulary import (
    LABEL_PREDICATES,
    SYNONYM_PREDICATES,
    UNCATEGORIZED,
)
from grasp.tasks.om.sparql_access.query_building import sparql_path

# Key contract: ?entity ?key, optional ?role = "secondary" (key only blocks
# ambiguous anchors; default primary). Labels are primary, synonyms secondary;
# entities without a label get the local name of their IRI as primary key.
_NAME_ANCHORS_TEMPLATE = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
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
    ?entity __LABELS__ ?name .
  } UNION {
    # synonyms: secondary keys, only block ambiguous anchors
    ?entity __SYNONYMS__ ?name .
    FILTER(isLiteral(?name))
    BIND("secondary" AS ?role)
  } UNION {
    # synonym as a resource with its own label
    ?entity __SYNONYMS__ ?synonym .
    ?synonym __LABELS__ ?name .
    BIND("secondary" AS ?role)
  } UNION {
    # fallback for entities without a label (?name unbound)
    BIND(true AS ?fallback)
  }
  FILTER(!BOUND(?fallback) || NOT EXISTS { ?entity __LABELS__ ?label })
  # outside the UNION, where ?entity is bound: local name of the IRI as primary key
  BIND(COALESCE(?name, REPLACE(STR(?entity), "^.*[/#]", "")) AS ?key)
}"""


NAME_ANCHORS_SPARQL = (
    _NAME_ANCHORS_TEMPLATE
    .replace("__LABELS__", sparql_path(LABEL_PREDICATES))
    .replace("__SYNONYMS__", sparql_path(SYNONYM_PREDICATES))
)


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
  } UNION {
    # untyped individuals (e.g. wiki articles): subjects with data beyond names and
    # without a schema role; helper nodes that only carry a label stay uncategorized
    { SELECT DISTINCT ?entity WHERE {
        ?entity ?p ?o .
        FILTER(?p NOT IN (rdf:type, rdfs:label, rdfs:comment, skos:prefLabel, skos:altLabel))
    } }
    FILTER NOT EXISTS { ?entity rdf:type ?anyType }
    FILTER NOT EXISTS { ?anyInstance rdf:type ?entity }
    FILTER NOT EXISTS { ?entity rdfs:subClassOf ?anySuper }
    FILTER NOT EXISTS { ?anySub rdfs:subClassOf ?entity }
    FILTER NOT EXISTS { ?anySubject ?entity ?anyObject }
    BIND("instance" AS ?partition) BIND("untyped" AS ?evidence)
  }
  FILTER(isIRI(?entity))
  FILTER(!STRSTARTS(STR(?entity), STR(rdf:)) && !STRSTARTS(STR(?entity), STR(rdfs:))
      && !STRSTARTS(STR(?entity), STR(owl:)) && !STRSTARTS(STR(?entity), STR(skos:)))
}"""


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


class PartitionGroup(BaseModel):
    """Source partitions compared with target partitions in one run of the anchor
    rule; uniqueness holds within the union of the partitions of each side."""
    model_config = ConfigDict(frozen=True)
    source: frozenset[str]
    target: frozenset[str]

    @property
    def label(self) -> str:
        return f"{'+'.join(sorted(self.source))} <-> {'+'.join(sorted(self.target))}"


class KeyAnchor(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_iri: IRI
    target_iri: IRI
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
class TryKeyAnchorResult:
    anchors: list[KeyAnchor]
    group_statistics: dict[PartitionGroup, GroupStats]
    # partitions that occur in the data but in no group
    unpaired_source: list[str]
    unpaired_target: list[str]
    # partitions named in the pairing that do not occur in the data
    missing_source: list[str]
    missing_target: list[str]
    # entities of each side that took part in a compared partition group
    source_entities: frozenset[str] = frozenset()
    target_entities: frozenset[str] = frozenset()


class SideQueries(BaseModel):
    """Queries of one side, joined on ?entity (see the contracts above)."""
    # unknown fields are errors, e.g. typos in LLM arguments
    model_config = ConfigDict(extra="forbid")
    key: str = NAME_ANCHORS_SPARQL
    scope: str | None = None
    partition: str | None = KIND_PARTITION_SPARQL


class PairingGroup(BaseModel):
    """Source partitions compared with target partitions, e.g. OWL classes with
    SKOS concepts: source ["class"], target ["class", "concept"]."""
    model_config = ConfigDict(extra="forbid")
    source: list[str]
    target: list[str]


class KeyAnchorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
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


class SelectedRun(BaseModel):
    run_id: str
    groups: list[str]
    params: KeyAnchorParams
    statistics: dict[str, dict[str, int | float]]   # Group -> stats


class AnchorRecord(KeyAnchor):
    stratum: str
    run_id: str


class AnchorConflict(BaseModel):
    source_iri: IRI
    target_iri: IRI
    runs: list[str]


class AnchorStageResult(BaseModel):
    source_kg: str
    target_kg: str
    entity_space: dict[str, list[str]]
    finished: bool
    fallback: str | None
    rationale: str | None
    runs_tried: int
    selected: list[SelectedRun]
    strata: dict[str, int]
    anchors: list[AnchorRecord]
    conflicts: list[AnchorConflict]
