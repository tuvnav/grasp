from dataclasses import dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Any, Callable, NewType
from urllib.parse import urlsplit
from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.manager.utils import merge_prefixes
from grasp.sparql.types import AskResult, Binding
from grasp.sparql.utils import prepare_identifier_for_sparql
from grasp.tasks.base import GraspTask
from grasp.tasks.entities import Entity
from grasp.utils import FunctionCallException, derive_label_from_iri
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdflib import Literal
import yaml


PROMPT_DIR = Path(__file__).parent / "prompts"
SIDES = ("source", "target")

# an IRI as such, e.g. http://human.owl#NCI_C12432
IRI = NewType("IRI", str)
# an IRI or literal prepared for insertion into SPARQL text, e.g. <http://human.owl#NCI_C12432>
SparqlTerm = NewType("SparqlTerm", str)


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def load_task_instructions(om_kwargs: dict) -> str | None:
    """
    Free-text instructions specific to the alignment task at hand (e.g. domain
    knowledge needed to decide on the relations), given inline or as the path
    to a .md/.txt file in the `task_instructions` om task kwargs.
    """
    spec = om_kwargs.get("task_instructions")
    if not spec:
        return None
    if "\n" not in spec and Path(spec).suffix in (".md", ".txt"):
        spec = Path(spec).expanduser().read_text(encoding="utf-8")
    return spec.strip() or None


class BindingType(StrEnum):
    """Values of Binding.type (SPARQL JSON results); equal to the plain strings
    GRASP uses, so it works with GRASP's unchanged Binding objects."""
    URI = "uri"
    LITERAL = "literal"
    BNODE = "bnode"


# default predicate canon, shared by the anchor key query and the entity context
RDF_TYPE = IRI("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
LABEL_PREDICATES = [
    IRI("http://www.w3.org/2000/01/rdf-schema#label"),
    IRI("http://www.w3.org/2004/02/skos/core#prefLabel"),
]
SYNONYM_PREDICATES = [
    IRI("http://www.w3.org/2004/02/skos/core#altLabel"),
    IRI("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
    IRI("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"),
    IRI("http://www.geneontology.org/formats/oboInOwl#hasSynonym"),
]
TYPE_PREDICATES = [RDF_TYPE]


# vocabulary namespaces, shown by their prefixed name (owl:Class) instead of a label
VOCABULARY_PREFIXES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://www.w3.org/2004/02/skos/core#": "skos",
    "http://www.w3.org/2001/XMLSchema#": "xsd",
}
VOCABULARY = tuple(VOCABULARY_PREFIXES)
RDF_FIRST = IRI("http://www.w3.org/1999/02/22-rdf-syntax-ns#first")
RDF_REST = IRI("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest")
OWL_ON_PROPERTY = IRI("http://www.w3.org/2002/07/owl#onProperty")

# blank node -> its (predicate, object) edges
type BlankNodeTree = dict[str, list[tuple[IRI, Binding]]]


def local_name(iri: str) -> str:
    """The part of an IRI after the last # or /."""
    return re.split(r"[#/]", iri.rstrip("/#"))[-1]


def namespace_of(iri: str) -> str:
    """Scheme and host of an IRI (e.g. http://mouse.owl), or the part up to the
    last # or / for IRIs without a host (e.g. URNs). Host rather than namespace,
    because some ontologies use a path per entity."""
    parts = urlsplit(iri)
    if parts.netloc:
        return f"{parts.scheme}://{parts.netloc}"
    cut = max(iri.rfind("#"), iri.rfind("/"))
    return iri[:cut + 1] if cut >= 0 else iri


def sparql_path(predicates: list[IRI]) -> str:
    """Alternative property path over the given predicates, e.g. <a>|<b>."""
    return "|".join(f"<{p}>" for p in predicates)


class EntityContextConfig(BaseModel):
    """How the context of an entity is built (entity_context.py); not to be
    confused with the list-based EntityContextConfig of agentic_search_verify."""
    model_config = ConfigDict(extra="forbid")
    name_predicates: list[IRI] = Field(default_factory=lambda: list(LABEL_PREDICATES))
    synonym_predicates: list[IRI] = Field(default_factory=lambda: list(SYNONYM_PREDICATES))
    type_predicates: list[IRI] = Field(default_factory=lambda: list(TYPE_PREDICATES))
    k_values: int = 5
    max_chars_per_literal: int = 300
    bn_resolution_max_depth: int = 6
    max_incoming_predicates: int = 8
    max_chars: int = 2000


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


class VerdictOption(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNSURE = "unsure"


class Verdict(AnchorRecord):
    verdict: VerdictOption
    rationale: str


class VerifyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_kg: str
    target_kg: str
    anchor: AnchorRecord


class JudgeMode(StrEnum):
    ONE_SHOT = "one_shot"  # A1: no KG functions, decision from the given context
    AGENTIC = "agentic"  # A2: KG functions within a small step budget
    ESCALATING = "escalating"  # A3: one-shot, agentic only if not accepted


class JudgeVariant(BaseModel):
    """How an anchor is judged (A1-A3, each optionally with competitors = A4)."""
    model_config = ConfigDict(extra="forbid")
    mode: JudgeMode = JudgeMode.ONE_SHOT
    competitors: bool = False
    # step budget of the agentic mode
    max_steps: int = 8

    @property
    def name(self) -> str:
        return self.mode.value + ("+competitors" if self.competitors else "")


class Judgment(Verdict):
    """A verdict with how it came about, one per judged anchor."""
    variant: str
    model: str
    steps: int  # LLM responses, summed over an escalation
    elapsed: float  # seconds, summed over an escalation
    error: str | None = None  # the run failed; the verdict is then unsure
    # escalating mode: verdict of the one-shot run if the agentic run followed
    first_verdict: VerdictOption | None = None


class CertificationParams(BaseModel):
    """Two-point sampling plan per stratum (hypergeometric): the smallest c, and
    for it the smallest n, with P(<= c rejects | p_max) <= alpha and
    P(<= c rejects | p_good) >= 1 - beta. Provisional values, to be revisited."""
    model_config = ConfigDict(extra="forbid")
    p_good: float = 0.01
    p_max: float = 0.05
    alpha: float = 0.05
    beta: float = 0.10
    # how unsure verdicts count: as rejects (conservative) or not at all (the
    # judged anchor is then replaced by the next one of the sample order)
    unsure_as_reject: bool = True
    seed: int = 0


class StratumDecisionType(StrEnum):
    ACCEPTED = "accepted"  # sample passed: all anchors of the stratum are certified
    REJECTED = "rejected"  # more than c rejects: only accepted judged anchors are kept
    INSPECTED = "inspected"  # no plan with n < N: every anchor judged individually


class StratumDecision(BaseModel):
    stratum: str
    size: int
    plan: tuple[int, int] | None  # (n, c); None for complete inspection
    decision: StratumDecisionType
    judged: int
    rejects: int  # including unsure verdicts if they count as rejects
    unsure: int


class CertificationResult(BaseModel):
    source_kg: str
    target_kg: str
    variant: str
    params: CertificationParams
    strata: list[StratumDecision]
    judgments: list[Judgment]
    # anchors certified: those of accepted strata and those accepted individually
    certified: list[AnchorRecord]


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


def entity_exists_in_kg(manager: KgManager, iri: str) -> bool:
    # subject and object position cover classes/instances; predicate position
    # is needed too since a property may appear only there (e.g. used in
    # triples without a separate rdf:type owl:ObjectProperty/DatatypeProperty
    # declaration triple of its own)
    wrapped = prepare_identifier_for_sparql(iri, manager.iri_literal_parser)
    sparql = (
        f"ASK {{ {{ {wrapped} ?p ?o }} UNION {{ ?s ?p {wrapped} }} "
        f"UNION {{ ?s {wrapped} ?o }} }}"
    )
    result = manager.execute_sparql(sparql)
    assert isinstance(result, AskResult)
    return result.boolean


def ensure_known(manager: KgManager, iri: str, known: set[str]) -> None:
    """
    Fast path: skip the SPARQL round-trip if the IRI was already surfaced by
    an earlier function call result during this conversation. Otherwise fall
    back to an actual live existence check against the KG, and only report
    the IRI as invalid to the LLM once that check has genuinely failed.
    """
    if iri in known:
        return

    if entity_exists_in_kg(manager, iri):
        known.add(iri)
        return

    raise FunctionCallException(
        f"The entity {iri} does not seem to exist in the knowledge graph. "
        "Double check the IRI."
    )


def resolve_managers(task: GraspTask, input: Any) -> tuple[KgManager, KgManager]:
    if isinstance(input, dict) and "source_kg" in input and "target_kg" in input:
        source, _ = find_manager(task.managers, input["source_kg"])
        target, _ = find_manager(task.managers, input["target_kg"])
        return source, target
    if len(task.managers) != 2:
        raise ValueError(
            "Input must name source_kg and target_kg unless exactly two KGs are configured"
        )
    return task.managers[0], task.managers[1]