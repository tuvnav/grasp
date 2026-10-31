"""
Predicate patterns (predicates and SPARQL patterns over ?s and ?o) and
structure relations, with the default structure relations.
"""
from enum import StrEnum, auto
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from grasp.tasks.om.foundation.types import IRI
from grasp.tasks.om.foundation.vocabulary import RDF_TYPE, VOCABULARY
from grasp.tasks.om.sparql_access.query_building import sparql_path


class RelationAlgebra(BaseModel):
    symmetric: bool = False
    transitive: bool = False
    inverse_of: Optional[str] = None


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

    def parts(self) -> list[str]:
        """The alternatives: each pattern, and the predicates as one path."""
        parts = list(self.patterns)
        if self.predicates:
            parts.append(f"?s {sparql_path(self.predicates)} ?o")
        return parts

    def alternatives(self) -> list[str]:
        """The alternatives for querying: each pattern, and each predicate on
        its own (a path <a>|<b> may be evaluated on the whole KG first)."""
        return list(self.patterns) + [f"?s <{p}> ?o" for p in self.predicates]

    def predicates_to_sparql(self) -> str:
        return " UNION ".join(f"{{ {p} }}" for p in self.parts())


class StructureRelationKind(StrEnum):
    SUBSUMPTION = auto()
    MEMBERSHIP = auto()
    PARTONOMY = auto()
    IDENTITY = auto()
    DISJOINTNESS = auto()
    ASSOCIATION = auto()
    OTHER = auto()


DEFAULT_KIND_ALGEBRA = {
    StructureRelationKind.SUBSUMPTION: {"transitive": True},
    StructureRelationKind.IDENTITY: {"transitive": True, "symmetric": True},
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
    _structure("existential_restriction", StructureRelationKind.ASSOCIATION,
               patterns=[f"?s <{_RDFS}subClassOf> [ <{_OWL}onProperty> ?r ; <{_OWL}someValuesFrom> ?o ] . FILTER(isIRI(?o))"]),
    _structure("same_as", StructureRelationKind.IDENTITY,
               [_OWL + "sameAs", _OWL + "equivalentClass", _OWL + "equivalentProperty"]),
    _structure("disjoint_with", StructureRelationKind.DISJOINTNESS,
               [_OWL + "disjointWith", _OWL + "propertyDisjointWith"]),
]
