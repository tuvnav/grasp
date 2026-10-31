"""
Vocabulary: standard prefixes, the default predicate canon (names, synonyms, ...),
the kind memberships of the default partitions, and IRI helpers.
"""
import re
from urllib.parse import urlsplit

from grasp.tasks.om.foundation.types import IRI

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


# strong definitions first; rdfs:comment is often a definition, but may be any note
DEFINITION_PREDICATES = [
    IRI("http://www.w3.org/2004/02/skos/core#definition"),
    IRI("http://purl.obolibrary.org/obo/IAO_0000115"),
    IRI("http://purl.org/dc/terms/description"),
    IRI("http://www.w3.org/2000/01/rdf-schema#comment"),
]


# codes of the entity itself within its own system
CODE_PREDICATES = [
    IRI("http://www.w3.org/2004/02/skos/core#notation"),
    IRI("http://www.geneontology.org/formats/oboInOwl#id"),
    IRI("http://purl.org/dc/elements/1.1/identifier"),
    IRI("http://purl.org/dc/terms/identifier"),
]


# codes or IRIs of the entity in other systems; two entities referring to
# the same external entity are anchor candidates
CROSS_REFERENCE_PREDICATES = [
    IRI("http://www.geneontology.org/formats/oboInOwl#hasDbXref"),
    IRI("http://www.w3.org/2004/02/skos/core#exactMatch"),
    IRI("http://www.w3.org/2004/02/skos/core#closeMatch"),
]


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


# Partition contract: ?entity ?partition. Joined optionally, entities without a
# partition get UNCATEGORIZED. An entity may be in several partitions (e.g.
# punning). ?evidence is for inspection only.
UNCATEGORIZED = "uncategorized"


# membership matched by every subject: in the profile, the entities outside all
# partitions are UNCATEGORIZED
UNIVERSE = "?entity ?anyPredicate ?anyObject"


# partition of all entities when a side has no partition query
ALL = "all"


# Default partitions of the profile (stage b1) by kind: membership patterns over
# ?entity, as alternatives, in order. Partitions are exclusive, an entity belongs
# to the first that matches: an entity used as a predicate is a property even if
# it is also typed as a class (punning), a class is not also an instance.
# KIND_PARTITION_SPARQL above is the older, non-exclusive form of the anchor stage.
_NOT_VOCABULARY_TYPE = ("FILTER(isIRI(?type) && !STRSTARTS(STR(?type), STR(owl:)) && !STRSTARTS(STR(?type), STR(rdfs:)) "
                        "&& !STRSTARTS(STR(?type), STR(rdf:)) && !STRSTARTS(STR(?type), STR(skos:)))")


KIND_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "property": (
        "?anySubject ?entity ?anyObject",
        "?entity rdf:type ?type VALUES ?type { owl:ObjectProperty owl:DatatypeProperty owl:AnnotationProperty rdf:Property }",
    ),
    "class": (
        "?entity rdf:type ?type VALUES ?type { owl:Class rdfs:Class }",
        "?anyInstance rdf:type ?entity",
        "?entity rdfs:subClassOf ?anySuper",
        "?anySub rdfs:subClassOf ?entity",
    ),
    "concept": ("?entity rdf:type skos:Concept",),
    "instance": (
        # owl:Thing: e.g. DBkWik articles typed with nothing else
        "?entity rdf:type ?type VALUES ?type { owl:NamedIndividual owl:Thing }",
        f"?entity rdf:type ?type {_NOT_VOCABULARY_TYPE}",
        # untyped individuals (e.g. wiki articles): data beyond names; helper nodes
        # that only carry a label match no kind (in a large KG left out: the
        # negations would be evaluated on the whole KG)
        "?entity ?p ?o FILTER(?p NOT IN (rdf:type, rdfs:label, rdfs:comment, skos:prefLabel, skos:altLabel)) "
        "FILTER NOT EXISTS { ?entity rdf:type ?anyType } FILTER NOT EXISTS { ?anyInstance rdf:type ?entity } "
        "FILTER NOT EXISTS { ?entity rdfs:subClassOf ?anySuper } FILTER NOT EXISTS { ?anySub rdfs:subClassOf ?entity } "
        "FILTER NOT EXISTS { ?anySubject ?entity ?anyObject }",
    ),
}


# prologue of the vocabulary prefixes used in membership patterns
VOCABULARY_PROLOGUE = "\n".join(f"PREFIX {prefix}: <{namespace}>" for namespace, prefix in VOCABULARY_PREFIXES.items())
