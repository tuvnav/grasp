"""
Basic types shared by all layers: IRI, Side of a KG pair, SPARQL binding types, name types.
"""
from enum import StrEnum
from typing import NewType

from grasp.sparql.types import Binding


class Side(StrEnum):
    """Side of a KG pair: the source or the target KG (also of a correspondence)."""
    SOURCE = "source"
    TARGET = "target"


SIDES = tuple(Side)


# an IRI as such, e.g. http://human.owl#NCI_C12432
IRI = NewType("IRI", str)


# an IRI or literal prepared for insertion into SPARQL text, e.g. <http://human.owl#NCI_C12432>
SparqlTerm = NewType("SparqlTerm", str)


class BindingType(StrEnum):
    """Values of Binding.type (SPARQL JSON results); equal to the plain strings
    GRASP uses, so it works with GRASP's unchanged Binding objects."""
    URI = "uri"
    LITERAL = "literal"
    BNODE = "bnode"


# blank node -> its (predicate, object) edges
type BlankNodeTree = dict[str, list[tuple[IRI, Binding]]]


class NameType(StrEnum):
    LABEL = "label"
    CODE = "code"
    OTHER = "other"
