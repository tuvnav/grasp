"""
How the context of an entity is built (entity_context.py).
"""
from pydantic import BaseModel, ConfigDict, Field

from grasp.tasks.om.foundation.types import IRI
from grasp.tasks.om.foundation.vocabulary import (
    LABEL_PREDICATES,
    SYNONYM_PREDICATES,
    TYPE_PREDICATES,
)


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
