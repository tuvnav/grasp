"""
Data model of the candidate stage (after the anchor certification): a
generator produces, per source entity, a list of target candidates.

    Generator = Population   which entities (scope.py)
              × Strategy     how candidates are found, one of three families:
                             key         exact equality of normalized values (hash join)
                             vector      similarity of values in a vector space (top-k search)
                             neighbours  witnesses: neighbours already matched to each other
              × Selection    how scores become a candidate list

Key and vector strategies compare values (scope.ValuePatterns), neighbour
strategies the current alignment, so they change from round to round.
"""
from enum import StrEnum, auto
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grasp.tasks.om.data_models.patterns import StructureRelation
from grasp.tasks.om.data_models.scope import Normalization, Population, ValuePatterns
from grasp.tasks.om.foundation.text_vectors import Vectorizer


class CandidateStrategyType(StrEnum):
    KEY = auto()
    VECTOR = auto()
    NEIGHBOURS = auto()


class KeyStrategy(BaseModel):
    """Candidates share a normalized value (e.g. a code, a cross reference, a
    name). The anchor rule is a key strategy on names with a unique selection."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal[CandidateStrategyType.KEY] = CandidateStrategyType.KEY
    values: ValuePatterns = ValuePatterns()
    normalization: Normalization = Normalization()


class VectorStrategy(BaseModel):
    """Candidates are the most similar targets: values are turned into vectors
    (IDF-weighted word tokens, character 3-grams or an embedding model) and
    compared by cosine; the best score over the value pairs counts."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal[CandidateStrategyType.VECTOR] = CandidateStrategyType.VECTOR
    values: ValuePatterns = ValuePatterns()
    normalization: Normalization = Normalization(levels=["words"])
    vectorizer: Vectorizer = "idf_tokens"
    model_name: str | None = None  # embedding model; None: foundation.text_vectors.DEFAULT_EMBEDDING_MODEL

    @model_validator(mode="after")
    def model_only_for_embedding(self) -> Self:
        if self.model_name is not None and self.vectorizer != "embedding":
            raise ValueError("model is only used with the embedding vectorizer")
        return self


class NeighbourStrategy(BaseModel):
    """Candidates have witnesses: a pair (x, y) is supported by a matched pair
    (a, b) with x related to a and y related to b by the same kind of structure
    relation in the same direction."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal[CandidateStrategyType.NEIGHBOURS] = CandidateStrategyType.NEIGHBOURS
    # structure relations of the profiles by name, or defined here; None: the
    # subsumption, partonomy and membership relations of the profiles
    relations: list[str | StructureRelation] | None = None
    depth: int = Field(default=1, ge=1, le=3)
    min_witnesses: int = Field(default=1, ge=1)
    # neighbours with more relations give no candidates (hubs make the
    # candidates quadratic and say little)
    max_degree: int = Field(default=50, ge=1)
    # generate: candidates from the neighbours of matched pairs; corroborate:
    # only count witnesses for the candidates of other generators
    mode: Literal["generate", "corroborate"] = "generate"


CandidateStrategy = Annotated[KeyStrategy | VectorStrategy | NeighbourStrategy, Field(discriminator="kind")]


class Selection(BaseModel):
    """
    From the collected candidates to the candidate list of an entity. Tied
    candidates (key strategy: same finest level) are taken whole or not at all:
    a tie group that does not fit into top_k is dropped and the entity counts as
    ambiguous, so a value shared by more than top_k entities gives no candidates.
    Whether a candidate is mutual, and the unique candidate, are always computed;
    mutual and unique only decide whether they also filter the list.
    """
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=5, ge=1)
    mutual: bool = False  # keep only candidates whose own list contains the entity too
    unique: bool = False  # keep only the unique candidate: alone on the finest level and mutual (anchor rule)


class Generator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population: Population = Population()
    strategy: CandidateStrategy = KeyStrategy()
    selection: Selection = Selection()
