"""
Data model of the candidate stage (after the anchor certification): a
generator produces, per source entity, a list of target candidates.

    Generator = Population   which entities (scope_model)
              × Strategy     how candidates are found, one of three families:
                             key         exact equality of normalized values (hash join)
                             vector      similarity of values in a vector space (top-k search)
                             neighbours  witnesses: neighbours already matched to each other
              × Selection    how scores become a candidate list

Key and vector strategies compare values (scope_model.SideValues), neighbour
strategies the current alignment, so they change from round to round.
"""
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grasp.tasks.om.model import StructureRelation
from grasp.tasks.om.scope_model import Normalization, Population, SideValues


class KeyStrategy(BaseModel):
    """Candidates share a normalized value (e.g. a code, a cross reference, a
    name). The anchor rule is a key strategy on names with a unique selection."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["key"] = "key"
    values: SideValues = SideValues()
    normalization: Normalization = Normalization()


class VectorStrategy(BaseModel):
    """Candidates are the most similar targets: values are turned into vectors
    (IDF-weighted word tokens, character 3-grams or an embedding model) and
    compared by cosine; the best score over the value pairs counts."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["vector"] = "vector"
    values: SideValues = SideValues()
    normalization: Normalization = Normalization(levels=["words"])
    vectorizer: Literal["idf_tokens", "char_3grams", "embedding"] = "idf_tokens"
    model: str | None = None  # embedding model; None: the default model

    @model_validator(mode="after")
    def model_only_for_embedding(self) -> Self:
        if self.model is not None and self.vectorizer != "embedding":
            raise ValueError("model is only used with the embedding vectorizer")
        return self


class NeighbourStrategy(BaseModel):
    """Candidates have witnesses: a pair (x, y) is supported by a matched pair
    (a, b) with x related to a and y related to b by the same kind of structure
    relation in the same direction."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["neighbours"] = "neighbours"
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
    """From scores to the candidate list of a source entity."""
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=5, ge=1)
    mutual: bool = False  # only candidates for which the source is among the target's top_k too
    unique: bool = False  # only sources with exactly one candidate, which has no other source (anchor rule)
    # key strategies: drop values shared by more targets than this; None: no limit
    max_ambiguity: int | None = Field(default=3, ge=1)


class Generator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population: Population = Population()
    strategy: CandidateStrategy = KeyStrategy()
    selection: Selection = Selection()
