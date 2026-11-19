from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum, auto
import math
import re
from typing import Sequence, Annotated, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


from grasp.tasks.om.foundation.text_normalization import EXTENDED_LEVELS, NORMALIZATION_LEVELS, REGEX_REPLACE, STANDARD_LEVELS, TRANSFORM_OPERATIONS, NormalizationLevel, TransformOperation, normalization_level_order, regex_replace, sort_normalization_levels
from grasp.tasks.om.foundation.types import IRI, EntityKey, Side
from grasp.tasks.om.data_models.profile import (
    GroupDefinition,
    PredicatePattern,
    StructureRelation,
    StructureRelationKind,
    INVERSE_PREFIX,
    Roles,
)
from grasp.tasks.om.foundation.text_vectors import Vectorizer



@dataclass(frozen=True)
class ValueRow:
    """
    One value of an entity. An entity in several groups of a population, or
    with several values, has several rows.
    """
    entity: IRI
    group_name: str             # group of the population the entity was retrieved for
    role_name: str              # role of the value: "names", "synonyms", ... or "own" (own pattern)
    raw_projection_value: str   # normalized keys are derived in value_comparison
    value_language: str | None   # language tag of a literal, if any (multilingual tracks)

    @property
    def primary(self) -> bool:
        """
        Whether the value is a name (the others only block ambiguous matches).
        Only for anchor task.
        """
        return self.role_name == "names"


# the entities of a side that are not matched yet
REST = "rest"

# a group of the profile by name, REST, or a definition for this strategy only
type GroupRef = str | GroupDefinition


def group_name_from_group_ref(group_ref: GroupRef):
    match group_ref:
        case str():
            return group_ref
        case GroupDefinition():
            return group_ref.name


class Population(BaseModel):
    """Source and target entities a strategy compares."""
    model_config = ConfigDict(extra="forbid")
    source_group_refs: list[GroupRef] = Field(default=[REST], min_length=1)
    target_group_refs: list[GroupRef] = Field(default=[REST], min_length=1)

    def group_names(self) -> dict[Side, list[str]]:
        return {
            Side.SOURCE: [group_name_from_group_ref(gr) for gr in self.source_group_refs],
            Side.TARGET: [group_name_from_group_ref(gr) for gr in self.target_group_refs]
        }


# values of an entity: one or more roles of its (effective) profile, or an own
# pattern over ?s (the entity) and ?o (the value); a pattern may compute ?o (BIND)
type ValuePatternSource = list[str | StructureRelation] | PredicatePattern


class ValuePatterns(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: ValuePatternSource = ["names", "synonyms"]
    target: ValuePatternSource | None = None  # None: like the source

    def by_side(self, side: Side) -> ValuePatternSource:
        return self.source if side == Side.SOURCE or self.target is None else self.target



class StringTransform(BaseModel):
    """A string transform applied to every value before the normalization
    ladder. Only what the ladder cannot do (case, word splitting, word sets,
    plural and inflection are levels)."""
    model_config = ConfigDict(extra="forbid")
    operation: TransformOperation
    pattern: str | None = None
    replacement: str = ""

    @model_validator(mode="after")
    def check_pattern(self) -> Self:
        if self.operation == REGEX_REPLACE:
            if not self.pattern:
                raise ValueError("regex_replace needs a pattern")
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"invalid regex: {e}") from e
        elif self.pattern is not None:
            raise ValueError(f"{self.operation} takes no pattern")
        return self

    def apply(self, text: str) -> str:
        if self.operation == REGEX_REPLACE:
            assert self.pattern is not None  # checked by check_pattern
            return regex_replace(text, self.pattern, self.replacement)
        return TRANSFORM_OPERATIONS[self.operation](text)


def apply_string_transforms(transforms: list[StringTransform], text: str) -> str:
    for transform in transforms:
        text = transform.apply(text)
    return text


class Normalization(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transforms: list[StringTransform] = []
    # levels tried from fine to coarse (the anchor rule records the finest level
    # on which a pair is unique); a single level for one fixed normalization
    levels: list[NormalizationLevel] = Field(default=list(STANDARD_LEVELS), min_length=1)

    @model_validator(mode="after")
    def levels_in_ladder_order(self) -> Self:
        order = [level for level in NORMALIZATION_LEVELS if level in self.levels]
        if order != self.levels:
            raise ValueError(f"levels must be distinct and in ladder order: {', '.join(NORMALIZATION_LEVELS)}")
        return self


class CandidateStrategyType(StrEnum):
    KEY = auto()
    VECTOR = auto()
    NEIGHBOURS = auto()


class KeyStrategy(BaseModel):
    """
    Candidates share a normalized value (e.g. a code, a cross reference, a
    name). The anchor rule is a key strategy on names with a unique selection.
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal[CandidateStrategyType.KEY] = CandidateStrategyType.KEY
    values: ValuePatterns = ValuePatterns()
    normalization: Normalization = Normalization()

    def value_pattern_source(self, side: Side, roles: Roles) -> ValuePatternSource:
        """The values compared on one side (roles: unused, all strategies take it)."""
        return self.values.by_side(side)


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

    def value_pattern_source(self, side: Side, roles: Roles) -> ValuePatternSource:
        """The values compared on one side (roles: unused, all strategies take it)."""
        return self.values.by_side(side)


# how the witnesses of a pair are scored: count (distinct witnesses; equal counts
# are real ties) or inverse_degree (a witness counts 1 / sqrt(source entities x
# target entities that share it), so rare witnesses decide; for instances with hubs)
type WitnessWeighting = Literal["count", "inverse_degree"]


class NeighbourStrategy(BaseModel):
    """Candidates have witnesses: a pair (x, y) is supported by a matched pair
    (a, b) with x related to a and y related to b by the same kind of structure
    relation in the same direction."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal[CandidateStrategyType.NEIGHBOURS] = CandidateStrategyType.NEIGHBOURS
    # the structure relations of these kinds in the profiles, in both directions ...
    relation_kinds: list[StructureRelationKind] = [
        StructureRelationKind.SUBSUMPTION,
        StructureRelationKind.PARTONOMY,
        StructureRelationKind.MEMBERSHIP
        ]
    # ... unless relations are named here, by name or defined inline, taken as
    # written ("^part_of": the other direction)
    relations: list[str | StructureRelation] | None = None
    depth: int = Field(default=1, ge=1, le=3)
    min_witnesses: int = Field(default=1, ge=1)
    witness_weighting: WitnessWeighting = "count"
    # generate: candidates from the neighbours of matched pairs; corroborate:
    # only count witnesses for the candidates of other generators
    mode: Literal["generate", "corroborate"] = "generate"

    def value_pattern_source(self, side: Side, roles: Roles) -> ValuePatternSource:
        """The structure relations whose neighbours are compared on one side: the
        named relations as written, else those of relation_kinds in the profile
        (roles), in both directions."""
        if self.relations is not None:
            return self.relations
        names = [relation.name for relation in roles.structure if relation.kind in self.relation_kinds]
        both_directions: list[str | StructureRelation] = [*names, *(INVERSE_PREFIX + name for name in names)]
        return both_directions


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


@dataclass(frozen=True)
class Match(ABC):
    source_rows: tuple[ValueRow, ...]
    target_rows: tuple[ValueRow, ...]

    def __post_init__(self):
        source_entities = {row.entity for row in self.source_rows}
        target_entities = {row.entity for row in self.target_rows}
        assert len(source_entities) > 0 and len(target_entities) > 0, \
            "Match instances must be initialised with ValueRows for both source and target" \
            " but this one was initialised with empty source_rows or empty target_rows."
        assert len(source_entities) == 1, "Match class maps to more than one source entity!"
        assert len(target_entities) == 1, "Match class maps to more than one target entity!"

    @property
    def source_entity(self) -> IRI:
        return self.source_rows[0].entity  # __post_init__: not empty, one entity

    @property
    def target_entity(self) -> IRI:
        return self.target_rows[0].entity

    @property
    @abstractmethod
    def kind(self) -> CandidateStrategyType:
        ...


@dataclass(frozen=True)
class KeyMatch(Match):
    """
    One key shared by ONE source entity and ONE target entity on one level,
    with the value rows of just these two entities under the key.
    """
    level: NormalizationLevel
    key: str

    @property
    def kind(self) -> CandidateStrategyType:
        return CandidateStrategyType.KEY


@dataclass(frozen=True)
class VectorMatch(Match):
    source_text: str
    target_text: str
    cosine: float

    @property
    def kind(self) -> CandidateStrategyType:
        return CandidateStrategyType.VECTOR


@dataclass(frozen=True)
class NeighbourMatch(Match):
    """
    One witness pair shared by one source 
    and one target entity over corresponding relations,
    with the rows of just these two entities.
    """
    relation_key: str                # e.g. "kind:partonomy"
    witness: tuple[IRI, IRI]         # the equivalence (a, b) of the neighbours
    # entities that share this witness over this relation (the bucket), per side
    num_source_entities: int
    num_target_entities: int

    @property
    def weight(self) -> float:
        """1 / sqrt(source entities x target entities sharing the witness): 1 for a
        witness of just these two entities, small for a hub."""
        return 1 / math.sqrt(self.num_source_entities * self.num_target_entities)

    @property
    def kind(self) -> CandidateStrategyType:
        return CandidateStrategyType.NEIGHBOURS


@dataclass
class EntityPair[T: (KeyMatch, VectorMatch, NeighbourMatch)](ABC):
    """
    A source and a target entity with all their matches in the run.
    """
    source_entity: IRI
    target_entity: IRI
    matches: list[T]

    def get_entity(self, side: Side) -> IRI:
        return self.source_entity if side == Side.SOURCE else self.target_entity

    def append_match(self, match: T):
        self.matches.append(match)

    def role_pairs(self) -> set[tuple[str, str]]:
        """
        (source role, target role) of all values that matched,
        e.g. ("names", "synonyms").
        """
        return {
            (source_row.role_name, target_row.role_name)
            for match in self.matches
            for source_row in match.source_rows
            for target_row in match.target_rows
        }

    def entity_groups(self) -> dict[Side, set[str]]:
        """
        The groups of both entities, by side (every row carries one group, and
        an entity in several groups has a row for each).
        """
        group_names_by_side: dict[Side, set[str]] = {
            Side.SOURCE: set(), Side.TARGET: set()
            }
        for match in self.matches:
            group_names_by_side[Side.SOURCE].update(row.group_name for row in match.source_rows)
            group_names_by_side[Side.TARGET].update(row.group_name for row in match.target_rows)
        return group_names_by_side

    @property
    @abstractmethod
    def rank_key(self) -> float:
        """
        Smaller is better.
        """
        ...


@dataclass
class KeyEntityPair(EntityPair[KeyMatch]):
    """
    Entity pair that shares at least one key. All of
    its shared keys on all levels are listed here
    (one KeyMatch per level and key).
    """

    def levels(self) -> list[NormalizationLevel]:
        """
        The levels on which the pair shares a key, from fine to coarse.
        """
        return sort_normalization_levels(key_match.level for key_match in self.matches)

    def finest_normalization_level(self) -> NormalizationLevel:
        return self.levels()[0]  # a pair has at least one key match

    @property
    def rank_key(self) -> int:
        return normalization_level_order(self.finest_normalization_level())


@dataclass
class VectorEntityPair(EntityPair[VectorMatch]):

    def cosines(self) -> list[float]:
        return [m.cosine for m in self.matches]

    def best_cosine(self) -> float:
        return max(self.cosines())

    @property
    def rank_key(self) -> float:
        """
        Returns negated best cosine of entity
        pair.
        In rank keys, smaller is better, hence
        negated.
        """
        return -round(self.best_cosine(), 6)


@dataclass
class NeighbourEntityPair(EntityPair[NeighbourMatch]):
    """
    Entity pair with at least one witness. witness_weighting: how its witnesses
    are scored (the same for all pairs of a run).
    """
    witness_weighting: WitnessWeighting = "count"

    def witnesses(self) -> set[tuple[IRI, IRI]]:
        return {match.witness for match in self.matches}

    def witness_score(self) -> float:
        """count: the distinct witnesses (one witness over two relations counts
        once); inverse_degree: the sum of their weights (of a witness over two
        relations: its larger weight)."""
        if self.witness_weighting == "count":
            return len(self.witnesses())
        weight_by_witness: dict[tuple[IRI, IRI], float] = {}
        for match in self.matches:
            weight_by_witness[match.witness] = max(weight_by_witness.get(match.witness, 0.0), match.weight)
        return sum(weight_by_witness.values())

    @property
    def rank_key(self) -> float:
        """Smaller is better: the negative witness score; a weighted score rounded
        to 6 places (only float noise: equal scores are ties)."""
        return -round(self.witness_score(), 6)


@dataclass
class Hub(ABC):
    num_source_entities: int
    num_target_entities: int


@dataclass
class KeyHub(Hub):
    """
    A key whose entities would give too many entity pairs (source entities x
    target entities above max_entity_pairs_per_bucket), so it gives none.
    """
    level: NormalizationLevel
    key: str


@dataclass
class VectorHub(Hub):
    """
    A pair of similar texts whose entities would give too many entity pairs
    (source entities x target entities above max_entity_pairs_per_bucket), so
    it gives none.
    """
    source_text: str
    target_text: str


@dataclass
class NeighbourHub(Hub):
    """
    A witness pair whose entities would give
    too many entity pairs: it creates no pairs,
    but counts for the pairs created by other witnesses.
    """
    relation_key: str
    witness: tuple[IRI, IRI]


@dataclass(frozen=True)
class Candidate(ABC):
    """
    A candidate of a candidate strategy, seen from one side: the entity on `side`
    gets the entity on the other side as candidate. A pair gives two candidates
    (one per side) when both entities are in their population. Abstract: each
    strategy adds its own fields in a subclass (KeyCandidate, ...).
    """
    own_side: Side                # side of the entity from whose perspective we are looking
    entity_pair: EntityPair
    # the candidate entity already has an equivalence (with another entity): it
    # stays a candidate at its rank (it may be a second equivalence), but cannot
    # be unique (it gets no candidate list of its own). Keyword-only, so that the
    # subclasses can add fields without defaults
    candidate_already_matched: bool = field(default=False, kw_only=True)

    @property
    def own_entity(self) -> IRI:
        return self.entity_pair.get_entity(self.own_side)

    @property
    def candidate_entity(self) -> IRI:
        return self.entity_pair.get_entity(Side.other_side(self.own_side))

    @property
    def rank_key(self) -> float:
        return self.entity_pair.rank_key


@dataclass(frozen=True)
class KeyCandidate(Candidate):
    """
    A candidate of the key strategy. The two candidates of a pair differ in
    rivals_of_candidate_by_level.
    """
    entity_pair: KeyEntityPair       # levels, keys, rows of both entities
    # finest level on which both entities have no other competing counterpart;
    # None: ambiguous on every level of the pair
    finest_unique_level: NormalizationLevel | None
    # per level of the pair: competing entities of the entity's own side that also
    # match the candidate there (the candidate's other counterparts)
    num_rivals_of_candidate_by_level: dict[NormalizationLevel, int]

    @property
    def num_rivals_of_candidate_on_finest_level(self) -> int:
        return self.num_rivals_of_candidate_by_level[
            self.entity_pair.finest_normalization_level()
            ]


@dataclass(frozen=True)
class VectorCandidate(Candidate):
    entity_pair: VectorEntityPair


@dataclass(frozen=True)
class NeighbourCandidate(Candidate):
    entity_pair: NeighbourEntityPair


@dataclass
class EntityCandidates:
    """
    What the key strategy found for one entity of a population (the collection;
    the selection is in candidate_selection): all its candidates and its other
    counterparts in summary. Facts only, independent of the Selection settings,
    so that a selection can be run again with other settings.
    """
    side: Side
    entity: IRI
    candidates: list[Candidate] = field(default_factory=list)
    # group -> competing counterparts outside the population (a counterpart in two
    # groups counts in both)
    num_counterparts_outside_population_by_group: Counter[str] = field(default_factory=Counter)
    # rank key of the pair -> competing counterparts outside the population
    # (each counted once); they take places in the tie groups of the selection
    num_counterparts_outside_population_by_rank: Counter[float] = field(default_factory=Counter)
    # partition -> counterparts whose partition does not compete with the entity's
    num_non_competing_counterparts_by_partition: Counter[str] = field(default_factory=Counter)

    @property
    def top_rank(self) -> float:
        return min(candidate.rank_key for candidate in self.candidates)

    @property
    def already_matched_at_top_rank(self) -> bool:
        """
        A candidate on the best rank already has an equivalence (with another
        entity), independent of the Selection settings.
        """
        if not self.candidates:
            return False
        top_rank = self.top_rank  # once: it goes over all candidates
        return any(candidate.candidate_already_matched for candidate in self.candidates
                   if candidate.rank_key == top_rank)

    def iris_of_already_matched_at_top_rank(self) -> list[IRI]:
        if not self.candidates:
            return []
        top_rank = self.top_rank  # once: it goes over all candidates
        return sorted(
            candidate.candidate_entity for candidate in self.candidates
            if candidate.candidate_already_matched
            and candidate.rank_key == top_rank
            )

    def get_top_rank_candidates(self) -> list[IRI]:
        if not self.candidates:
            return []
        top_rank = self.top_rank
        return sorted(
            candidate.candidate_entity for candidate in self.candidates
            if candidate.rank_key == top_rank
        )

    @property
    def ambiguous_at_top_rank(self) -> bool:
        """More than one entity on the best rank: the strategy cannot tell which
        one is best. Rivals outside the population count too."""
        entries_by_rank = Counter(candidate.rank_key for candidate in self.candidates)
        entries_by_rank.update(self.num_counterparts_outside_population_by_rank)
        entries_by_rank = +entries_by_rank          # drops ranks with count 0
        return bool(entries_by_rank) and entries_by_rank[min(entries_by_rank)] > 1


class AnchorSpecificity(StrEnum):
    """
    Whether the words of an anchor's name are extended by another name of one
    of the KGs ({paper} by {poster, paper}): a head anchor may name a broader
    or narrower thing than its partner (granularity homonymy), a specific one
    hardly. Only for names; for codes and other values not applicable.
    """
    SPECIFIC = "specific"
    HEAD = "head"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Anchor:
    """A pair that is unique in at least one run of the ensemble (the anchor rule)."""
    source_entity: IRI
    target_entity: IRI
    families: frozenset[str]   # families of the runs that found it
    normalization_level: NormalizationLevel  # finest normalization level of its key over these runs
    specificity: AnchorSpecificity
    # lexical information content of its name (Seco 2004), the minimum over both
    # sides; 1: specific, lower: extended by more names; None: not applicable
    information_content: float | None

    @property
    def pair(self) -> tuple[IRI, IRI]:
        return self.source_entity, self.target_entity

    @property
    def extended(self) -> bool:
        """Unique only on a coarse level (word set, plural, inflection)."""
        return self.normalization_level in EXTENDED_LEVELS


type RunRole = Literal["generate", "corroborate"]

# entity -> candidate IRI -> run id -> rank of the candidate in that run (1 = first tie group)
type CandidateRankIndex = dict[EntityKey, dict[IRI, dict[str, int]]]
type ParetoIndex = dict[EntityKey, list[list[list[IRI]]]]
type VennIndex = dict[frozenset[str], list[tuple[IRI, IRI]]]


class EnsembleRun(BaseModel):
    """
    A tried run in an ensemble of runs to merge. family: the kind of its
    evidence, named by the LLM: runs of one family count as one source of
    evidence for a pair (a Venn cell is a set of families), whatever their
    roles; None: a family of its own.
    """
    model_config = ConfigDict(extra="forbid")
    run_id: str
    role: RunRole = "generate"
    family: str | None = None

    @property
    def family_name(self) -> str:
        return self.family or self.run_id


class RunEnsemble(BaseModel):
    """
    The runs whose candidates are merged, each with its role and family.
    """
    model_config = ConfigDict(extra="forbid")
    runs: list[EnsembleRun]

    @model_validator(mode="after")
    def runs_distinct_and_one_generating(self) -> Self:
        run_ids = [run.run_id for run in self.runs]
        duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
        if duplicates:
            raise ValueError(f"runs named more than once: {duplicates}")
        if not self.run_ids("generate"):
            raise ValueError("an ensemble needs at least one run with the role generate")
        return self

    def run_ids(self, role: RunRole | None = None) -> list[str]:
        """
        The ids of the runs (of a role), sorted: the fixed order of the
        positions in every rank tuple.
        """
        return sorted(
            run.run_id for run in self.runs if role is None or run.role == role
            )

    def family(self, run_id: str) -> str:
        return next(run.family_name for run in self.runs if run.run_id == run_id)


@dataclass(frozen=True)
class SelectedCandidates:
    """
    The candidates of one entity after the selection:
    whole tie groups in rank order.
    """
    collected: EntityCandidates                  # what was found (unchanged)
    # groups of candidates with the same rank. group's position = rank (0 = best)
    tie_rank_groups: list[list[Candidate]]
    # rank key of a tie group dropped although places were free (top_k would have
    # cut it in the middle); None: no such group
    dropped_tie_group_rank: float | None
    # candidates that have the entity in their own
    # selection too (after selection process in both sides)
    mutual_candidates: frozenset[IRI]
    # the anchor rule: this candidate stands alone in front for the entity (first
    # tie group, no rivals outside the population on its rank or better) AND the
    # entity stands alone in front for it; implies mutual. None otherwise
    unique_candidate: IRI | None

    @property
    def selected_candidates(self) -> list[Candidate]:
        """
        All selected candidates, best group first.
        """
        return [candidate for tie_group in self.tie_rank_groups for candidate in tie_group]


@dataclass(frozen=True)
class StrategySignals:
    """
    Summary of a strategy's selection on one side, for the LLM. All shares
    are of the whole population of the side without the entities that already
    have an equivalence (they get no candidates); entities without any match
    count too: they are the ones the strategy does not cover.

    In a trial run, the entities of one side may be a random sample of its
    population (sampled_side; population_size is then the sample size). On the
    sampled side, coverage, mean_list_length and ambiguous_at_top_rank are
    estimates of the full run, but unique (and the mutual candidates) are upper
    estimates: the other side sees only the sampled entities, so a candidate may
    seem alone in front there. The shares of the other side are distorted (it
    sees only a sample of its counterparts).
    """
    side: Side
    population_size: int
    coverage: float          # share with at least one selected candidate
    mean_list_length: float  # selected candidates per entity, over those with at least one
    ambiguous_at_top_rank: float  # share with more than one entity on the best rank
    unique: float            # share with a unique candidate (the anchor rule)
    # share whose best rank holds a candidate that already has an equivalence
    # (with another entity): the strategy mostly finds the partners of others,
    # noise or a second equivalence
    already_matched_at_top_rank: float = 0.0
    sampled_side: Side | None = None  # trial run: the side whose population was sampled


@dataclass(frozen=True)
class StrategyResult[H: Hub]:
    """
    One run of a candidate strategy. The collection is kept with the selection,
    so the selection can be repeated with other settings (select_candidates on
    `collected`) without retrieving and comparing again. H: the hub kind of the
    strategy (KeyHub, VectorHub).
    """
    # per side the entities of the run's population (generator.population),
    # without those that already have an equivalence; None: a large KG, whose
    # population is not retrieved
    population: dict[Side, frozenset[IRI] | None]
    # per side the entities the run compared: the population, or on the side
    # sampled in a trial run the sample
    compared_entities: dict[Side, frozenset[IRI]]
    collected: dict[tuple[Side, IRI], EntityCandidates]  # what was found, per (side, entity)
    selected: dict[tuple[Side, IRI], SelectedCandidates]  # after the Selection settings
    signals: dict[Side, StrategySignals]
    # the equivalences found so far that the strategy finds again (as entity
    # pairs, before the selection), of existing_equivalences in all: an estimate
    # of its recall on them
    confirmed_equivalences: list[tuple[IRI, IRI]]
    existing_equivalences: int
    # keys/texts with too many entity pairs to pair them
    # (a list of KeyHub, VectorHub or NeighbourHub)
    hubs: Sequence[H]  # make Pyright happy: list not covariant
    # key strategy: per side the word sets of all values compared (the
    # competition set, not only the population), to tell head anchors from
    # specific ones; None: another strategy
    value_wordsets: dict[Side, frozenset[frozenset[str]]] | None = None


class TriedRun(BaseModel):
    """A run the LLM tried in the candidate stage, in summary (the candidates
    stay in the task state)."""
    model_config = ConfigDict(extra="forbid")
    generator: Generator
    signals: dict[Side, StrategySignals]
    confirmed_equivalences: int  # existing equivalences found again, of existing_equivalences
    existing_equivalences: int
    num_hubs: int


class CandidateStageResult(BaseModel):
    """The output of the candidate stage: the ensemble of runs selected for the
    full runs, and all tried runs. finished: the LLM called finish (otherwise
    nothing is selected)."""
    model_config = ConfigDict(extra="forbid")
    source_kg: str
    target_kg: str
    finished: bool
    rationale: str | None
    run_ensemble: RunEnsemble | None  # the selected runs; None: none
    runs: dict[str, TriedRun]


@dataclass(frozen=True)
class MergedCandidates:
    """
    The candidate lists of an ensemble of runs, merged per entity
    (kg_operations/candidate_selection.py: merge_runs).
    """
    run_ensemble: RunEnsemble
    candidate_rank_index: CandidateRankIndex
    # per entity: Pareto layers of the generate runs, each in sublayers by the
    # corroborate runs; before and after the budget
    uncut_pareto_layers_by_entity: ParetoIndex
    pareto_layers_by_entity: ParetoIndex
    # the pairs of the cut lists by the families of the runs that selected them
    venn_index: VennIndex
    # entities whose first sublayer is larger than allowed: no list
    too_ambiguous_entities: frozenset[EntityKey]
    # per side the entities on which the merge equals a merge of full runs
    # (evaluable_entities); None: the side is not measurable
    evaluable_entities: dict[Side, frozenset[IRI] | None]


@dataclass(frozen=True)
class EnsembleEvaluation:
    """
    The merge of an ensemble of trial runs on the evaluable entities of one
    side, for the LLM (evaluate_merged_candidates). Pairs are (entity of the
    side, candidate).
    """
    side: Side
    num_evaluated: int
    num_with_list: int
    num_too_ambiguous: int
    num_cut_by_budget: int  # lists that lost candidates to the budget
    list_length_counts: dict[str, int]  # lists per length class ("1", "2-3", ...)
    pairs_by_cell: list[tuple[frozenset[str], int]]  # number of pairs per Venn cell, largest first
    cell_examples: list[tuple[frozenset[str], IRI, IRI]]  # cell, entity, its candidate in the cell
    # entity, size of its first sublayer, the first candidates of it
    too_ambiguous_examples: list[tuple[IRI, int, list[IRI]]]

    @property
    def num_without_candidates(self) -> int:
        return self.num_evaluated - self.num_with_list - self.num_too_ambiguous


class Stratum(BaseModel):
    """
    The pairs of one Venn cell: found by exactly these run families; label: a
    further division of the cell (anchors: e.g. head or specific), empty if none.
    """
    families: frozenset[str]
    pairs: list[tuple[IRI, IRI]]
    label: str = ""

    @property
    def name(self) -> str:
        cell = "+".join(sorted(self.families))
        return f"{cell}/{self.label}" if self.label else cell


class AnchorStageResult(BaseModel):
    """
    The output of the anchor stage: the ensemble of key runs selected for the
    full runs (None: no anchors, e.g. no strategy gives reliable ones), and all
    tried runs. finished: the LLM called finish.
    """
    source_kg: str
    target_kg: str
    finished: bool
    rationale: str | None
    run_ensemble: RunEnsemble | None
    runs: dict[str, TriedRun]


@dataclass(frozen=True)
class AnchorMerge:
    """
    The anchors of an ensemble of key runs. Each run gives only 1:1 pairs, but
    two runs may give an entity different partners: these pairs are conflicts,
    left out of the anchors (they become candidates of a later stage).
    """
    anchors: list[Anchor]
    conflicts: list[tuple[IRI, IRI, frozenset[str]]]  # (source, target, families)

