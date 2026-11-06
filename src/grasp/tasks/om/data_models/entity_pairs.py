"""
Pairs of entities with their matches (shared keys, similar texts, ...), the
result of comparing the values of the two sides (kg_operations/value_comparison.py),
and the candidates built from them. Internal results of one computation, not stored: dataclasses.
"""
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field

from grasp.tasks.om.data_models.candidate_generators import CandidateStrategyType
from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.text_normalization import NormalizationLevel, normalization_level_order, sort_normalization_levels
from grasp.tasks.om.foundation.text_vectors import Vectorizer
from grasp.tasks.om.foundation.types import IRI, Side


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


@dataclass
class EntityPair[T: (KeyMatch, VectorMatch)](ABC):
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
        """The groups of both entities, by side (every row carries one group, and
        an entity in several groups has a row for each)."""
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
class KeyHub:
    """
    A key whose entities would give too many entity pairs (source entities x
    target entities above max_entity_pairs_per_bucket), so it gives none.
    """
    level: NormalizationLevel
    key: str
    num_source_entities: int
    num_target_entities: int


@dataclass
class VectorHub:
    """
    A pair of similar texts whose entities would give too many entity pairs
    (source entities x target entities above max_entity_pairs_per_bucket), so
    it gives none.
    """
    source_text: str
    target_text: str
    num_source_entities: int
    num_target_entities: int


@dataclass(frozen=True)
class Candidate(ABC):
    """
    A candidate of a candidate strategy, seen from one side: the entity on `side`
    gets the entity on the other side as candidate. A pair gives two candidates
    (one per side) when both entities are in their population. Abstract: each
    strategy adds its own fields in a subclass (KeyCandidate, ...).
    """
    side: Side                # side of the entity from whose perspective we are looking
    entity_pair: EntityPair

    @property
    def own_entity(self) -> IRI:
        return self.entity_pair.get_entity(self.side)

    @property
    def candidate_entity(self) -> IRI:
        return self.entity_pair.get_entity(Side.other_side(self.side))

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


class VectorCandidate(Candidate):
    entity_pair: VectorEntityPair


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
    rivals_outside_population: Counter[str] = field(default_factory=Counter)
    # rank key of the pair -> competing counterparts outside the population
    # (each counted once); they take places in the tie groups of the selection
    rivals_outside_by_rank: Counter[float] = field(default_factory=Counter)
    # partition -> counterparts whose partition does not compete with the entity's
    non_competing: Counter[str] = field(default_factory=Counter)

    @property
    def ambiguous_at_top_rank(self) -> bool:
        """More than one entity on the best rank: the strategy cannot tell which
        one is best. Rivals outside the population count too."""
        entries_by_rank = Counter(candidate.rank_key for candidate in self.candidates)
        entries_by_rank.update(self.rivals_outside_by_rank)
        entries_by_rank = +entries_by_rank          # drops ranks with count 0
        return bool(entries_by_rank) and entries_by_rank[min(entries_by_rank)] > 1
