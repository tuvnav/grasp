"""
Pairs of entities that share keys, the result of intersecting the value indices
of the two sides (kg_operations/value_comparison.py), and the candidates built
from them. Internal results of one computation, not stored: dataclasses.
"""
from collections import Counter
from dataclasses import dataclass, field

from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.text_normalization import NormalizationLevel, sort_normalization_levels
from grasp.tasks.om.foundation.types import IRI, Side


@dataclass(frozen=True)
class KeyMatch:
    """
    One key shared by ONE source entity and ONE target entity on one level,
    with the value rows of just these two entities under the key.
    """
    level: NormalizationLevel
    key: str
    source_rows: tuple[ValueRow, ...]
    target_rows: tuple[ValueRow, ...]

    def __post_init__(self):
        source_entities = {row.entity for row in self.source_rows}
        target_entities = {row.entity for row in self.target_rows}
        assert len(source_entities) > 0 and len(target_entities) > 0, \
            "KeyMatch must be initialised with ValueRows for both source and target" \
            " but was initialised with empty source_rows or empty target_rows."
        assert len(source_entities) == 1, "KeyMatch maps to more than one source entity!"
        assert len(target_entities) == 1, "KeyMatch maps to more than one target entity!"

    @property
    def source_entity(self) -> IRI:
        return self.source_rows[0].entity  # __post_init__: not empty, one entity

    @property
    def target_entity(self) -> IRI:
        return self.target_rows[0].entity


@dataclass
class EntityKeyPair:
    """A source and a target entity that share at least one key, with all their
    shared keys on all levels (one KeyMatch per level and key)."""
    source_entity: IRI
    target_entity: IRI
    key_matches: list[KeyMatch]

    def append_key_match(self, key_match: KeyMatch):
        self.key_matches.append(key_match)

    def get_entity(self, side: Side) -> IRI:
        """The entity of the pair on this side."""
        return self.source_entity if side == Side.SOURCE else self.target_entity

    def levels(self) -> list[NormalizationLevel]:
        """The levels on which the pair shares a key, from fine to coarse."""
        return sort_normalization_levels(key_match.level for key_match in self.key_matches)

    def finest_normalization_level(self) -> NormalizationLevel:
        return self.levels()[0]  # a pair has at least one key match

    def role_pairs(self) -> set[tuple[str, str]]:
        """(source role, target role) of all values that matched, e.g. ("names", "synonyms")."""
        return {
            (source_row.role_name, target_row.role_name)
            for key_match in self.key_matches
            for source_row in key_match.source_rows
            for target_row in key_match.target_rows
        }

    def entity_groups(self) -> dict[Side, set[str]]:
        """The groups of both entities, by side (every row carries one group, and
        an entity in several groups has a row for each)."""
        group_names_by_side: dict[Side, set[str]] = {Side.SOURCE: set(), Side.TARGET: set()}
        for key_match in self.key_matches:
            group_names_by_side[Side.SOURCE].update(row.group_name for row in key_match.source_rows)
            group_names_by_side[Side.TARGET].update(row.group_name for row in key_match.target_rows)
        return group_names_by_side


@dataclass
class OverfullKey:
    """
    A key shared by too many entities to pair them (above max_bucket_entities).
    """
    level: NormalizationLevel
    key: str
    num_source_entities: int
    num_target_entities: int


@dataclass(frozen=True)
class KeyCandidate:
    """
    A candidate of the key strategy, seen from one side: the entity on `side`
    gets the entity on the other side as candidate. A pair gives two candidates
    (one per side) when both entities are in their population; they differ in
    rivals_of_candidate_by_level.
    """
    side: Side                                       # side of the entity whose candidate this is
    entity_key_pair: EntityKeyPair                   # levels, keys, rows of both entities
    # finest level on which both entities have no other competing counterpart;
    # None: ambiguous on every level of the pair
    finest_unique_level: NormalizationLevel | None
    # per level of the pair: competing entities of the entity's own side that also
    # match the candidate there (the candidate's other counterparts)
    rivals_of_candidate_by_level: dict[NormalizationLevel, int]

    @property
    def own_entity(self) -> IRI:
        return self.entity_key_pair.get_entity(self.side)

    @property
    def candidate_entity(self) -> IRI:
        return self.entity_key_pair.get_entity(Side.other_side(self.side))

    @property
    def rivals_of_candidate_on_finest_level(self) -> int:
        return self.rivals_of_candidate_by_level[
            self.entity_key_pair.finest_normalization_level()
            ]


# later also NeighbourCandidate, EmbeddingCandidate
type Candidate = KeyCandidate


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
    # finest level of the pair -> competing counterparts outside the population
    # (each counted once); they take places in the tie groups of the selection
    rivals_outside_by_level: Counter[NormalizationLevel] = field(default_factory=Counter)
    # partition -> counterparts whose partition does not compete with the entity's
    non_competing: Counter[str] = field(default_factory=Counter)
