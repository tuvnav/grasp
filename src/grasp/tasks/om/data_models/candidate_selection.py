"""
The selection layer of the key strategy: what the Selection settings make of the
collected candidates (data_models/entity_pairs.py). Kept apart from the
collection, so that the selection can be run again with other settings (e.g.
another top_k) without retrieving and intersecting again.

Tie groups: candidates of an entity whose pairs have the same finest level are
tied, since the key strategy cannot tell them apart. A tie group is taken whole
or not at all, and no coarser group follows a dropped one. A group dropped
although places were free (it would be cut in the middle at top_k) makes the
entity "ambiguous"; a group that finds no free place is an ordinary cut.
"""
from dataclasses import dataclass

from grasp.tasks.om.data_models.entity_pairs import EntityCandidates, KeyCandidate, OverfullKey
from grasp.tasks.om.foundation.text_normalization import NormalizationLevel
from grasp.tasks.om.foundation.types import IRI, Side


@dataclass(frozen=True)
class SelectedCandidates:
    """The candidates of one entity after the selection: whole tie groups in rank order."""
    collected: EntityCandidates                  # what was found (unchanged)
    tie_groups: list[list[KeyCandidate]]         # rank of a group = its position (0 = best)
    # level of a tie group dropped although places were free (it would have been
    # cut in the middle at top_k): the entity is ambiguous there; None: not ambiguous
    dropped_tie_group_level: NormalizationLevel | None
    # candidates that have the entity in their own selection too (both sides selected first)
    mutual_candidates: frozenset[IRI]
    # the anchor rule: this candidate stands alone in front for the entity (first
    # tie group, no rivals outside the population on its level or finer) AND the
    # entity stands alone in front for it; implies mutual. None otherwise
    unique_candidate: IRI | None

    @property
    def selected_candidates(self) -> list[KeyCandidate]:
        """All selected candidates, best group first."""
        return [candidate for tie_group in self.tie_groups for candidate in tie_group]


@dataclass(frozen=True)
class KeyStrategySignals:
    """
    Summary of a key strategy's selection on one side, for the LLM. All shares
    are of the whole population of the side (entities without any key match
    count too: they are the ones the strategy does not cover).
    """
    side: Side
    population_size: int
    coverage: float          # share with at least one selected candidate
    mean_list_length: float  # selected candidates per entity, over those with at least one
    ambiguous: float         # share with a dropped tie group
    unique: float            # share with a unique candidate (the anchor rule)


@dataclass(frozen=True)
class KeyStrategyResult:
    """
    One run of a key strategy. The collection is kept with the selection, so the
    selection can be repeated with other settings (select_key_candidates on
    `collected`) without retrieving and intersecting again.
    """
    collected: dict[tuple[Side, IRI], EntityCandidates]   # what was found, per (side, entity)
    selected: dict[tuple[Side, IRI], SelectedCandidates]  # after the Selection settings
    signals: dict[Side, KeyStrategySignals]
    overfull_keys: list[OverfullKey]                      # keys too ambiguous to pair
