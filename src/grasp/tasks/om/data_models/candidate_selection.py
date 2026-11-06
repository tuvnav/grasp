"""
The selection layer of the key strategy: what the Selection settings make of the
collected candidates (data_models/entity_pairs.py). Kept apart from the
collection, so that the selection can be run again with other settings (e.g.
another top_k) without retrieving and intersecting again.

Tie groups: candidates of an entity whose pairs have the same rank key are
tied, since the strategy cannot tell them apart. A tie group is taken whole or
not at all, and no worse group follows a dropped one. A group dropped although
places were free (top_k would cut it in the middle) is noted with its rank; a
group that finds no free place is an ordinary cut. An entity is "ambiguous" when
more than one entity stands on its best rank (EntityCandidates.ambiguous_at_top_rank),
independent of top_k.
"""
from collections.abc import Sequence
from dataclasses import dataclass

from grasp.tasks.om.data_models.entity_pairs import EntityCandidates, Candidate, Hub, KeyHub
from grasp.tasks.om.foundation.types import IRI, Side


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
    are of the whole population of the side (entities without any match
    count too: they are the ones the strategy does not cover).

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
    sampled_side: Side | None = None  # trial run: the side whose population was sampled


@dataclass(frozen=True)
class StrategyResult[H: Hub]:
    """
    One run of a candidate strategy. The collection is kept with the selection,
    so the selection can be repeated with other settings (select_candidates on
    `collected`) without retrieving and comparing again. H: the hub kind of the
    strategy (KeyHub, VectorHub).
    """
    collected: dict[tuple[Side, IRI], EntityCandidates]  # what was found, per (side, entity)
    selected: dict[tuple[Side, IRI], SelectedCandidates]  # after the Selection settings
    signals: dict[Side, StrategySignals]
    hubs: Sequence[H]  # keys/texts with too many entity pairs to pair them (Sequence: a list of KeyHub or of VectorHub)
