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
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from grasp.tasks.om.data_models.candidate_generators import Generator
from grasp.tasks.om.data_models.entity_pairs import EntityCandidates, Candidate, Hub
from grasp.tasks.om.foundation.types import IRI, EntityKey, Side


# how a selected run takes part in the merge: generate (its candidates enter
# the lists, its ranks form the Pareto layers) or corroborate (it only orders
# the candidates within a layer)
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

