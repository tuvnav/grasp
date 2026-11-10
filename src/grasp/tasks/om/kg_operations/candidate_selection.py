"""
Selection of the collected candidates (kg_operations/value_comparison.py)
by the `Selection` settings, and its summary for the LLM. Every function returns
new values and changes nothing it gets, so a selection can be repeated with
other settings on the same collection.

Order (select_<...>_candidates):
    1. per entity, on both sides: whole tie groups that fit into top_k
    2. mutual: candidates that have the entity in their own selection too
    3. per entity: the unique candidate (the anchor rule: alone in front, seen
       from both sides), then the optional filters selection.unique and
       selection.mutual
"""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import Iterable, Literal

from grasp.tasks.om.data_models.candidate_generators import CandidateStrategyType, Generator, Selection
from grasp.tasks.om.data_models.candidate_selection import RunRole, SelectedCandidates, StrategyResult, StrategySignals
from grasp.tasks.om.data_models.entity_pairs import Candidate, EntityCandidates
from grasp.tasks.om.foundation.text_vectors import Vectorizer
from grasp.tasks.om.foundation.types import IRI, SIDES, Side

# TODO: move to central place and use as central keying
# method for indexing Entities>
type EntityKey = tuple[Side, IRI]

# entity -> candidate IRI -> run id -> rank of the candidate in that run (1 = first tie group)
type CandidateRankIndex = dict[EntityKey, dict[IRI, dict[str, int]]]

type VennIndex = dict[frozenset[str], tuple[EntityKey, EntityKey]]


def select_tie_rank_groups(
        entity_candidates: EntityCandidates,
        top_k: int,
        ) -> tuple[list[list[Candidate]], float | None]:
    """
    The whole tie groups that fit into top_k, best first, and the rank of a tie
    group that was dropped although places were free: top_k would have cut it in
    the middle (None: no such group; a group that only finds no free place left
    is an ordinary cut by top_k). A
    tie rank group: the candidates whose pair has the same best rank. The rivals
    outside the population on that rank take places too, but are not listed.
    """
    candidates_by_rank: dict[float, list[Candidate]] = defaultdict(list)
    for candidate in entity_candidates.candidates:
        candidates_by_rank[candidate.rank_key].append(candidate)
    rivals_by_rank = entity_candidates.num_rivals_outside_by_rank
    ranks = sorted(set(candidates_by_rank) | set(rivals_by_rank))

    tie_rank_groups: list[list[Candidate]] = []
    places_used = 0
    for rank in ranks:
        tie_rank_group = candidates_by_rank.get(rank, [])
        rank_group_size = len(tie_rank_group) + rivals_by_rank[rank]
        if places_used + rank_group_size > top_k:
            # this rank group and all worse ones are dropped; ambiguous only if places were free
            return tie_rank_groups, (rank if places_used < top_k else None)
        if tie_rank_group:
            tie_rank_groups.append(tie_rank_group)
        places_used += rank_group_size
    return tie_rank_groups, None


def find_mutual_candidates(
        tie_groups_by_entity: dict[EntityKey, list[list[Candidate]]]
        ) -> dict[EntityKey, frozenset[IRI]]:
    """
    For each entity: its selected candidates whose own selection contains the
    entity. Needs the selections of both sides.
    """
    entities_selected_by_entity = {
        entity_key: {
            candidate.candidate_entity for tie_group in tie_groups
            for candidate in tie_group
            }
        for entity_key, tie_groups in tie_groups_by_entity.items()
    }
    mutual_by_entity = {}
    for (side, entity), selected in entities_selected_by_entity.items():
        other_side = Side.other_side(side)
        mutual_by_entity[(side, entity)] = frozenset(
            candidate_entity for candidate_entity in selected
            if entity in entities_selected_by_entity.get((other_side, candidate_entity), set())
            )
    return mutual_by_entity


def find_alone_candidate(
        entity_candidates: EntityCandidates,
        tie_groups: list[list[Candidate]],
        ) -> IRI | None:
    """
    One direction of the anchor rule: the candidate that stands alone in front,
    i.e. the first tie group holds only it and no rival outside the population
    has its rank or a better one. None otherwise.
    """
    if not tie_groups or len(tie_groups[0]) != 1:
        return None
    candidate = tie_groups[0][0]
    rival_at_this_rank_or_better = any(
        count > 0 and rank <= candidate.rank_key
        for rank, count in entity_candidates.num_rivals_outside_by_rank.items()
        )
    return None if rival_at_this_rank_or_better else candidate.candidate_entity


def filter_tie_groups(
        tie_groups: list[list[Candidate]],
        keep: frozenset[IRI],
        ) -> list[list[Candidate]]:
    """
    The tie groups with only the candidates in 'keep'; empty groups are left out.
    """
    filtered = [
        [candidate for candidate in tie_group
         if candidate.candidate_entity in keep]
        for tie_group in tie_groups
        ]
    return [tie_group for tie_group in filtered if tie_group]


def select_candidates(
        candidates_by_entity: dict[EntityKey, EntityCandidates],
        selection: Selection,
        ) -> dict[EntityKey, SelectedCandidates]:
    """
    The selection of every entity of both sides (see the module docstring).
    """
    # 1. whole tie groups that fit into top_k, on both sides
    tie_groups_by_entity: dict[EntityKey, list[list[Candidate]]] = {}
    dropped_rank_by_entity: dict[EntityKey, float | None] = {}
    for entity_key, entity_candidates in candidates_by_entity.items():
        tie_groups_by_entity[entity_key], dropped_rank_by_entity[entity_key] = select_tie_rank_groups(
            entity_candidates, selection.top_k)

    # 2. mutual, from the selections of both sides
    mutual_by_entity = find_mutual_candidates(tie_groups_by_entity)

    # 3. unique candidate: alone in front from both sides (the anchor rule; implies mutual)
    alone_by_entity = {
        entity_key: find_alone_candidate(
            entity_candidates, tie_groups_by_entity[entity_key]
            )
        for entity_key, entity_candidates in candidates_by_entity.items()
    }
    selected_by_entity = {}
    for entity_key, entity_candidates in candidates_by_entity.items():
        tie_groups, mutual_candidates = tie_groups_by_entity[entity_key], mutual_by_entity[entity_key]
        side, entity = entity_key
        alone_candidate = alone_by_entity[entity_key]
        unique_candidate = (alone_candidate if alone_candidate is not None
                            and alone_by_entity.get((Side.other_side(side), alone_candidate)) == entity else None)
        # then the optional filters
        if selection.unique:
            tie_groups = filter_tie_groups(tie_groups, frozenset({unique_candidate} if unique_candidate else ()))
        elif selection.mutual:
            tie_groups = filter_tie_groups(tie_groups, mutual_candidates)
        selected_by_entity[entity_key] = SelectedCandidates(
            collected=entity_candidates,
            tie_rank_groups=tie_groups,
            dropped_tie_group_rank=dropped_rank_by_entity[entity_key],
            mutual_candidates=mutual_candidates,
            unique_candidate=unique_candidate,
        )
    return selected_by_entity


def summarize_candidate_selection(
        selected_by_entity: dict[EntityKey, SelectedCandidates],
        population_sizes: dict[Side, int],
        sampled_side: Side | None = None,
        ) -> dict[Side, StrategySignals]:
    """
    Four signals per side. population_sizes: all entities of each side's
    population, also those without any match (they are not in selected_by_entity);
    for a sampled side the sample size. sampled_side: see StrategySignals.
    """
    signals = {}
    for side in SIDES:
        entries = [selected for (entry_side, _), selected in selected_by_entity.items() if entry_side == side]
        list_lengths = [len(entry.selected_candidates) for entry in entries if entry.selected_candidates]
        population_size = population_sizes[side]
        signals[side] = StrategySignals(
            side=side,
            population_size=population_size,
            coverage=share(
                len(list_lengths), population_size
                ),
            mean_list_length=(
                sum(list_lengths) / len(list_lengths) if list_lengths else 0.0
                ),
            ambiguous_at_top_rank=share(
                sum(entry.collected.ambiguous_at_top_rank for entry in entries),
                population_size
                ),
            unique=share(
                sum(entry.unique_candidate is not None for entry in entries),
                population_size
                ),
            already_matched_at_top_rank=share(
                sum(entry.collected.already_matched_at_top_rank for entry in entries),
                population_size
                ),
            sampled_side=sampled_side,
        )
    return signals


def share(count: int, population_size: int) -> float:
    if population_size:
        return count / population_size
    return 0.0


@dataclass
class RunEnsemble:
    all_run_names: list[str]
    selected_run_names: list[str]
    belong_together: list[str]
    __run_names_to_ids: dict[str, int] = field(init=False)
    __ids_to_run_names: dict[int, str] = field(init=False)

    def __post_init__(self):
        names_to_ids, ids_to_names = translate_run_names_to_ints(
            self.selected_run_names
            )
        self.__run_names_to_ids = names_to_ids
        self.__ids_to_run_names = ids_to_names

    @property
    def run_names_to_ids(self) -> dict[str, int]:
        return dict(self.__run_names_to_ids)

    @property
    def ids_to_run_names(self) -> dict[int, str]:
        return dict(self.__ids_to_run_names)


def translate_run_names_to_ints(
    run_names: Iterable[str]
) -> tuple[dict[str, int], dict[int, str]]:
    run_name_to_int: dict[str, int] = {}
    int_to_run_name: dict[int, str] = {}

    seen = set()
    for id, run_name in enumerate(run_names):
        if run_name in seen:
            run_names = list(run_names)
            duplicates = set(run_name) | {
                rn for rn in run_names if run_names.count(rn) > 1
            }
            raise ValueError(
                f"Run names are not distinct: {list(duplicates)} "
                "appear multiple times in run names."
                )
        run_name_to_int[run_name] = id
        int_to_run_name[id] = run_name

    return run_name_to_int, int_to_run_name


def build_candidate_rank_index(
        run_results: dict[str, tuple[Generator, StrategyResult]],
        selected_run_ids: Iterable[str],
        ) -> CandidateRankIndex:
    """
    Per entity of both sides: its selected candidates in the selected runs,
    each with its rank in every run that selected it (1 = the run's first tie
    group). Built from the selections only: what a run did not select (top_k)
    has no rank.
    """
    candidate_rank_index: CandidateRankIndex = {}
    for run_id in selected_run_ids:
        _, result = run_results[run_id]
        for entity_key, selected_candidates in result.selected.items():
            for rank, tie_group in enumerate(
                    selected_candidates.tie_rank_groups, start=1
                    ):
                for candidate in tie_group:
                    candidate_rank_index.setdefault(
                        entity_key, {}
                        ).setdefault(
                        candidate.candidate_entity, {}
                        )[run_id] = rank
    return candidate_rank_index


def rank_tuple(
        ranks_by_run: dict[str, int], 
        run_ids: Sequence[str]
        ) -> tuple[float, ...]:
    """
    The ranks of a candidate in the runs run_ids, in this order;
    a run that did not select it gives inf.
    """
    return tuple(ranks_by_run.get(run_id, math.inf) for run_id in run_ids)


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """
    a is at least as good as b in every run and better in at least one
    (smaller ranks are better).
    """
    return all(a_rank <= b_rank for a_rank, b_rank in zip(a, b)) and a != b


def pareto_layers(
        rank_tuples_by_candidate: dict[IRI, tuple[float, ...]]
        ) -> list[list[IRI]]:
    """
    Non-dominated sorting: layer 1 holds the candidates no other candidate
    dominates; without them, layer 2 the next ones, and so on. Candidates of
    a layer are sorted by IRI (a fixed order, they are equal in rank).
    E.g. a (1, 2), b (2, 3), c (5, 1) -> [[a, c], [b]]: b is dominated by a,
    c by nobody.
    """
    remaining = dict(rank_tuples_by_candidate)
    layers: list[list[IRI]] = []
    while remaining:
        layer = sorted(
            candidate for candidate, own_ranks in remaining.items()
            if not any(
                dominates(other_ranks, own_ranks)
                for other_ranks in remaining.values()
                )
        )
        layers.append(layer)
        for candidate in layer:
            del remaining[candidate]
    return layers


def pareto_merge_entity_candidates(
        ranks_by_candidate: dict[IRI, dict[str, int]],
        generate_run_ids: Sequence[str],
        corroborate_run_ids: Sequence[str],
        ) -> list[list[IRI]]:
    """
    The merged candidates of one entity as layers, best first: the candidates
    of the generate runs in Pareto layers by their ranks there; within a layer
    ordered by their ranks in the corroborate runs (their own Pareto layers,
    one after another). Corroborate runs add no candidates and move none to
    another layer.
    """
    generated = {
        candidate: rank_tuple(ranks_by_run_id, generate_run_ids)
        for candidate, ranks_by_run_id in ranks_by_candidate.items()
        if any(run_id in ranks_by_run_id for run_id in generate_run_ids)
    }
    return [
        [
            candidate
            for sublayer in pareto_layers({
                candidate: rank_tuple(ranks_by_candidate[candidate], corroborate_run_ids)
                for candidate in layer
            })
            for candidate in sublayer
        ]
        for layer in pareto_layers(generated)
    ]


def pareto_merge_all_candidates(
        candidate_rank_index: CandidateRankIndex,
        roles_by_run: dict[str, RunRole],
        ) -> dict[EntityKey, list[list[IRI]]]:
    """
    The merged candidate layers of every entity (pareto_merge_entity_candidates).
    Entities whose candidates come only from corroborate runs get no entry.
    """
    # a fixed order of the runs: the positions in every rank tuple mean the same run
    generate_run_ids = sorted(run_id for run_id, role in roles_by_run.items() if role == "generate")
    corroborate_run_ids = sorted(run_id for run_id, role in roles_by_run.items() if role == "corroborate")
    merged: dict[EntityKey, list[list[IRI]]] = {}
    for entity_key, ranks_by_candidate in candidate_rank_index.items():
        layers = pareto_merge_entity_candidates(ranks_by_candidate, generate_run_ids, corroborate_run_ids)
        if layers:
            merged[entity_key] = layers
    return merged


def run_ids_to_run_families(
        run_ids: Iterable[str],
        belong_together: Iterable[set[str]]
        ) -> dict[str, str]:
    """
    Builds a dict mapping from run_id to the name of
    it's run family.
    """
    res: dict[str, str] = {}
    new_family_names = ...
    for run_id in run_ids:
        ...


def run_suffix(run_id: str) -> int | None:
    res = ""
    for char in reversed(run_id):
        if char.isdigit():
            res = char + res
        else:
            break
    if res == "":
        return None
    return int(res)
