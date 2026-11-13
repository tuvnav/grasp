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
from collections import Counter, defaultdict
from collections.abc import Sequence
import math
import random

from grasp.tasks.om.data_models.candidate_generators import Selection
from grasp.tasks.om.data_models.candidate_selection import CandidateRankIndex, EnsembleEvaluation, MergedCandidates, ParetoIndex, RunEnsemble, SelectedCandidates, StrategyResult, StrategySignals, Stratum, VennIndex
from grasp.tasks.om.data_models.entity_pairs import Candidate, EntityCandidates
from grasp.tasks.om.foundation.types import IRI, SIDES, EntityKey, Side


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


def build_candidate_rank_index(
        ensemble: RunEnsemble,
        strategy_results_by_run: dict[str, StrategyResult],
        ) -> CandidateRankIndex:
    """
    Per entity of both sides: its selected candidates in the runs of the
    ensemble, each with its rank in every run that selected it (1 = the run's
    first tie group). Built from the selections only: what a run did not
    select (top_k) has no rank.
    """
    candidate_rank_index: CandidateRankIndex = {}
    for run_id in ensemble.run_ids():
        for entity_key, selected in strategy_results_by_run[run_id].selected.items():
            for rank, tie_group in enumerate(selected.tie_rank_groups, start=1):
                for candidate in tie_group:
                    candidate_rank_index.setdefault(entity_key, {}).setdefault(
                        candidate.candidate_entity, {})[run_id] = rank
    return candidate_rank_index


def generate_rank_tuple(
        ranks_by_run: dict[str, int],
        relevant_run_ids: Sequence[str]
        ) -> tuple[float, ...]:
    """
    The ranks of a candidate in the runs run_ids, in this order; a run that
    did not select it gives inf.
    """
    return tuple(ranks_by_run.get(run_id, math.inf) for run_id in relevant_run_ids)


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """
    a is at least as good as b in every run and better in at least one
    (smaller ranks are better).
    """
    return all(
        a_element <= b_element for a_element, b_element in zip(a, b)
        ) and a != b


def build_pareto_layers(
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
            if not any(dominates(other_ranks, own_ranks) for other_ranks in remaining.values())
        )
        layers.append(layer)
        for candidate in layer:
            del remaining[candidate]
    return layers


def pareto_merge_entity_candidates(
        ranks_by_candidate: dict[IRI, dict[str, int]],
        generating_run_ids: Sequence[str],
        corroborating_run_ids: Sequence[str],
        ) -> list[list[list[IRI]]]:
    """
    The merged candidates of one entity as layers, best first: the candidates
    of the generate runs in Pareto layers by their ranks there; within a layer
    sublayers by ranks in the corroborate runs. Corroborate runs add no
    candidates and move none to another layer.
    """
    layer_tuples = {
        candidate: generate_rank_tuple(ranks_by_run, generating_run_ids)
        for candidate, ranks_by_run in ranks_by_candidate.items()
        if any(run_id in generating_run_ids for run_id in ranks_by_run)
    }
    pareto_layers = build_pareto_layers(layer_tuples)

    res: list[list[list[IRI]]] = []
    for layer in pareto_layers:
        sublayer_tuples = {
            candidate: generate_rank_tuple(
                ranks_by_candidate[candidate], corroborating_run_ids
                )
            for candidate in layer
            }
        res.append(build_pareto_layers(sublayer_tuples))

    return res


def build_pareto_index(
        candidate_rank_index: CandidateRankIndex,
        ensemble: RunEnsemble,
        ) -> ParetoIndex:
    """
    The merged candidate layers of every entity (pareto_merge_entity_candidates).
    Entities whose candidates come only from corroborate runs get no entry.
    """
    generate_run_ids = ensemble.run_ids("generate")
    corroborate_run_ids = ensemble.run_ids("corroborate")
    merged: ParetoIndex = {}
    for entity_key, ranks_by_candidate in candidate_rank_index.items():
        layers = pareto_merge_entity_candidates(ranks_by_candidate, generate_run_ids, corroborate_run_ids)
        if layers:
            merged[entity_key] = layers
    return merged


def cut_pareto_layers_by_budget(
        pareto_index: ParetoIndex,
        budget: int,
        max_first_sublayer_size: int
) -> tuple[ParetoIndex, set[EntityKey]]:
    """
    Per entity: whole sublayers in rank order while they fit into the budget;
    no worse sublayer follows one that did not fit. The first sublayer is
    always taken, up to max_first_sublayer_size; an entity whose first sublayer
    is larger gets no list and is returned as too ambiguous.
    """
    new_index: ParetoIndex = {}
    too_ambiguous_entities: set[EntityKey] = set()

    for entity_key, pareto_layers in pareto_index.items():
        # avoid index error
        if not pareto_layers or not pareto_layers[0]:
            continue

        if len(pareto_layers[0][0]) > max_first_sublayer_size:
            too_ambiguous_entities.add(entity_key)
            continue

        new_layers: list[list[list[IRI]]] = []
        remaining_budget = budget
        out_of_budget = False
        is_first_sublayer = True

        for layer in pareto_layers:
            new_layer: list[list[IRI]] = []
            for sublayer in layer:
                if not is_first_sublayer and len(sublayer) > remaining_budget:
                    out_of_budget = True
                    break

                new_layer.append(sublayer)
                remaining_budget -= len(sublayer)
                is_first_sublayer = False

            if new_layer:
                new_layers.append(new_layer)

            if out_of_budget:
                break

        new_index[entity_key] = new_layers

    return new_index, too_ambiguous_entities


def build_venn_index(
    pareto_index: ParetoIndex,
    candidate_rank_index: CandidateRankIndex,
    run_ensemble: RunEnsemble
) -> VennIndex:

    venn_index: VennIndex = {}
    entity_pairs: set[tuple[IRI, IRI]] = set()

    for (side, entity), pareto_layers in pareto_index.items():
        for layer in pareto_layers:
            for sublayer in layer:
                for candidate_entity in sublayer:
                    entity_pairs.add(
                        (entity, candidate_entity) if side == Side.SOURCE
                        else (candidate_entity, entity)
                    )

    for (source_entity, target_entity) in sorted(entity_pairs):
        runs_finding_source_to_target = candidate_rank_index.get(
            (Side.SOURCE, source_entity), {}
            ).get(target_entity, {}).keys()
        runs_finding_target_to_source = candidate_rank_index.get(
            (Side.TARGET, target_entity), {}
            ).get(source_entity, {}).keys()
        all_evidence_runs = runs_finding_source_to_target | runs_finding_target_to_source

        run_families = frozenset(run_ensemble.family(run) for run in all_evidence_runs)

        venn_index.setdefault(run_families, []).append((source_entity, target_entity))

    return venn_index


def evaluable_entities(
        ensemble: RunEnsemble,
        strategy_results_by_run: dict[str, StrategyResult],
        side: Side,
        ) -> frozenset[IRI] | None:
    """
    The entities of one side on which the merge of the (trial) runs equals the
    merge of full runs: those that every run whose population holds them has
    compared. An entity a run did not draw into its sample would miss that
    run's candidates. Of full runs: all entities of the populations.
    None: the side is not measurable, because a run compared only a sample of
    the other side (the lists of this side then lack counterparts) or the side
    is a large KG (no population retrieved).
    """
    results = [strategy_results_by_run[run_id] for run_id in ensemble.run_ids()]
    other_side = Side.other_side(side)
    if any(result.signals[side].sampled_side == other_side for result in results):
        return None
    populations_and_compared: list[tuple[frozenset[IRI], frozenset[IRI]]] = []
    for result in results:
        population = result.population[side]
        if population is None:
            return None
        populations_and_compared.append((population, result.compared_entities[side]))
    union_of_populations = frozenset().union(*(population for population, _ in populations_and_compared))
    return frozenset(
        entity for entity in union_of_populations
        if all(entity in compared for population, compared in populations_and_compared if entity in population)
        )


def merge_runs(
        run_ensemble: RunEnsemble,
        strategy_results_by_run: dict[str, StrategyResult],
        budget: int,
        max_first_sublayer_size: int,
        ) -> MergedCandidates:
    """
    The candidate lists of the ensemble per entity: Pareto layers of the generate
    runs, sublayers by the corroborate runs, cut by the budget; the Venn index
    of the pairs in the cut lists; and the entities on which the merge can be
    evaluated.
    """
    candidate_rank_index = build_candidate_rank_index(run_ensemble, strategy_results_by_run)
    uncut_pareto_index = build_pareto_index(candidate_rank_index, run_ensemble)
    cut_pareto_index, too_ambiguous_entities = cut_pareto_layers_by_budget(
        uncut_pareto_index, budget, max_first_sublayer_size
        )
    return MergedCandidates(
        run_ensemble=run_ensemble,
        candidate_rank_index=candidate_rank_index,
        uncut_pareto_layers_by_entity=uncut_pareto_index,
        pareto_layers_by_entity=cut_pareto_index,
        venn_index=build_venn_index(cut_pareto_index, candidate_rank_index, run_ensemble),
        too_ambiguous_entities=frozenset(too_ambiguous_entities),
        evaluable_entities={
            side: evaluable_entities(run_ensemble, strategy_results_by_run, side) for side in SIDES
            },
        )


# list length classes of the evaluation: (smallest, largest length, name)
LIST_LENGTH_CLASSES: list[tuple[int, float, str]] = [
    (1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"), (11, math.inf, "more than 10"),
]


def list_length_class(length: int) -> str:
    return next(name for smallest, largest, name in LIST_LENGTH_CLASSES if smallest <= length <= largest)


def flatten_layers(layers: list[list[list[IRI]]]) -> list[IRI]:
    """The candidates of an entity's layers in rank order."""
    return [candidate for layer in layers for sublayer in layer for candidate in sublayer]


def to_entity_pair(side: Side, entity: IRI, candidate: IRI) -> tuple[IRI, IRI]:
    """(source IRI, target IRI) of an entity of the side and its candidate."""
    return (entity, candidate) if side == Side.SOURCE else (candidate, entity)


def evaluate_merged_candidates(
        merged: MergedCandidates,
        side: Side,
        num_cell_examples: int,
        num_too_ambiguous_examples: int,
        num_shown_tied_candidates: int,
        seed: int = 0,
        ) -> EnsembleEvaluation | None:
    """
    The merge on the evaluable entities of one side (None: not measurable):
    how many get a list, how long the lists are, what the budget cuts, the
    pairs of the lists per Venn cell (only pairs of evaluable entities: the
    whole Venn index also holds pairs of entities whose lists are incomplete),
    a random pair of each of the largest cells and random too ambiguous
    entities with the first candidates of their first sublayer.
    """
    evaluable = merged.evaluable_entities[side]
    if evaluable is None:
        return None
    cell_by_pair = {pair: cell for cell, pairs in merged.venn_index.items() for pair in pairs}
    pairs_by_cell: dict[frozenset[str], list[tuple[IRI, IRI]]] = {}
    list_length_counter: Counter[str] = Counter()
    too_ambiguous: list[IRI] = []
    num_with_list = 0
    num_cut_by_budget = 0
    for entity in sorted(evaluable):
        entity_key = (side, entity)
        if entity_key in merged.too_ambiguous_entities:
            too_ambiguous.append(entity)
            continue
        candidates = flatten_layers(merged.pareto_layers_by_entity.get(entity_key, []))
        if not candidates:
            continue
        num_with_list += 1
        list_length_counter[list_length_class(len(candidates))] += 1
        if len(candidates) < len(flatten_layers(merged.uncut_pareto_layers_by_entity[entity_key])):
            num_cut_by_budget += 1
        for candidate in candidates:
            cell = cell_by_pair[to_entity_pair(side, entity, candidate)]
            pairs_by_cell.setdefault(cell, []).append((entity, candidate))

    cells_largest_first = sorted(pairs_by_cell, key=lambda cell: (-len(pairs_by_cell[cell]), sorted(cell)))
    random_generator = random.Random(seed)
    cell_examples = [
        (cell, *random_generator.choice(pairs_by_cell[cell]))
        for cell in cells_largest_first[:num_cell_examples]
    ]
    too_ambiguous_examples = []
    for entity in random_generator.sample(too_ambiguous, min(num_too_ambiguous_examples, len(too_ambiguous))):
        first_sublayer = merged.uncut_pareto_layers_by_entity[(side, entity)][0][0]
        too_ambiguous_examples.append((entity, len(first_sublayer), first_sublayer[:num_shown_tied_candidates]))
    return EnsembleEvaluation(
        side=side,
        num_evaluated=len(evaluable),
        num_with_list=num_with_list,
        num_too_ambiguous=len(too_ambiguous),
        num_cut_by_budget=num_cut_by_budget,
        list_length_counts={
            name: list_length_counter[name] for _, _, name in LIST_LENGTH_CLASSES if list_length_counter[name]
        },
        pairs_by_cell=[(cell, len(pairs_by_cell[cell])) for cell in cells_largest_first],
        cell_examples=cell_examples,
        too_ambiguous_examples=too_ambiguous_examples,
    )


def strata_of(venn_index: VennIndex) -> list[Stratum]:
    """
    Output sorted by size of stratum.
    """
    return sorted([
        Stratum(
            families=families, pairs=pairs
            ) for families, pairs in venn_index.items()
        ],
        key=lambda stratum: len(stratum.pairs)
        )


def build_verification_strata(
        merged_candidates: MergedCandidates
        ) -> list[Stratum]:
    return strata_of(merged_candidates.venn_index)