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
from collections.abc import Callable, Iterable, Sequence
import math
import random
from typing import Literal

from grasp.tasks.om.data_models.candidates import (
    Anchor, AnchorMerge, AnchorSpecificity, AnchorStageResult, CandidateRankIndex, FamilyRoundOutcome, CandidateStrategyKind, EnsembleEvaluation, Generator, KeyCandidate, KeyEntityPair, KeyMatch, KeyMatchSignal, KeyStrategy, Match, MatchSignal, MergedCandidates, NeighbourMatch, NeighbourStrategy, NeighboursMatchSignal, PairSignalIndex, ParetoIndex, RunEnsemble, RunSignalInfos, SelectedCandidates, Selection,
    EntityCandidates,
    Candidate, CandidateStrategy, StrategyResult, StrategySignals, Stratum, VectorMatch, VectorMatchSignal, VectorStrategy, VennIndex,

    )
from grasp.tasks.om.data_models.verification import (
    CertificationResult, GetCandidateLayersOf, GetCandidateSignalsOf, FocusEntity, JudgingUnit, ListedCandidate,
    UnitJudgment,
)
from grasp.tasks.om.foundation.text_normalization import EXTENDED_LEVELS, NormalizationLevel, normalization_level_order, words
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
    rivals_by_rank = entity_candidates.num_counterparts_outside_population_by_rank
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
        for rank, count in entity_candidates.num_counterparts_outside_population_by_rank.items()
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


def is_exact(signal: MatchSignal) -> bool:
    """
    Exact evidence: a key on a standard normalization level, equal texts of a
    vector match, a witness that no other entities share.
    """
    match signal:
        case KeyMatchSignal():
            return signal.level not in EXTENDED_LEVELS
        case VectorMatchSignal():
            return signal.texts[0] == signal.texts[1]
        case NeighboursMatchSignal():
            return signal.bucket_size == (1, 1)


def strategy_type(strategy: CandidateStrategy) -> str:
    """
    The strategy type of a run, finer than its kind: 'key', 'neighbours' or,
    for a vector run, its vectorizer ('idf_char_3grams', 'embedding'); types
    fail on different pairs.
    """
    return strategy.vectorizer if isinstance(strategy, VectorStrategy) else strategy.kind


def family_strength(
        run_ids: Sequence[str], signals_by_run: dict[str, list[MatchSignal]], strategy_type_by_run: dict[str, str]
        ) -> str:
    """
    How strongly the runs of one family found a pair: exact if one of them gave
    an exact match (is_exact); else several_types if runs of several strategy
    types found it (strategy_type, e.g. idf_char_3grams and embedding: they fail
    on different pairs), one_type if only one type did. STRENGTH_DESCRIPTIONS
    says them in words.
    """
    if any(is_exact(signal) for run_id in run_ids for signal in signals_by_run.get(run_id, [])):
        return "exact"
    if len({strategy_type_by_run[run_id] for run_id in run_ids}) > 1:
        return "several_types"
    return "one_type"


# the strengths of family_strength in words, for the LLM
STRENGTH_DESCRIPTIONS = {
    "exact": "exact matches",
    "several_types": "matched by 2+ strategy types",
    "one_type": "matched by 1 strategy type",
}


def build_venn_index(
    pareto_index: ParetoIndex,
    candidate_rank_index: CandidateRankIndex,
    run_ensemble: RunEnsemble,
    pair_signal_index: PairSignalIndex,
    strategy_type_by_run: dict[str, str],
) -> VennIndex:
    """
    The pairs of the cut lists by their Venn cell: the families of the runs that
    found them, each with its strength (family_strength), e.g.
    {'names:exact', 'neighbours over subsumption:one_type'}.
    """

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

        run_ids_by_family: dict[str, list[str]] = defaultdict(list)
        for run_id in all_evidence_runs:
            run_ids_by_family[run_ensemble.family(run_id)].append(run_id)
        signals_by_run = pair_signal_index.get((source_entity, target_entity), {})
        venn_cell = frozenset(
            f"{family}:{family_strength(run_ids, signals_by_run, strategy_type_by_run)}"
            for family, run_ids in run_ids_by_family.items()
            )

        venn_index.setdefault(venn_cell, []).append((source_entity, target_entity))

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
    the other side (the lists of this side then lack counterparts).
    """
    results = [strategy_results_by_run[run_id] for run_id in ensemble.run_ids()]
    other_side = Side.other_side(side)
    if any(result.signals[side].sampled_side == other_side for result in results):
        return None
    populations_and_compared: list[tuple[frozenset[IRI], frozenset[IRI]]] = []
    for result in results:
        populations_and_compared.append((result.population_minus_existing_equivs[side], result.compared_entities[side]))
    union_of_populations = frozenset().union(*(population for population, _ in populations_and_compared))
    return frozenset(
        entity for entity in union_of_populations
        if all(entity in compared for population, compared in populations_and_compared if entity in population)
        )


def merge_candidate_runs(
        run_ensemble: RunEnsemble,
        strategy_results_by_run: dict[str, StrategyResult],
        strategy_type_by_run: dict[str, str],
        budget: int,
        max_first_sublayer_size: int,
        ) -> MergedCandidates:
    """
    The candidate lists of the ensemble per entity: Pareto layers of the generate
    runs, sublayers by the corroborate runs, cut by the budget; the Venn index
    of the pairs in the cut lists (strategy_type_by_run: see strategy_type);
    and the entities on which the merge can be evaluated.
    """
    candidate_rank_index = build_candidate_rank_index(run_ensemble, strategy_results_by_run)
    uncut_pareto_index = build_pareto_index(candidate_rank_index, run_ensemble)
    cut_pareto_index, too_ambiguous_entities = cut_pareto_layers_by_budget(
        uncut_pareto_index, budget, max_first_sublayer_size
        )
    pair_signal_index = build_pair_signal_index(run_ensemble, strategy_results_by_run)
    return MergedCandidates(
        run_ensemble=run_ensemble,
        candidate_rank_index=candidate_rank_index,
        uncut_pareto_layers_by_entity=uncut_pareto_index,
        pareto_layers_by_entity=cut_pareto_index,
        venn_index=build_venn_index(
            cut_pareto_index, candidate_rank_index, run_ensemble, pair_signal_index, strategy_type_by_run
            ),
        too_ambiguous_entities=frozenset(too_ambiguous_entities),
        evaluable_entities={
            side: evaluable_entities(run_ensemble, strategy_results_by_run, side) for side in SIDES
            },
        pair_signal_index=pair_signal_index
        )


# list length classes of the evaluation: (smallest, largest length, name)
LIST_LENGTH_CLASSES: list[tuple[int, float, str]] = [
    (1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"), (11, math.inf, "more than 10"),
]


def list_length_class(length: int) -> str:
    return next(
        name for smallest, largest, name in LIST_LENGTH_CLASSES
        if smallest <= length <= largest
        )


def flatten_layers(layers: list[list[list[IRI]]]) -> list[IRI]:
    """
    The candidates of an entity's layers in rank order; tied ones (within a
    sublayer) sorted, so that the order is the same in every run.
    """
    return [
        candidate for layer in layers for sublayer in layer
        for candidate in sorted(sublayer)
        ]


def only_best_candidate(layers: list[list[list[IRI]]]) -> IRI | None:
    """
    The candidate alone in the first sublayer of the first layer, if one.
    """
    if layers and layers[0] and len(layers[0][0]) == 1:
        return layers[0][0][0]
    return None


def to_source_target_pair(
        side: Side, entity: IRI, candidate: IRI
        ) -> tuple[IRI, IRI]:
    """
    (source IRI, target IRI) of an entity of the side and its candidate.
    """
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
            cell = cell_by_pair[to_source_target_pair(side, entity, candidate)]
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


def families_of_values(
        run_ensemble: RunEnsemble, runs: dict[str, tuple[Generator, StrategyResult]]
        ) -> dict[str, str]:
    """
    The family of each run by its values: runs on the same value projection
    share one, whatever their kind of comparison, named by the role names
    ('names', 'names~synonyms'), with 'neighbours over' for neighbour runs, or,
    for own patterns, after the first run on them.
    """
    family_by_own_values: dict[str, str] = {}
    family_by_run: dict[str, str] = {}
    for run_id in run_ensemble.run_ids():
        strategy = runs[run_id][0].strategy
        source, target = (strategy.values.by_side(side) for side in (Side.SOURCE, Side.TARGET))
        if isinstance(source, str) and isinstance(target, str):
            values = source if source == target else f"{source}~{target}"
        else:
            own_values = "~".join(
                value_source if isinstance(value_source, str) else value_source.model_dump_json()
                for value_source in (source, target)
                )
            values = family_by_own_values.setdefault(own_values, run_id)
        family_by_run[run_id] = (
            f"neighbours over {values}" if strategy.kind == CandidateStrategyKind.NEIGHBOURS else values
            )
    return family_by_run


def join_families(run_ensemble: RunEnsemble, runs: dict[str, tuple[Generator, StrategyResult]]) -> RunEnsemble:
    """
    The ensemble with the families of the candidate stage: runs on the same
    values are one family (families_of_values); runs on different values that
    the LLM gave one family name join theirs under that name (e.g. an own
    pattern and a preset role on mostly the same values). The LLM can join
    families, not split them: a name given only to runs on the same values
    splits nothing; it names their family if all its runs carry it (e.g. an
    own pattern, which has no name of its own).
    """
    family_by_run = families_of_values(run_ensemble, runs)
    for named_family in sorted({run.family for run in run_ensemble.runs if run.family}):
        joined = {family_by_run[run.run_id] for run in run_ensemble.runs if run.family == named_family}
        if len(joined) < 2:
            (value_family,) = joined
            if all(run.family == named_family for run in run_ensemble.runs
                   if family_by_run[run.run_id] == value_family):
                family_by_run = {
                    run_id: named_family if family == value_family else family
                    for run_id, family in family_by_run.items()
                    }
            continue
        family_by_run = {
            run_id: named_family if family in joined else family for run_id, family in family_by_run.items()
            }
    return run_ensemble.model_copy(update={"runs": [
        run.model_copy(update={"family": family_by_run[run.run_id]}) for run in run_ensemble.runs
        ]})


def build_verification_strata(
        merged_candidates: MergedCandidates
        ) -> list[Stratum]:
    return strata_of(merged_candidates.venn_index)


def families_by_pair(venn_index: VennIndex) -> dict[tuple[IRI, IRI], frozenset[str]]:
    """
    The Venn cell (run families) of each pair of the cut lists.
    """
    return {pair: families for families, pairs in venn_index.items() for pair in pairs}


def layers_within_budget(layers: list[list[IRI]], budget: int) -> list[list[IRI]]:
    """
    Whole layers in rank order while they fit into the budget; no worse layer
    follows one that did not fit. The first layer is always taken, cut to the
    budget if it is larger (its candidates are sorted by IRI).
    """
    taken: list[list[IRI]] = []
    size = 0
    for layer in layers:
        if taken and size + len(layer) > budget:
            break
        taken.append(layer[:budget - size])
        size += len(taken[-1])
    return taken


def build_judging_units(
        merged_candidates: MergedCandidates,
        get_candidate_layers_of: GetCandidateLayersOf,
        get_candidate_signals_of: GetCandidateSignalsOf,
        max_listed_candidates: int
        ) -> list[tuple[JudgingUnit, str]]:
    """
    The units of the candidate judging, each with its stratum (for the
    evaluation only): every mutual pair (each entity the other's only best
    candidate) with both lists, and every entity with a cut list of this
    round, on both sides, alone (so an entity of a mutual pair is judged three
    times: conflicts between the judgments are information). The list of a
    focus entity holds every candidate the strategies found so far, this and
    earlier stages (candidate_layers_of: the global Pareto layers over all
    runs), whole layers up to max_listed_candidates, each candidate with its
    layer and the evidence of every run that found it (evidence_of). The
    units and their strata come from this round's cut lists: stratum = kind/
    cell of the best pair/length class of the cut list, e.g.
    "source/key+vector/2-3".
    """
    layers_by_entity = merged_candidates.pareto_layers_by_entity
    venn_cell_by_pair = families_by_pair(merged_candidates.venn_index)

    def get_listed_candidates(side: Side, entity: IRI) -> list[ListedCandidate]:
        """
        The global layers within the budget, each candidate with its evidence.
        """
        signales_by_candidate = get_candidate_signals_of(side, entity)
        return [
            ListedCandidate(
                entity=candidate,
                pareto_layer=layer_number,
                signals=signales_by_candidate.get(candidate, [])
            )
            for layer_number, layer in enumerate(
                layers_within_budget(
                    get_candidate_layers_of(side, entity),
                    max_listed_candidates
                    ),
                start=1
                )
            for candidate in layer
            ]

    def focus_entity(side: Side, entity: IRI) -> FocusEntity:
        return FocusEntity(
            side=side,
            entity=entity,
            candidates=get_listed_candidates(side, entity)
            )

    def venn_cell_name(side: Side, entity: IRI, candidate: IRI) -> str:
        return "+".join(sorted(venn_cell_by_pair.get(
            to_source_target_pair(side, entity, candidate), frozenset()
            )))

    judging_units: list[tuple[JudgingUnit, str]] = []

    for (side, entity), layers in sorted(layers_by_entity.items()):
        candidates = flatten_layers(layers)

        if not candidates:
            continue

        judging_units.append((
            JudgingUnit(
                focus=[focus_entity(side, entity)]),
            f"{Side(side).value}/{venn_cell_name(side, entity, candidates[0])}/{list_length_class(len(candidates))}"
            ))

        best_candidate = only_best_candidate(layers)

        if Side(side) == Side.SOURCE and best_candidate is not None:

            partner_layers = layers_by_entity.get((Side.TARGET, best_candidate), [])
            if only_best_candidate(partner_layers) == entity:
                judging_units.append((
                    JudgingUnit(
                        focus=[
                            focus_entity(Side.SOURCE, entity),
                            focus_entity(Side.TARGET, best_candidate)
                            ]),
                    f"mutual/{venn_cell_name(side, entity, best_candidate)}"
                    ))
    return judging_units


# --- anchors: the unique pairs of key runs ------------------------------------------

# roles whose values are names: only their anchors are head or specific
NAME_ROLES = frozenset({"names", "synonyms"})


def lexical_information_content(extensions: int, num_wordsets: int) -> float:
    """
    Seco et al. (2004): 1 - log(extensions + 1) / log(N); 1: 
    no other word set extends it.
    """
    if num_wordsets <= 1:
        return 1.0
    return 1 - math.log(extensions + 1) / math.log(num_wordsets)


def extension_counter(wordsets: frozenset[frozenset[str]]) -> Callable[[frozenset[str]], int]:
    """
    Counts the word sets that properly extend a given one
    ({paper} by {poster, paper}).
    """
    wordsets_by_word: dict[str, set[frozenset[str]]] = defaultdict(set)
    for wordset in wordsets:
        for word in wordset:
            wordsets_by_word[word].add(wordset)

    def count(wordset: frozenset[str]) -> int:
        if not wordset or any(word not in wordsets_by_word for word in wordset):
            return 0
        containing = set.intersection(*(wordsets_by_word[word] for word in wordset))
        return sum(len(other) > len(wordset) for other in containing)

    return count


def compares_names(generator: Generator) -> bool:
    """
    The key strategy compares names (roles names, synonyms) on both sides.
    """
    strategy = generator.strategy
    if not isinstance(strategy, KeyStrategy):
        return False
    return all(
        isinstance(value_source := strategy.values.by_side(side), str)
        and value_source in NAME_ROLES
        for side in SIDES
        )


def anchor_wordset(
        entity_pair: KeyEntityPair, level: NormalizationLevel
        ) -> frozenset[str]:
    """The word set of an anchor's name: the largest among the values behind its key on that level."""
    rows = [row for match in entity_pair.matches if match.level == level
            for row in (*match.source_rows, *match.target_rows)]
    return max((frozenset(words(row.raw_projection_value)) for row in rows), key=len, default=frozenset())


def unique_key_candidates(
        result: StrategyResult
        ) -> dict[tuple[IRI, IRI], KeyCandidate]:
    """The unique pairs of a key run (the anchor rule), with the source entity's candidate."""
    pairs: dict[tuple[IRI, IRI], KeyCandidate] = {}
    for (side, entity), selected in result.selected.items():
        if side != Side.SOURCE or selected.unique_candidate is None:
            continue
        candidate = next(candidate for candidate in selected.collected.candidates
                         if candidate.candidate_entity == selected.unique_candidate)
        if isinstance(candidate, KeyCandidate):
            pairs[(entity, selected.unique_candidate)] = candidate
    return pairs


# family, level, specificity, IC
type RunAnchor = tuple[str, NormalizationLevel, AnchorSpecificity, float | None]


def run_anchors(family: str, generator: Generator, result: StrategyResult) -> dict[tuple[IRI, IRI], RunAnchor]:
    """
    The anchors of one key run with their level,
    specificity and information content.
    """
    counters = None
    if compares_names(generator) and result.value_wordsets is not None:
        counters = {side: (extension_counter(result.value_wordsets[side]), len(result.value_wordsets[side]))
                    for side in SIDES}
    anchors: dict[tuple[IRI, IRI], RunAnchor] = {}
    for pair, candidate in unique_key_candidates(result).items():
        level = candidate.finest_unique_level or candidate.entity_pair.finest_normalization_level()
        if counters is None:
            anchors[pair] = (family, level, AnchorSpecificity.NOT_APPLICABLE, None)
            continue
        wordset = anchor_wordset(candidate.entity_pair, level)
        extensions = {side: counters[side][0](wordset) for side in SIDES}
        specificity = AnchorSpecificity.HEAD if any(extensions.values()) else AnchorSpecificity.SPECIFIC
        information_content = min(lexical_information_content(extensions[side], counters[side][1]) for side in SIDES)
        anchors[pair] = (family, level, specificity, information_content)
    return anchors


def merge_run_anchors(pair: tuple[IRI, IRI], found: list[RunAnchor]) -> Anchor:
    """
    One anchor from the runs that found it: the finest level; head if one run
    sees a head, specific if one sees it specific, else not applicable.
    """
    specificities = {specificity for _, _, specificity, _ in found}
    information_contents = [ic for _, _, _, ic in found if ic is not None]
    if AnchorSpecificity.HEAD in specificities:
        specificity = AnchorSpecificity.HEAD
    elif AnchorSpecificity.SPECIFIC in specificities:
        specificity = AnchorSpecificity.SPECIFIC
    else:
        specificity = AnchorSpecificity.NOT_APPLICABLE
    return Anchor(
        source_entity=pair[0], target_entity=pair[1],
        families=frozenset(family for family, _, _, _ in found),
        normalization_level=min((level for _, level, _, _ in found), key=normalization_level_order),
        specificity=specificity,
        information_content=min(information_contents) if information_contents else None,
    )


def merge_anchors(run_ensemble: RunEnsemble, runs: dict[str, tuple[Generator, StrategyResult]]) -> AnchorMerge:
    """
    The anchors of an ensemble of key runs: the unique pairs of every run,
    merged; pairs whose entity has another partner in another run are conflicts.
    """
    family_by_run = {run.run_id: run.family or run.run_id for run in run_ensemble.runs}
    found: dict[tuple[IRI, IRI], list[RunAnchor]] = defaultdict(list)
    for run_id in run_ensemble.run_ids():
        generator, result = runs[run_id]
        for pair, run_anchor in run_anchors(family_by_run[run_id], generator, result).items():
            found[pair].append(run_anchor)

    partners: dict[EntityKey, set[IRI]] = defaultdict(set)
    for source_entity, target_entity in found:
        partners[(Side.SOURCE, source_entity)].add(target_entity)
        partners[(Side.TARGET, target_entity)].add(source_entity)

    anchors, conflicts = [], []
    for pair, run_anchors_of_pair in sorted(found.items()):
        if len(partners[(Side.SOURCE, pair[0])]) > 1 or len(partners[(Side.TARGET, pair[1])]) > 1:
            conflicts.append((*pair, frozenset(family for family, _, _, _ in run_anchors_of_pair)))
        else:
            anchors.append(merge_run_anchors(pair, run_anchors_of_pair))
    return AnchorMerge(anchors=anchors, conflicts=conflicts)


def anchor_stratum_label(anchor: Anchor) -> str:
    """
    head or specific (names only), and extended for a
    coarse level; empty: neither.
    """
    parts = [] if anchor.specificity == AnchorSpecificity.NOT_APPLICABLE else [anchor.specificity.value]
    if anchor.extended:
        parts.append("extended")
    return "-".join(parts)


def get_anchor_strata(anchor_merge: AnchorMerge) -> list[Stratum]:
    """
    The anchors in certification strata: by the families that found them, and
    by head/specific and extended.
    """
    pairs_by_stratum: dict[tuple[frozenset[str], str], list[tuple[IRI, IRI]]] = defaultdict(list)
    for anchor in anchor_merge.anchors:
        pairs_by_stratum[(anchor.families, anchor_stratum_label(anchor))].append(anchor.pair)
    return [
        Stratum(families=families, pairs=sorted(pairs), label=label)
        for (families, label), pairs in sorted(pairs_by_stratum.items(), key=lambda item: (sorted(item[0][0]), item[0][1]))
        ]


def match_signal(match: Match) -> MatchSignal:
    """
    The raw evidence of one match, without what its run says for every pair.
    """
    match match:
        case KeyMatch():
            return KeyMatchSignal(
                key=match.key,
                level=match.level,
                # each value once (an entity may hold the same value under several roles)
                values=(list(dict.fromkeys(row.raw_projection_value for row in match.source_rows)),
                        list(dict.fromkeys(row.raw_projection_value for row in match.target_rows)))
                )
        case VectorMatch():
            return VectorMatchSignal(
                texts=(match.source_text, match.target_text), cosine=round(match.cosine, 3)
                )
        case NeighbourMatch():
            return NeighboursMatchSignal(
                witness=match.witness,
                relations=match.relations,
                bucket_size=(match.num_source_entities, match.num_target_entities))
        case _:
            raise NotImplementedError(
                f"match_signal does not support {type(match).__name__}"
                )


def build_pair_signal_index(
        ensemble: RunEnsemble, strategy_results_by_run: dict[str, StrategyResult]
        ) -> PairSignalIndex:
    """
    The matches of every pair the runs of the ensemble selected, per run; a
    pair selected from both sides is collected once.
    """
    index: PairSignalIndex = {}
    for run_id in ensemble.run_ids():
        for (side, entity), selected_candidates in strategy_results_by_run[run_id].selected.items():
            for candidate in selected_candidates.selected_candidates:
                pair = to_source_target_pair(side, entity, candidate.candidate_entity)
                if run_id not in index.setdefault(pair, {}):
                    index[pair][run_id] = [
                        match_signal(match)
                        for match in filter_informative_matches(candidate.entity_pair.matches)
                        ]
    return index


def filter_informative_matches(matches: Sequence[Match]) -> list[Match]:
    """
    The matches without those implied by a finer one: a key match whose value
    pairs all matched already on a finer normalization level says nothing new
    (what matches on 'case' also matches on 'words').
    """
    seen_value_pairs: set[tuple[str, str]] = set()
    informative: list[Match] = []
    for match in sorted(
            matches,
            key=lambda match: normalization_level_order(match.level)
            if isinstance(match, KeyMatch) else -1
            ):
        contained_value_pairs = {
            (source_row.raw_projection_value, target_row.raw_projection_value)
            for source_row in match.source_rows for target_row in match.target_rows
            }
        if isinstance(match, KeyMatch) and contained_value_pairs <= seen_value_pairs:
            continue
        seen_value_pairs |= contained_value_pairs
        informative.append(match)
    return informative


def run_signal_infos(
        stage: Literal["anchors", "candidates"],
        round_number: int,
        run_id: str,
        generator: Generator
        ) -> RunSignalInfos:
    strategy = generator.strategy
    return RunSignalInfos(
        stage=stage, round=round_number, run_id=run_id, kind=strategy.kind,
        values_projection=strategy.values,
        normalization=strategy.normalization if isinstance(strategy, KeyStrategy) else None,
        vectorizer=strategy.vectorizer if isinstance(strategy, VectorStrategy) else None,
        witness_weighting=strategy.witness_weighting if isinstance(strategy, NeighbourStrategy) else None)


# --- the strategy history: per family and round, what it contributed and how it fared ---

def new_outcome(family: str, strategy_types: set[str], stage: str, round_number: int) -> FamilyRoundOutcome:
    return FamilyRoundOutcome(
        family=family, strategy_types=sorted(strategy_types), stage=stage, round=round_number,
        pairs_by_strength={}, pairs_of_this_family_only=0, checked=0, rejected=0,
        )


def add_pairs(outcome: FamilyRoundOutcome, strength: str, number: int, of_this_family_only: bool) -> None:
    outcome.pairs_by_strength[strength] = outcome.pairs_by_strength.get(strength, 0) + number
    if of_this_family_only:
        outcome.pairs_of_this_family_only += number


def sorted_by_pairs(outcomes: Iterable[FamilyRoundOutcome]) -> list[FamilyRoundOutcome]:
    return sorted(outcomes, key=lambda outcome: (-sum(outcome.pairs_by_strength.values()), outcome.family))


def anchor_round_outcomes(
        round_number: int, anchor_stage_result: AnchorStageResult, certification_result: CertificationResult
        ) -> list[FamilyRoundOutcome]:
    """
    Per strategy family of an anchor round: the anchors of its certification
    strata (all exact, but on a coarse normalization level: one_type), those
    of strata it alone found, and the verdicts the stop rule counted. The
    families follow the rule of the candidate stage (join_families: one per
    value source), whatever the anchor stage named; a stratum of several
    families counts for each.
    """
    ensemble = anchor_stage_result.run_ensemble
    if ensemble is None:
        return []
    generators = {run_id: (tried_run.generator, None) for run_id, tried_run in anchor_stage_result.runs.items()}
    joined = join_families(ensemble, generators)
    value_family_by_anchor_family = {run.family_name: joined.family(run.run_id) for run in ensemble.runs}
    outcomes: dict[str, FamilyRoundOutcome] = {}
    for family in set(value_family_by_anchor_family.values()):
        strategy_types = {strategy_type(generators[run.run_id][0].strategy) for run in joined.runs if run.family == family}
        outcomes[family] = new_outcome(family, strategy_types, "anchors", round_number)
    for stratum in certification_result.strata:
        decision = stratum.decision
        anchor_families, _, label = decision.stratum.partition("/")
        families = {value_family_by_anchor_family[name] for name in anchor_families.split("+")}
        strength = "one_type" if "extended" in label else "exact"
        for family in families:
            outcome = outcomes[family]
            add_pairs(outcome, strength, decision.size, len(families) == 1)
            outcome.checked += decision.counted
            outcome.rejected += decision.rejects
    return sorted_by_pairs(outcomes.values())


def candidate_round_outcomes(
        round_number: int, merged_candidates: MergedCandidates,
        strategy_type_by_run: dict[str, str], unit_judgments: Sequence[UnitJudgment]
        ) -> list[FamilyRoundOutcome]:
    """
    Per strategy family of a candidate round: the pairs of the cut lists by
    their strength in the family (the Venn cells), those it alone found, the
    pairs of a judged unit's lists (checked) and of them those no judge set
    (rejected). A pair of several families counts for each.
    """
    ensemble = merged_candidates.run_ensemble
    outcomes = {
        family: new_outcome(
            family, {strategy_type_by_run[run.run_id] for run in ensemble.runs if run.family_name == family},
            "candidates", round_number)
        for family in {run.family_name for run in ensemble.runs}
        }
    venn_cell_by_pair = families_by_pair(merged_candidates.venn_index)
    accepted = {tuple(pair) for judgment in unit_judgments for pair in judgment.equivalences}
    checked = {
        pair
        for judgment in unit_judgments if judgment.error is None
        for focus in judgment.unit.focus
        for candidate in focus.candidates
        if (pair := to_source_target_pair(Side(focus.side), focus.entity, candidate.entity)) in venn_cell_by_pair
        }
    for pair, venn_cell in venn_cell_by_pair.items():
        families = {element.rsplit(":", 1)[0] for element in venn_cell}
        for element in venn_cell:
            family, strength = element.rsplit(":", 1)
            outcome = outcomes[family]
            add_pairs(outcome, strength, 1, len(families) == 1)
            if pair in checked:
                outcome.checked += 1
                outcome.rejected += pair not in accepted
    return sorted_by_pairs(outcomes.values())
