"""
Selection of the collected key candidates (kg_operations/value_comparison.py)
by the Selection settings, and its summary for the LLM. Every function returns
new values and changes nothing it gets, so a selection can be repeated with
other settings on the same collection.

Order (select_key_candidates):
    1. per entity, on both sides: whole tie groups that fit into top_k
    2. mutual: candidates that have the entity in their own selection too
    3. per entity: the unique candidate (the anchor rule: alone in front, seen
       from both sides), then the optional filters selection.unique and
       selection.mutual
"""
from collections import defaultdict

from grasp.tasks.om.data_models.candidate_generators import Selection
from grasp.tasks.om.data_models.candidate_selection import KeyStrategySignals, SelectedCandidates
from grasp.tasks.om.data_models.entity_pairs import Candidate, EntityCandidates
from grasp.tasks.om.foundation.types import IRI, SIDES, Side

type EntityKey = tuple[Side, IRI]


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
    rivals_by_rank = entity_candidates.rivals_outside_by_rank
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
        for rank, count in entity_candidates.rivals_outside_by_rank.items())
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


def select_key_candidates(
        candidates_by_entity: dict[EntityKey, EntityCandidates],
        selection: Selection,
        ) -> dict[EntityKey, SelectedCandidates]:
    """The selection of every entity of both sides (see the module docstring)."""
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
        entity_key: find_alone_candidate(entity_candidates, tie_groups_by_entity[entity_key])
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
            tie_groups=tie_groups,
            dropped_tie_group_rank=dropped_rank_by_entity[entity_key],
            mutual_candidates=mutual_candidates,
            unique_candidate=unique_candidate,
        )
    return selected_by_entity


def summarize_key_selection(
        selected_by_entity: dict[EntityKey, SelectedCandidates],
        population_sizes: dict[Side, int],
        ) -> dict[Side, KeyStrategySignals]:
    """The four signals per side. population_sizes: all entities of each side's
    population, also those without any key match (they are not in selected_by_entity)."""
    signals = {}
    for side in SIDES:
        entries = [selected for (entry_side, _), selected in selected_by_entity.items() if entry_side == side]
        list_lengths = [len(entry.selected_candidates) for entry in entries if entry.selected_candidates]
        population_size = population_sizes[side]
        share = (lambda count: count / population_size) if population_size else (lambda count: 0.0)
        signals[side] = KeyStrategySignals(
            side=side,
            population_size=population_size,
            coverage=share(len(list_lengths)),
            mean_list_length=sum(list_lengths) / len(list_lengths) if list_lengths else 0.0,
            ambiguous_at_top_rank=share(sum(entry.collected.ambiguous_at_top_rank for entry in entries)),
            unique=share(sum(entry.unique_candidate is not None for entry in entries)),
        )
    return signals
