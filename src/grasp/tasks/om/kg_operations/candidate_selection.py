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
from grasp.tasks.om.data_models.entity_pairs import EntityCandidates, KeyCandidate
from grasp.tasks.om.foundation.text_normalization import (
    NormalizationLevel, normalization_level_order, sort_normalization_levels,
)
from grasp.tasks.om.foundation.types import IRI, SIDES, Side

type EntityKey = tuple[Side, IRI]


def select_tie_groups(
        entity_candidates: EntityCandidates,
        top_k: int,
        ) -> tuple[list[list[KeyCandidate]], NormalizationLevel | None]:
    """
    The whole tie groups that fit into top_k, best first, and the level of a tie
    group that was dropped although places were free: it would have been cut in
    the middle at top_k, so the entity is ambiguous there (None: no such group;
    a group that only finds no free place left is an ordinary cut by top_k). A
    tie group: the candidates whose pair has the same finest level. The rivals
    outside the population on that level take places too, but are not listed.
    """
    candidates_by_level: dict[NormalizationLevel, list[KeyCandidate]] = defaultdict(list)
    for candidate in entity_candidates.candidates:
        candidates_by_level[candidate.entity_key_pair.finest_normalization_level()].append(candidate)
    rivals_by_level = entity_candidates.rivals_outside_by_level
    levels = sort_normalization_levels(set(candidates_by_level) | set(rivals_by_level))

    tie_groups: list[list[KeyCandidate]] = []
    places_used = 0
    for level in levels:
        tie_group = candidates_by_level.get(level, [])
        group_size = len(tie_group) + rivals_by_level[level]
        if places_used + group_size > top_k:
            # this group and all coarser ones are dropped; ambiguous only if places were free
            return tie_groups, (level if places_used < top_k else None)
        if tie_group:
            tie_groups.append(tie_group)
        places_used += group_size
    return tie_groups, None


def find_mutual_candidates(
        tie_groups_by_entity: dict[EntityKey, list[list[KeyCandidate]]]
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
        tie_groups: list[list[KeyCandidate]],
        ) -> IRI | None:
    """
    One direction of the anchor rule: the candidate that stands alone in front,
    i.e. the first tie group holds only it and no rival outside the population
    matches on its level or a finer one. None otherwise.
    """
    if not tie_groups or len(tie_groups[0]) != 1:
        return None
    candidate = tie_groups[0][0]
    candidate_level_order = normalization_level_order(candidate.entity_key_pair.finest_normalization_level())
    rival_at_this_level_or_finer = any(
        count > 0 and normalization_level_order(level) <= candidate_level_order
        for level, count in entity_candidates.rivals_outside_by_level.items())
    return None if rival_at_this_level_or_finer else candidate.candidate_entity


def filter_tie_groups(
        tie_groups: list[list[KeyCandidate]],
        keep: frozenset[IRI],
        ) -> list[list[KeyCandidate]]:
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
    tie_groups_by_entity: dict[EntityKey, list[list[KeyCandidate]]] = {}
    dropped_level_by_entity: dict[EntityKey, NormalizationLevel | None] = {}
    for entity_key, entity_candidates in candidates_by_entity.items():
        tie_groups_by_entity[entity_key], dropped_level_by_entity[entity_key] = select_tie_groups(
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
            dropped_tie_group_level=dropped_level_by_entity[entity_key],
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
            ambiguous=share(sum(entry.dropped_tie_group_level is not None for entry in entries)),
            unique=share(sum(entry.unique_candidate is not None for entry in entries)),
        )
    return signals
