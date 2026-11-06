"""
The candidate stage (in progress): candidates of a key strategy for the entities
of a population, on both sides of a KG pair.

Per strategy (run_key_strategy):
    1. collect (get_key_candidates): per side the values of the entities in the
       groups -> indices per level; intersect the two sides; per entity in its
       population: candidates, rivals, non-competing counterparts
    2. select (kg_operations/candidate_selection.py): whole tie groups within
       top_k, mutual, the unique candidate, optional filters
    3. summarize: the four signals per side, over the whole population
       (count_population)
Once per run (retrieve_partition_names): the partition of every entity, which
does not depend on the strategy.
"""
from grasp.manager import KgManager
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidate_generators import Generator, KeyStrategy
from grasp.tasks.om.data_models.candidate_selection import KeyStrategyResult
from grasp.tasks.om.data_models.entity_pairs import EntityCandidates, KeyHub
from grasp.tasks.om.data_models.profile import GroupDefinition, KgProfile
from grasp.tasks.om.data_models.scope import REST, GroupRef, Population
from grasp.tasks.om.foundation.types import IRI, SIDES, Side
from grasp.tasks.om.kg_operations.candidate_selection import select_key_candidates, summarize_key_selection
from grasp.tasks.om.kg_operations.grouping import Grouping, group_sizes, profile_groups_to_grouping
from grasp.tasks.om.kg_operations.value_comparison import (
    build_side_indices, intersect_level_indices, key_entity_pairs_to_candidates,
)
from grasp.tasks.om.kg_operations.value_retrieval import retrieve_partitions_by_entity, retrieve_value_rows
from grasp.utils import FunctionCallException


def group_refs_of_side(population: Population, side: Side) -> list[GroupRef]:
    return population.source_group_refs if side == Side.SOURCE else population.target_group_refs


def inline_groups(population: Population, side: Side) -> list[GroupDefinition]:
    """The groups a strategy defines inline for one side (not in the KgProfile)."""
    return [group_ref for group_ref in group_refs_of_side(population, side) if isinstance(group_ref, GroupDefinition)]


def population_group_names(kg_profile: KgProfile, population: Population, side: Side) -> set[str]:
    """
    The names of the groups of one side's population. REST stands for all groups
    of the profile that are not excluded (for now; later: only the entities that
    have no correspondence yet). Raises FunctionCallException for a name that is
    neither a group of the profile nor defined inline, and for an inline group
    named like a group of the profile.
    """
    clashing = sorted(group.name for group in inline_groups(population, side)
                      if kg_profile.has_group(group.name))
    if clashing:
        raise FunctionCallException(
            f"Groups defined inline must have new names; already groups of {kg_profile.kg_name}: {clashing}")

    names: set[str] = set()
    for group_ref in group_refs_of_side(population, side):
        if isinstance(group_ref, GroupDefinition):
            names.add(group_ref.name)
        elif group_ref == REST:
            names |= set(kg_profile.group_names()) - set(kg_profile.excluded_group_names())
        elif kg_profile.has_group(group_ref):
            names.add(group_ref)
        else:
            raise FunctionCallException(
                f"{group_ref} is not a group of {kg_profile.kg_name}; groups: {kg_profile.group_names()}, "
                f"or {REST!r} for all of them, or define a group inline")
    return names


def get_group_grouping(kg_profile: KgProfile, population: Population, side: Side) -> Grouping:
    """
    The groups whose values are retrieved for one side: in a small KG all groups of
    the profile and the inline groups (all are needed to count rivals outside the
    population by group), without the entities of excluded groups.
    """
    if kg_profile.large:
        # large KG: only the groups of the population (not built yet)
        raise NotImplementedError("key candidates for large KGs: retrieve only the population's groups")
    return profile_groups_to_grouping(kg_profile, inline_groups(population, side))


# name of the whole population when its entities are counted once
POPULATION = "_population"


def count_population(manager: KgManager, kg_profile: KgProfile, population: Population, side: Side) -> int:
    """The entities of one side's population, each once (also those without any
    value of the strategy: the denominator of the signals)."""
    retrieved = get_group_grouping(kg_profile, population, side)
    group_names = population_group_names(kg_profile, population, side)
    population_grouping = Grouping(
        tuple(member for member in retrieved.members if member[0] in group_names),
        exclusive=False, minus=retrieved.minus,
    ).merged(POPULATION)
    return group_sizes(manager, population_grouping, kg_profile.entity_space).get(POPULATION, 0)


def retrieve_partition_names(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        ) -> dict[Side, dict[IRI, str]]:
    """The partition of every entity, per side. Depends only on the profiles, so it
    is retrieved once per run and reused by every strategy."""
    partition_names_by_entity = {}
    for side in SIDES:
        kg_profile = alignment_state.get_profile(side)
        partition_names_by_entity[side] = retrieve_partitions_by_entity(
            managers[side], kg_profile, kg_profile.entity_space)
    return partition_names_by_entity


def get_key_candidates(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        max_entity_pairs_per_bucket: int | None = None,
        ) -> tuple[dict[tuple[Side, IRI], EntityCandidates], list[KeyHub]]:
    """
    The candidates of a key strategy for every entity in its population, on both
    sides, and the keys whose entities would give too many entity pairs
    (above max_entity_pairs_per_bucket).
    Not yet ranked or cut by generator.selection.
    """
    strategy = generator.strategy
    if not isinstance(strategy, KeyStrategy):
        raise FunctionCallException(f"get_key_candidates needs a key strategy, not {strategy.kind!r}")

    # 1. per side: the values of the entities in the groups -> indices per level
    indices_by_side = {}
    group_names_by_side = {}
    for side in SIDES:
        kg_profile = alignment_state.get_profile(side)
        group_names_by_side[side] = population_group_names(kg_profile, generator.population, side)
        value_rows = retrieve_value_rows(
            managers[side],
            get_group_grouping(kg_profile, generator.population, side),
            kg_profile.entity_space,
            strategy.values.side(side),
            kg_profile.roles,
        )
        indices_by_side[side] = build_side_indices(value_rows, strategy.normalization)

    # 2. intersect the two sides: entity pairs with their shared keys
    entity_pairs, overfull_keys = intersect_level_indices(
        indices_by_side[Side.SOURCE], indices_by_side[Side.TARGET], max_entity_pairs_per_bucket)

    # 3. per entity in its population: candidates, rivals, non-competing counterparts
    candidates_by_entity = key_entity_pairs_to_candidates(
        alignment_state, entity_pairs, group_names_by_side, partition_names_by_entity)
    return candidates_by_entity, overfull_keys


def get_


def run_key_strategy(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        max_entity_pairs_per_bucket: int | None = None,
        ) -> KeyStrategyResult:
    """A key strategy from the KGs to the selected candidates and the signals."""
    # 1. collect
    collected, overfull_keys = get_key_candidates(
        managers, alignment_state, generator, partition_names_by_entity, max_entity_pairs_per_bucket)
    # 2. select
    selected = select_key_candidates(collected, generator.selection)
    # 3. summarize over the whole population of each side
    population_sizes = {
        side: count_population(managers[side], alignment_state.get_profile(side), generator.population, side)
        for side in SIDES
    }
    signals = summarize_key_selection(selected, population_sizes)
    return KeyStrategyResult(collected=collected, selected=selected, signals=signals, overfull_keys=overfull_keys)
