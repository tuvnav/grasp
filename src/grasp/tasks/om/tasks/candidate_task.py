"""
The candidate stage (in progress): candidates of a key or vector strategy for
the entities of a population, on both sides of a KG pair.

Per strategy (run_strategy):
    1. collect: per side the values of the entities in the groups -> indices;
       compare the two sides into entity pairs (get_key_candidates: intersect
       the indices per level; get_vector_candidates: nearest texts by cosine);
       per entity in its population: candidates, rivals, non-competing
       counterparts
    2. select (kg_operations/candidate_selection.py): whole tie groups within
       top_k, mutual, the unique candidate, optional filters
    3. summarize: the four signals per side, over the whole population
       (count_population)
Once per run (retrieve_partition_names): the partition of every entity, which
does not depend on the strategy.

Trial runs (run_strategy with trial=True), while the LLM tries strategies: if
the larger population is above the limit of the vectorizer
(TRIAL_ENTITY_LIMITS), only a random sample of it is compared, against the
whole other side (draw_trial_sample). Only one side is sampled: sampling both
would lose the true counterparts and distort every signal. A large KG is always
the sampled side, drawn per group at spread offsets (large_kg_sampling) in
proportion to the estimated group sizes, and queried for the sampled entities
only; two large KGs are not supported yet.
"""
import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from grasp.manager import KgManager
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidate_generators import CandidateStrategy, CandidateStrategyType, Generator, KeyStrategy, VectorStrategy
from grasp.tasks.om.data_models.candidate_selection import StrategyResult
from grasp.tasks.om.data_models.entity_pairs import EntityCandidates, KeyHub, VectorHub
from grasp.tasks.om.data_models.profile import GroupDefinition, KgProfile
from grasp.tasks.om.data_models.scope import REST, GroupRef, Population
from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.types import IRI, SIDES, Side
from grasp.tasks.om.kg_operations.candidate_selection import select_candidates, summarize_candidate_selection
from grasp.tasks.om.kg_operations.grouping import (
    Grouping, group_sizes, profile_groups_to_grouping, profile_partitions_to_grouping,
)
from grasp.tasks.om.kg_operations.large_kg_sampling import KgSample, use_sample
from grasp.tasks.om.kg_operations.value_comparison import (
    ValueIndex, build_side_indices, compare_value_indices_by_vectors, intersect_level_indices, key_entity_pairs_to_candidates, vector_entity_pairs_to_candidates,
)
from grasp.tasks.om.kg_operations.value_retrieval import (
    retrieve_entities, retrieve_partitions_by_entity, retrieve_value_rows,
)
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


def generate_group_grouping(kg_profile: KgProfile, population: Population, side: Side) -> Grouping:
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


def generate_population_groups_grouping(kg_profile: KgProfile, population: Population, side: Side) -> Grouping:
    """The groups of one side's population only (no rivals), without the entities
    of excluded groups; also for a large KG."""
    all_groups = profile_groups_to_grouping(kg_profile, inline_groups(population, side))
    group_names = population_group_names(kg_profile, population, side)
    return Grouping(
        tuple(member for member in all_groups.members if member[0] in group_names),
        exclusive=False, minus=all_groups.minus,
    )


def generate_population_grouping(kg_profile: KgProfile, population: Population, side: Side) -> Grouping:
    """The groups of one side's population merged into one (POPULATION), so that
    an entity in several of them counts once."""
    return generate_population_groups_grouping(kg_profile, population, side).merged(POPULATION)


# TODO: Move to central place?
def count_population(
        manager: KgManager,
        kg_profile: KgProfile,
        population: Population,
        side: Side
        ) -> int:
    """
    The entities of one side's population, each once (also those without any
    value of the strategy: the denominator of the signals).
    """
    return group_sizes(
        manager, generate_population_grouping(kg_profile, population, side), kg_profile.entity_space
        ).get(POPULATION, 0)


# trial runs: the most entities of the larger population that are compared, per
# vectorizer (in a small KG the key strategy is cheap and runs on the whole population)
TRIAL_ENTITY_LIMITS: dict[str, int] = {
    "idf_tokens": 20_000,
    "char_3grams": 20_000,
    "embedding": 5_000,
}
# a large KG is always sampled in a trial run, the key strategy with this limit
KEY_TRIAL_ENTITY_LIMIT_IN_LARGE_KG = 20_000


@dataclass(frozen=True)
class EntitySample:
    """The entities of one side's population that a trial run compares; the
    other side is compared whole. kg_sample: in a large KG, the sample of the
    membership patterns the entities come from; while it is in use
    (large_kg_sampling.use_sample) the queries over the groups of that KG see
    only these entities."""
    side: Side
    entities: frozenset[IRI]
    kg_sample: KgSample | None = field(default=None, compare=False)


def trial_entity_limit(strategy: CandidateStrategy, large: bool = False) -> int | None:
    """The trial limit of a strategy; None: no sampling. large: of a large KG,
    which is always sampled."""
    if isinstance(strategy, VectorStrategy):
        return TRIAL_ENTITY_LIMITS.get(strategy.vectorizer)
    return KEY_TRIAL_ENTITY_LIMIT_IN_LARGE_KG if large else None


def check_large_kgs(large_sides: set[Side], trial: bool) -> None:
    """A run with a large KG is only possible as a trial run, and only if the
    other KG is not large too (then the large side is sampled and the other one
    compared whole)."""
    if len(large_sides) == 2:
        raise FunctionCallException(
            "Both KGs are large: trial runs of candidate strategies are not supported yet "
            "(neither side can be compared whole)")
    if large_sides and not trial:
        raise FunctionCallException(
            "A large KG can only be compared in a trial run (on a sample) for now")


def choose_sampled_side(population_sizes: dict[Side, int], limit: int | None) -> Side | None:
    """The side to sample in a trial run of two KGs that are not large: the one
    with the larger population, if it is above the limit (the source on equal
    sizes); None: compare both whole."""
    larger_side = max(SIDES, key=lambda side: population_sizes[side])  # first maximum: the source
    if limit is None or population_sizes[larger_side] <= limit:
        return None
    return larger_side


def draw_entity_sample(entities: Iterable[IRI], size: int, seed: int) -> frozenset[IRI]:
    """A random sample of `size` entities, the same for the same entities and seed."""
    return frozenset(random.Random(seed).sample(sorted(entities), size))


def allocate_sample_sizes(estimated_sizes: dict[str, int | None], limit: int) -> dict[str, int]:
    """The members to draw per group, `limit` in all, in proportion to the
    estimated sizes of the groups (so that the shares over the population are not
    skewed towards small groups); at least one per group with members, at most
    its size. A group without an estimate (None) gets an equal share."""
    group_count = len(estimated_sizes)
    if group_count == 0:
        return {}
    known_total = sum(size for size in estimated_sizes.values() if size)
    sizes: dict[str, int] = {}
    for name, size in estimated_sizes.items():
        if size is None:
            sizes[name] = max(1, limit // group_count)
        elif size == 0 or known_total == 0:
            sizes[name] = 0
        else:
            sizes[name] = min(size, max(1, round(limit * size / known_total)))
    return sizes


def draw_large_kg_sample(manager: KgManager, kg_profile: KgProfile, generator: Generator, side: Side,
                         limit: int) -> EntitySample:
    """The sample of the population of a large KG: per group, members drawn at
    spread offsets (large_kg_sampling; not random, but cheap), as many as its
    share of the estimated population size; assigned to their groups in Python."""
    grouping = generate_population_groups_grouping(kg_profile, generator.population, side)
    space = tuple(kg_profile.entity_space)
    estimated_sizes = KgSample(manager, space).estimated_sizes(grouping)
    sample_sizes = allocate_sample_sizes({name: estimated_sizes.get(name) for name in grouping.names}, limit)
    # a name with several patterns: each pattern draws the size of the name
    kg_sample = KgSample(manager, space, size_by_membership={
        membership.strip(): sample_sizes[name] for name, membership in grouping.members})
    entities = frozenset(entity for entity, _ in kg_sample.assignments(grouping))
    return EntitySample(side=side, entities=entities, kg_sample=kg_sample)


def draw_trial_sample(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        population_sizes: dict[Side, int],
        seed: int,
        ) -> EntitySample | None:
    """The sample of a trial run (see the module docstring); None: no sampling
    needed. A large KG is always the sampled side (its population is not counted:
    population_sizes holds only the sides that are not large); seed: only for KGs
    that are not large."""
    large_side = next((side for side in SIDES if alignment_state.get_profile(side).large), None)
    if large_side is not None:
        limit = trial_entity_limit(generator.strategy, large=True)
        assert limit is not None
        return draw_large_kg_sample(
            managers[large_side], alignment_state.get_profile(large_side), generator, large_side, limit)
    limit = trial_entity_limit(generator.strategy)
    side = choose_sampled_side(population_sizes, limit)
    if side is None or limit is None:
        return None
    kg_profile = alignment_state.get_profile(side)
    entities = retrieve_entities(
        managers[side], generate_population_grouping(kg_profile, generator.population, side),
        kg_profile.entity_space)
    return EntitySample(side=side, entities=draw_entity_sample(entities, min(limit, len(entities)), seed))


def retrieve_strategy_value_rows(
        manager: KgManager,
        kg_profile: KgProfile,
        generator: Generator,
        side: Side,
        entity_sample: EntitySample | None = None,
        ) -> Iterator[ValueRow]:
    """The values a key or vector strategy compares on one side: of the entities
    of all groups (rivals outside the population included); on a sampled side
    only those of the sampled entities (in a large KG: of the groups of the
    population, queried for the sampled entities only)."""
    strategy = generator.strategy
    assert isinstance(strategy, KeyStrategy | VectorStrategy)
    side_sample = entity_sample if entity_sample is not None and entity_sample.side == side else None
    if side_sample is not None and side_sample.kg_sample is not None:
        use_sample(manager, side_sample.kg_sample)
        try:
            value_rows = list(retrieve_value_rows(
                manager,
                generate_population_groups_grouping(kg_profile, generator.population, side),
                kg_profile.entity_space,
                strategy.values.side(side),
                kg_profile.roles,
            ))
        finally:
            use_sample(manager, None)
        return iter(value_rows)
    value_rows = retrieve_value_rows(
        manager,
        generate_group_grouping(kg_profile, generator.population, side),
        kg_profile.entity_space,
        strategy.values.side(side),
        kg_profile.roles,
    )
    if side_sample is None:
        return value_rows
    return (row for row in value_rows if row.entity in side_sample.entities)


def sampled_partition_names(kg_profile: KgProfile, entity_sample: EntitySample) -> dict[IRI, str]:
    """The partition of every sampled entity of a large KG, asked for these
    entities only (the first matching partition, as in a small KG)."""
    assert entity_sample.kg_sample is not None
    assignments = entity_sample.kg_sample.assignments(
        profile_partitions_to_grouping(kg_profile), sorted(entity_sample.entities))
    return {entity: partition for entity, partition in assignments}


def retrieve_partition_names(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        ) -> dict[Side, dict[IRI, str]]:
    """The partition of every entity, per side. Depends only on the profiles, so it
    is retrieved once per run and reused by every strategy. A large KG is left
    empty here: a trial run asks for the partitions of its sampled entities only."""
    partition_names_by_entity = {}
    for side in SIDES:
        kg_profile = alignment_state.get_profile(side)
        if kg_profile.large:
            partition_names_by_entity[side] = {}
            continue
        partition_names_by_entity[side] = retrieve_partitions_by_entity(
            managers[side], kg_profile, kg_profile.entity_space)
    return partition_names_by_entity


def get_key_candidates(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        max_entity_pairs_per_bucket: int | None = None,
        entity_sample: EntitySample | None = None,
        ) -> tuple[dict[tuple[Side, IRI], EntityCandidates], list[KeyHub]]:
    """
    The candidates of a key strategy for every entity in its population, on both
    sides, and the keys whose entities would give too many entity pairs
    (above max_entity_pairs_per_bucket). entity_sample: a trial run compares
    only these entities on their side.
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
        group_names_by_side[side] = population_group_names(
            kg_profile, generator.population, side
            )
        value_rows = retrieve_strategy_value_rows(managers[side], kg_profile, generator, side, entity_sample)
        indices_by_side[side] = build_side_indices(value_rows, strategy.normalization)

    # 2. intersect the two sides: entity pairs with their shared keys
    entity_pairs, overfull_keys = intersect_level_indices(
        indices_by_side[Side.SOURCE], indices_by_side[Side.TARGET], max_entity_pairs_per_bucket)

    # 3. per entity in its population: candidates, rivals, non-competing counterparts
    candidates_by_entity = key_entity_pairs_to_candidates(
        alignment_state, entity_pairs, group_names_by_side, partition_names_by_entity)
    return candidates_by_entity, overfull_keys


def get_vector_candidates(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        max_entity_pairs_per_bucket: int | None = None,
        neighbours_per_text: int = 50,
        entity_sample: EntitySample | None = None,
        ) -> tuple[dict[tuple[Side, IRI], EntityCandidates], list[VectorHub]]:
    """
    The candidates of a vector strategy for every entity in its population, on
    both sides, and the text pairs whose entities would give too many entity
    pairs (above max_entity_pairs_per_bucket). The strategy compares the texts
    of ONE normalization level. neighbours_per_text: how many nearest texts of
    the other side each text keeps (a pre-cut before the selection, larger than
    any top_k). entity_sample: a trial run compares only these entities on their
    side. Not yet ranked or cut by generator.selection.
    """
    strategy = generator.strategy
    if not isinstance(strategy, VectorStrategy):
        raise FunctionCallException(f"get_vector_candidates needs a vector strategy, not {strategy.kind!r}")

    # 1. per side: the values of the entities in the groups -> index of the one level
    index_by_side: dict[Side, ValueIndex] = {}
    group_names_by_side = {}
    for side in SIDES:
        kg_profile = alignment_state.get_profile(side)
        group_names_by_side[side] = population_group_names(kg_profile, generator.population, side)
        value_rows = retrieve_strategy_value_rows(managers[side], kg_profile, generator, side, entity_sample)
        # exactly one level: the unpacking fails otherwise
        (index_by_side[side],) = build_side_indices(value_rows, strategy.normalization).values()

    # 2. vectorize, build similarity matrix, determine a preliminary
    # number of best neighbours per text on both sides (not top-k yet,
    # but a bigger pool to draw top-k from. Top-k ist selected only
    # on 'selection' level)
    entity_pairs, vector_hubs = compare_value_indices_by_vectors(
        index_by_side[Side.SOURCE],
        index_by_side[Side.TARGET],
        strategy.vectorizer,
        strategy.model_name,
        max_entity_pairs_per_bucket,
        neighbours_per_text
    )

    # 3. per entity in its population: candidates, rivals
    candidates_by_entity = vector_entity_pairs_to_candidates(
        alignment_state, entity_pairs, group_names_by_side, partition_names_by_entity
        )
    return candidates_by_entity, vector_hubs


def run_strategy(
    managers: dict[Side, KgManager],
    alignment_state: AlignmentState,
    generator: Generator,
    partition_names_by_entity: dict[Side, dict[IRI, str]],
    max_entity_pairs_per_bucket: int | None = None,
    trial: bool = False,
    sample_seed: int = 0,
        ) -> StrategyResult:
    """
    A key or vector strategy from the KGs to the selected candidates and the
    signals. trial: a trial run, which may compare only a sample of the larger
    population (see the module docstring); sample_seed: the seed of the sample,
    so that trials of different strategies see the same entities. A large KG
    only in a trial run, and only against a KG that is not large.
    """
    # 0. the populations, and in a trial run the sample of the larger one (a
    # large KG is not counted: it is always the sampled side)
    large_sides = {side for side in SIDES if alignment_state.get_profile(side).large}
    check_large_kgs(large_sides, trial)
    population_sizes = {
        side: count_population(
            managers[side],
            alignment_state.get_profile(side),
            generator.population,
            side
            )
        for side in SIDES if side not in large_sides
    }
    entity_sample = (
        draw_trial_sample(managers, alignment_state, generator, population_sizes, sample_seed)
        if trial else None
        )
    if entity_sample is not None and entity_sample.kg_sample is not None:
        partition_names_by_entity = {
            **partition_names_by_entity,
            entity_sample.side: sampled_partition_names(
                alignment_state.get_profile(entity_sample.side), entity_sample),
        }

    # 1. collect
    match generator.strategy.kind:
        case CandidateStrategyType.KEY:
            collected, hubs = get_key_candidates(
                managers, alignment_state, generator,
                partition_names_by_entity, max_entity_pairs_per_bucket,
                entity_sample=entity_sample,
            )
        case CandidateStrategyType.VECTOR:
            collected, hubs = get_vector_candidates(
                managers, alignment_state, generator,
                partition_names_by_entity, max_entity_pairs_per_bucket,
                entity_sample=entity_sample,
            )
        case _:
            raise NotImplementedError(
                f"run_strategy does not support the {generator.strategy.kind!r} strategy yet"
                )

    # 2. select
    selected = select_candidates(collected, generator.selection)

    # 3. summarize over the whole population of each side (on a sampled side: the sample)
    sampled_side = entity_sample.side if entity_sample is not None else None
    if entity_sample is not None:
        population_sizes[entity_sample.side] = len(entity_sample.entities)
    signals = summarize_candidate_selection(selected, population_sizes, sampled_side)

    return StrategyResult(
        collected=collected,
        selected=selected,
        signals=signals,
        hubs=hubs
        )
