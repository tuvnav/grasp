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
from collections import Counter, defaultdict
import dataclasses
from enum import StrEnum, auto
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Iterable, Iterator, Sequence, get_args

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidate_generators import CandidateStrategy, CandidateStrategyType, Generator, KeyStrategy, NeighbourStrategy, Selection, VectorStrategy
from grasp.tasks.om.data_models.candidate_selection import (
    CandidateStageResult, SelectedCandidates, StrategyResult, StrategySignals, TriedRun,
)
from grasp.tasks.om.data_models.entity_pairs import (
    Candidate, EntityCandidates, Hub, KeyHub, KeyMatch, Match, NeighbourEntityPair, NeighbourHub, NeighbourMatch,
    VectorHub, VectorMatch,
)
from grasp.tasks.om.data_models.patterns import StructureRelationKind
from grasp.tasks.om.data_models.profile import GroupDefinition, KgProfile, describe_profile
from grasp.tasks.om.data_models.relations import EQUIVALENCE
from grasp.tasks.om.data_models.scope import REST, GroupRef, Population, ValuePatternSource
from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.configuration import load_task_instructions
from grasp.tasks.om.foundation.text_normalization import NORMALIZATION_LEVELS, TRANSFORM_OPERATIONS
from grasp.tasks.om.foundation.text_vectors import Vectorizer
from grasp.tasks.om.foundation.types import IRI, SIDES, Side
from grasp.tasks.om.kg_operations.candidate_selection import select_candidates, summarize_candidate_selection
from grasp.tasks.om.kg_operations.entity_rendering import EntityDisplay
from grasp.tasks.om.kg_operations.grouping import (
    Grouping, counts_by_group, group_sizes, profile_groups_to_grouping, profile_partitions_to_grouping,
)
from grasp.tasks.om.kg_operations.large_kg_sampling import KgSample, use_sample
from grasp.tasks.om.kg_operations.value_comparison import (
    NeighbourIndex, ValueIndex, WitnessesByEntity, build_side_indices, build_side_witness_index,
    build_witness_maps, compare_value_indices_by_vectors, compare_witness_indices, find_confirmed_equivalences,
    intersect_level_indices,
    key_entity_pairs_to_candidates, neighbour_entity_pairs_to_candidates, relation_keys_by_role_name,
    vector_entity_pairs_to_candidates,
)
from grasp.tasks.om.kg_operations.value_retrieval import (
    retrieve_entities, retrieve_partitions_by_entity, retrieve_value_rows,
)
from grasp.tasks.om.sparql_access.values import values_clause
from grasp.tasks.om.tasks.function_registry import FunctionRegistry
from grasp.tasks.om.tasks.prompting import Outline, OutlineItem, add_equivalence_condition, format_outline, load_prompt
from grasp.tasks.om.tasks.task_input import resolve_alignment_state, resolve_managers
from grasp.utils import FunctionCallException, format_section
from pydantic import ValidationError


# TODO: Perhaps move to retrieve_values.py
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
    """
    The groups of one side's population only 
    (no rivals), without the entities
    of excluded groups; also for a large KG.
    """
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
        side: Side,
        matched_entities: Collection[IRI] = (),
        ) -> int:
    """
    The entities of one side's population, each once (also those without any
    value of the strategy: the denominator of the signals), without the
    matched_entities (those with an equivalence: they get no candidates).
    """
    population_grouping = generate_population_grouping(kg_profile, population, side)
    size = group_sizes(manager, population_grouping, kg_profile.entity_space).get(POPULATION, 0)
    if not matched_entities:
        return size
    matched_in_population = counts_by_group(
        manager, population_grouping, kg_profile.entity_space, values_clause("entity", matched_entities)
        ).get(POPULATION, 0)
    return size - matched_in_population


# trial runs: the most entities of the larger population that are compared, per
# vectorizer (in a small KG the key strategy is cheap and runs on the whole population)
TRIAL_ENTITY_LIMITS: dict[str, int] = {
    "idf_tokens": 20_000,
    "char_3grams": 20_000,
    "embedding": 5_000,
}
# a large KG is always sampled in a trial run; the key and neighbour strategies
# (without a limit in a small KG) with this limit
LARGE_KG_TRIAL_ENTITY_LIMIT = 20_000


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
    return LARGE_KG_TRIAL_ENTITY_LIMIT if large else None


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
        witnesses_by_entity: WitnessesByEntity | None = None,
        ) -> EntitySample | None:
    """The sample of a trial run (see the module docstring); None: no sampling
    needed. A large KG is always the sampled side (its population is not counted:
    population_sizes holds only the sides that are not large); seed: only for KGs
    that are not large. Entities that already have an equivalence
    (witnesses_by_entity) are not drawn: they get no candidates."""
    matched_entities = {side: set(witnesses_by_entity[side]) if witnesses_by_entity else set() for side in SIDES}
    large_side = next((side for side in SIDES if alignment_state.get_profile(side).large), None)
    if large_side is not None:
        limit = trial_entity_limit(generator.strategy, large=True)
        assert limit is not None
        large_kg_sample = draw_large_kg_sample(
            managers[large_side], alignment_state.get_profile(large_side), generator, large_side, limit)
        return dataclasses.replace(large_kg_sample,
                                   entities=large_kg_sample.entities - matched_entities[large_side])
    limit = trial_entity_limit(generator.strategy)
    side = choose_sampled_side(population_sizes, limit)
    if side is None or limit is None:
        return None
    kg_profile = alignment_state.get_profile(side)
    entities = [entity for entity in retrieve_entities(
        managers[side], generate_population_grouping(kg_profile, generator.population, side),
        kg_profile.entity_space) if entity not in matched_entities[side]]
    return EntitySample(side=side, entities=draw_entity_sample(entities, min(limit, len(entities)), seed))


def check_value_pattern_source(value_pattern_source: ValuePatternSource, kg_profile: KgProfile) -> None:
    """Every name in the value source is a role or a structure relation of the
    profile (checked before querying, for a message the LLM can act on)."""
    if not isinstance(value_pattern_source, list):
        return
    for name in value_pattern_source:
        if isinstance(name, str):
            try:
                kg_profile.roles.predicate_pattern_by_name(name)
            except ValueError as error:
                raise FunctionCallException(f"{kg_profile.kg_name}: {error}") from error


def retrieve_strategy_value_rows(
        manager: KgManager,
        kg_profile: KgProfile,
        generator: Generator,
        side: Side,
        entity_sample: EntitySample | None = None,
        object_iris: Collection[IRI] | None = None,
        ) -> Iterator[ValueRow]:
    """
    The values a strategy compares on one side (its value_pattern_source: roles,
    or the structure relations of a neighbour strategy): of the entities of all
    groups (rivals outside the population included); on a sampled side only
    those of the sampled entities (in a large KG: of the groups of the
    population, queried for the sampled entities only). object_iris: only values
    among these (a neighbour strategy: the entities of the witnesses).
    """
    value_pattern_source = generator.strategy.value_pattern_source(side, kg_profile.roles)
    check_value_pattern_source(value_pattern_source, kg_profile)
    side_sample = entity_sample if entity_sample is not None and entity_sample.side == side else None
    if side_sample is not None and side_sample.kg_sample is not None:
        use_sample(manager, side_sample.kg_sample)
        try:
            value_rows = list(retrieve_value_rows(
                manager,
                generate_population_groups_grouping(kg_profile, generator.population, side),
                kg_profile.entity_space,
                value_pattern_source,
                kg_profile.roles,
                object_iris,
            ))
        finally:
            use_sample(manager, None)
        return iter(value_rows)

    value_rows = retrieve_value_rows(
        manager,
        generate_group_grouping(kg_profile, generator.population, side),
        kg_profile.entity_space,
        value_pattern_source,
        kg_profile.roles,
        object_iris,
    )
    if side_sample is None:
        return value_rows
    return (row for row in value_rows if row.entity in side_sample.entities)


def sampled_partition_names(kg_profile: KgProfile, entity_sample: EntitySample) -> dict[IRI, str]:
    """
    The partition of every sampled entity of a large KG, asked for these
    entities only (the first matching partition, as in a small KG).
    """
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
        ) -> tuple[dict[tuple[Side, IRI], EntityCandidates], list[KeyHub], list[tuple[IRI, IRI]]]:
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
        indices_by_side[Side.SOURCE],
        indices_by_side[Side.TARGET],
        max_entity_pairs_per_bucket
        )

    # 3. per entity in its population without an equivalence: candidates, rivals,
    # non-competing counterparts; and the equivalences found again
    witnesses_by_entity = build_witness_maps(alignment_state.correspondences)
    candidates_by_entity = key_entity_pairs_to_candidates(
        alignment_state, entity_pairs,
        group_names_by_side, partition_names_by_entity, witnesses_by_entity
        )
    return candidates_by_entity, overfull_keys, find_confirmed_equivalences(entity_pairs, witnesses_by_entity)


def get_vector_candidates(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        max_entity_pairs_per_bucket: int | None = None,
        neighbours_per_text: int = 50,
        entity_sample: EntitySample | None = None,
        ) -> tuple[dict[tuple[Side, IRI], EntityCandidates], list[VectorHub], list[tuple[IRI, IRI]]]:
    """
    The candidates of a vector strategy for every entity in its population, on
    both sides, and the text pairs whose entities would give too many entity
    pairs (above max_entity_pairs_per_bucket). The strategy compares the texts
    of one normalization level. neighbours_per_text: how many nearest texts of
    the other side each text keeps (a pre-cut before the selection, larger than
    any top_k). entity_sample: a trial run compares only these entities on their
    side. Not yet ranked or cut by generator.selection.
    """
    strategy = generator.strategy
    if not isinstance(strategy, VectorStrategy):
        raise FunctionCallException(
            f"get_vector_candidates needs a vector strategy, not {strategy.kind!r}"
            )

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

    # 3. per entity in its population without an equivalence: candidates, rivals,
    # non-competing counterparts; and the equivalences found again
    witnesses_by_entity = build_witness_maps(alignment_state.correspondences)
    candidates_by_entity = vector_entity_pairs_to_candidates(
        alignment_state, entity_pairs, group_names_by_side, partition_names_by_entity, witnesses_by_entity
        )
    return candidates_by_entity, vector_hubs, find_confirmed_equivalences(entity_pairs, witnesses_by_entity)


def get_neighbour_candidates(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        max_entity_pairs_per_bucket: int | None = None,
        entity_sample: EntitySample | None = None,
        ) -> tuple[dict[tuple[Side, IRI], EntityCandidates], list[NeighbourHub], list[tuple[IRI, IRI]]]:
    """
    The candidates of a neighbour strategy for every entity in its population,
    on both sides, and the witnesses whose entities would give too many entity
    pairs (above max_entity_pairs_per_bucket: they create no pairs, but count
    for the others). The witnesses are the equivalences of the alignment state,
    so the candidates change from round to round. entity_sample: a trial run
    compares only these entities on their side.
    Not yet ranked or cut by generator.selection.
    """
    strategy = generator.strategy
    if not isinstance(strategy, NeighbourStrategy):
        raise FunctionCallException(
            f"get_neighbour_candidates needs a neighbour strategy, not {strategy.kind!r}"
            )

    # 0. the witnesses: the equivalences found so far
    witnesses_by_entity = build_witness_maps(alignment_state.correspondences)
    if not witnesses_by_entity[Side.SOURCE]:
        raise FunctionCallException(
            "A neighbour strategy needs equivalences found before (its witnesses), "
            "but there are none yet: run a key or vector strategy first."
        )

    # 1. per side: the neighbours that take part in a witness -> witness index
    index_by_side: dict[Side, NeighbourIndex] = {}
    group_names_by_side = {}
    for side in SIDES:
        kg_profile = alignment_state.get_profile(side)
        group_names_by_side[side] = population_group_names(
            kg_profile, generator.population, side
            )
        value_rows = retrieve_strategy_value_rows(
            managers[side], kg_profile, generator,
            side, entity_sample, witnesses_by_entity[side].keys()
            )
        relation_keys = relation_keys_by_role_name(
            strategy.value_pattern_source(side, kg_profile.roles),
            kg_profile.roles
        )
        index_by_side[side] = build_side_witness_index(
            witnesses_by_entity[side],
            value_rows,
            relation_keys
        )

    # 2. compare the two sides: entity pairs with their shared witnesses (the
    # hubs create no pairs, but add their witnesses to the pairs of the others)
    entity_pairs, neighbour_hubs = compare_witness_indices(
        index_by_side[Side.SOURCE],
        index_by_side[Side.TARGET],
        max_entity_pairs_per_bucket,
        strategy.min_witnesses,
        strategy.witness_weighting,
    )

    # 3. per entity in its population without an equivalence: candidates, rivals,
    # non-competing counterparts; and the equivalences found again
    candidates_by_entity = neighbour_entity_pairs_to_candidates(
        alignment_state, entity_pairs, group_names_by_side, partition_names_by_entity, witnesses_by_entity
        )
    return candidates_by_entity, neighbour_hubs, find_confirmed_equivalences(entity_pairs, witnesses_by_entity)


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
    A key, vector or neighbour strategy from the KGs to the selected candidates
    and the signals. Entities that already have an equivalence get no
    candidates, but stay candidates of the others (marked candidate_already_matched). trial: a trial run, which may compare only a sample of the larger
    population (see the module docstring); sample_seed: the seed of the sample,
    so that trials of different strategies see the same entities. A large KG
    only in a trial run, and only against a KG that is not large.
    """
    # 0. the populations without the entities that already have an equivalence,
    # and in a trial run the sample of the larger one (a large KG is not counted:
    # it is always the sampled side)
    large_sides = {side for side in SIDES if alignment_state.get_profile(side).large}
    check_large_kgs(large_sides, trial)
    witnesses_by_entity = build_witness_maps(alignment_state.correspondences)
    population_sizes = {
        side: count_population(
            managers[side],
            alignment_state.get_profile(side),
            generator.population,
            side,
            witnesses_by_entity[side].keys(),
            )
        for side in SIDES if side not in large_sides
    }
    entity_sample = (
        draw_trial_sample(managers, alignment_state, generator, population_sizes, sample_seed,
                          witnesses_by_entity)
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
            collected, hubs, confirmed_equivalences = get_key_candidates(
                managers, alignment_state, generator,
                partition_names_by_entity, max_entity_pairs_per_bucket,
                entity_sample=entity_sample,
            )
        case CandidateStrategyType.VECTOR:
            collected, hubs, confirmed_equivalences = get_vector_candidates(
                managers, alignment_state, generator,
                partition_names_by_entity, max_entity_pairs_per_bucket,
                entity_sample=entity_sample,
            )
        case CandidateStrategyType.NEIGHBOURS:
            collected, hubs, confirmed_equivalences = get_neighbour_candidates(
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
        confirmed_equivalences=confirmed_equivalences,
        existing_equivalences=sum(len(witnesses) for witnesses in witnesses_by_entity[Side.SOURCE].values()),
        hubs=hubs
        )


@dataclass
class MergedCandidates:
    ...


@dataclass
class CandidateTaskState:
    kg_managers: dict[Side, KgManager]
    alignment_state: AlignmentState  # doesn't change over the task
    partition_names_by_entity: dict[Side, dict[IRI, str]] = field(default_factory=dict)
    runs: dict[str, tuple[Generator, StrategyResult]] = field(default_factory=dict)
    # how entities are shown, per side; keeps the names retrieved so far
    entity_displays: dict[Side, EntityDisplay] = field(default_factory=dict)
    selection: list[str] | None = None
    rationale: str | None = None
    merged: MergedCandidates | None = None

    def add_run(self, generator: Generator, result: StrategyResult) -> str:
        run_id = f"run-{len(self.runs)}"
        self.runs[run_id] = (generator, result)
        return run_id


class TaskFunctionName(StrEnum):
    TRY_STRATEGY = auto()
    RESELECT = auto()
    SHOW_CANDIDATES = auto()
    # COMPARE_RUNS = auto()
    FINISH = auto()


func_reg = FunctionRegistry()


# entity pairs a bucket (entities sharing a key, a text, a witness) may give at
# most (source entities x target entities); a larger bucket is a hub: it gives
# no pairs (neighbours: it only adds witnesses to pairs of other buckets).
# Provisional values, to be checked on the hub counts of the first runs
MAX_ENTITY_PAIRS_PER_BUCKET: dict[CandidateStrategyType, int] = {
    CandidateStrategyType.KEY: 1_000,
    CandidateStrategyType.VECTOR: 1_000,
    CandidateStrategyType.NEIGHBOURS: 100,
}


@func_reg.register(TaskFunctionName.TRY_STRATEGY)
def try_strategy(state: CandidateTaskState, fn_args: dict) -> str:
    try:
        generator = Generator(**fn_args.get("generator", {}))
    except ValidationError as error:
        raise FunctionCallException(f"Invalid generator: {error}") from error

    try:
        result = run_strategy(
            state.kg_managers,
            state.alignment_state,
            generator,
            state.partition_names_by_entity,
            MAX_ENTITY_PAIRS_PER_BUCKET[generator.strategy.kind],
            trial=True,
            )
    except (ValueError, NotImplementedError) as error:
        # e.g. an unknown group or role, a value source the strategy cannot use
        raise FunctionCallException(str(error)) from error

    run_id = state.add_run(generator, result)
    examples = choose_examples(result)
    prefetch_shown_entities(examples, result.hubs, state.entity_displays)
    return format_outline(
        [
            (
                f"Result {run_id}",
                format_strategy_result(result, examples, state.entity_displays)
                )
            ]
        )


def reselect_result(
        strategy_result: StrategyResult,
        selection: Selection
        ) -> StrategyResult:

    collected_candidates = strategy_result.collected
    old_signals = strategy_result.signals
    population_sizes: dict[Side, int] = {
        side: old_signals[side].population_size
        for side in SIDES
    }
    sampled_side = old_signals[Side.SOURCE].sampled_side
    selected_candidates = select_candidates(collected_candidates, selection)
    new_signals = summarize_candidate_selection(
        selected_candidates, population_sizes, sampled_side
        )

    return dataclasses.replace(
        strategy_result, selected=selected_candidates, signals=new_signals
        )


@func_reg.register(TaskFunctionName.RESELECT)
def reselect(state: CandidateTaskState, fn_args: dict) -> str:
    run_id = fn_args.get("run_id")
    if run_id not in state.runs:
        raise FunctionCallException(f"Unknown run_id {run_id!r}; existing runs: {list(state.runs)}")
    try:
        selection = Selection(**fn_args.get("selection", {}))
    except ValidationError as error:
        raise FunctionCallException(f"Invalid selection object: {error}") from error

    generator, result = state.runs[run_id]
    new_generator = generator.model_copy(update={"selection": selection})
    new_result = reselect_result(result, selection)
    new_run_id = state.add_run(new_generator, new_result)

    examples = choose_examples(new_result)
    prefetch_shown_entities(examples, new_result.hubs, state.entity_displays)
    return format_outline([(
        f"Result {new_run_id} (run {run_id} with another selection)",
        format_strategy_result(new_result, examples, state.entity_displays),
    )])


@func_reg.register(TaskFunctionName.FINISH)
def finish(state: CandidateTaskState, fn_args: dict) -> str:
    """Selects the runs whose generators are run in full later, with a rationale."""
    run_ids = fn_args.get("run_ids")
    rationale = fn_args.get("rationale")
    if not isinstance(run_ids, list) or not all(isinstance(run_id, str) for run_id in run_ids):
        raise FunctionCallException("run_ids must be a list of run ids (it may be empty)")
    if not rationale:
        raise FunctionCallException("A rationale is required")
    unknown = [run_id for run_id in run_ids if run_id not in state.runs]
    if unknown:
        raise FunctionCallException(f"Unknown run ids {unknown}; existing runs: {list(state.runs)}")
    state.selection = list(dict.fromkeys(run_ids))  # each once, in order
    state.rationale = rationale
    return f"Selected runs: {', '.join(state.selection) or 'none'}."


class CandidateListKind(StrEnum):
    """Which candidate lists show_candidates picks, matching the signals."""
    RANDOM = "random"                    # any entity with a selected candidate
    AMBIGUOUS = "ambiguous"              # more than one entity on the best rank
    UNIQUE = "unique"                    # a unique candidate (the anchor rule)
    ALREADY_MATCHED = "already_matched"  # the best rank holds an already matched candidate
    DROPPED = "dropped"                  # a tie group dropped because it did not fit into top_k


CANDIDATE_LIST_FILTERS: dict[CandidateListKind, Callable[[SelectedCandidates], bool]] = {
    CandidateListKind.RANDOM: lambda selected: bool(selected.tie_rank_groups),
    CandidateListKind.AMBIGUOUS: lambda selected: selected.collected.ambiguous_at_top_rank,
    CandidateListKind.UNIQUE: lambda selected: selected.unique_candidate is not None,
    CandidateListKind.ALREADY_MATCHED: lambda selected: selected.collected.already_matched_at_top_rank,
    CandidateListKind.DROPPED: lambda selected: selected.dropped_tie_group_rank is not None,
}

DEFAULT_SHOWN_LISTS = 5
MAX_SHOWN_LISTS = 20


def choose_listed_entities(
        result: StrategyResult, side: Side, kind: CandidateListKind, num_entities: int,
        ) -> tuple[list[SelectedCandidates], int]:
    """A random sample (fixed seed: the same call shows the same entities) of
    the entities of the side whose list is of the kind, and how many there are."""
    keep = CANDIDATE_LIST_FILTERS[kind]
    matching = sorted(
        entity for (entity_side, entity), selected in result.selected.items()
        if entity_side == side and keep(selected)
    )
    chosen = random.Random(0).sample(matching, min(num_entities, len(matching)))
    return [result.selected[(side, entity)] for entity in chosen], len(matching)


def find_listed_entities(
        result: StrategyResult, side: Side, entities: list[IRI],
        ) -> tuple[list[SelectedCandidates], list[IRI]]:
    """The lists of the given entities, and the entities without a list (no
    candidate, already matched, or not in the population)."""
    found = [result.selected[(side, entity)] for entity in entities if (side, entity) in result.selected]
    missing = [entity for entity in entities if (side, entity) not in result.selected]
    return found, missing


@func_reg.register(TaskFunctionName.SHOW_CANDIDATES)
def show_candidates(state: CandidateTaskState, fn_args: dict) -> str:
    """
    More candidate lists of a run than its examples: of one side, either of a
    kind (CandidateListKind) or of given entities. Arguments: run_id, side
    ("source" or "target"), and kind (default "random") with n (default
    DEFAULT_SHOWN_LISTS, at most MAX_SHOWN_LISTS), or entities (IRIs).
    """
    run_id = fn_args.get("run_id")
    if run_id not in state.runs:
        raise FunctionCallException(f"Unknown run_id {run_id!r}; existing runs: {list(state.runs)}")
    try:
        side = Side(fn_args.get("side", Side.SOURCE))
        kind = CandidateListKind(fn_args.get("kind", CandidateListKind.RANDOM))
    except ValueError as error:
        raise FunctionCallException(
            f"{error}; side: {[str(s) for s in SIDES]}, kind: {[str(k) for k in CandidateListKind]}"
        ) from error
    num_entities = int(fn_args.get("n", DEFAULT_SHOWN_LISTS))
    if not 1 <= num_entities <= MAX_SHOWN_LISTS:
        raise FunctionCallException(f"n must be between 1 and {MAX_SHOWN_LISTS}")

    _, result = state.runs[run_id]
    entities = fn_args.get("entities")
    outline: Outline = []
    if entities:
        shown, missing = find_listed_entities(result, side, [IRI(entity) for entity in entities])
        title = f"Run {run_id}, the given {side} entities"
        if missing:
            outline.append(
                "without a candidate list (no candidate, already matched, or not in the population): "
                + ", ".join(missing))
    else:
        shown, num_matching = choose_listed_entities(result, side, kind, num_entities)
        title = f"Run {run_id}, {side} entities of kind {kind}: {len(shown)} of {num_matching} shown"
        if result.signals[side].sampled_side is not None:
            title += " (trial run on a sample)"

    prefetch_shown_entities(shown, [], state.entity_displays)
    outline.extend(format_selected_candidates(selected, state.entity_displays) for selected in shown)
    return format_outline([(title, outline or ["no such entity"])])


# ---------------------------------------------------------------- formatting for the LLM
# Every function returns outline items (prompting.format_outline lays them out).
NUM_EXAMPLE_ENTITIES = 3  # entities with candidates shown per side in a result
NUM_SHOWN_HUBS = 3        # largest hubs shown in a result
NUM_SHOWN_MATCHES = 2     # matches shown per candidate


def display_entity(entity_displays: dict[Side, EntityDisplay], side: Side, entity: IRI) -> str:
    """An entity of the side as one line (its names must have been prefetched,
    prefetch_shown_entities)."""
    return entity_displays[side].show(entity)


def format_share(share: float) -> str:
    return f"{share:.0%}"


def format_signals(signals: StrategySignals) -> str:
    text = (
        f"{signals.side}: {signals.population_size} entities"
        f"{' (random sample)' if signals.sampled_side == signals.side else ''}; "
        f"coverage {format_share(signals.coverage)}, "
        f"mean list length {signals.mean_list_length:.1f}, "
        f"ambiguous at top rank {format_share(signals.ambiguous_at_top_rank)}, "
        f"unique {format_share(signals.unique)}"
    )
    if signals.already_matched_at_top_rank:
        text += f", already matched at top rank {format_share(signals.already_matched_at_top_rank)}"
    if signals.sampled_side is not None and signals.sampled_side != signals.side:
        text += " (distorted: sees only the sampled counterparts)"
    elif signals.sampled_side == signals.side:
        text += " (unique is an upper estimate)"
    return text


def format_hub(hub: Hub, entity_displays: dict[Side, EntityDisplay]) -> str:
    """A hub with the size of its bucket."""
    match hub:
        case KeyHub():
            shared = f'{hub.level} key "{hub.key}"'
        case VectorHub():
            shared = f'texts "{hub.source_text}" ~ "{hub.target_text}"'
        case NeighbourHub():
            source_witness, target_witness = hub.witness
            shared = (f"witness {display_entity(entity_displays, Side.SOURCE, source_witness)} = "
                      f"{display_entity(entity_displays, Side.TARGET, target_witness)} over {hub.relation_key}")
        case _:
            raise NotImplementedError(f"format_hub does not support {type(hub).__name__}")
    return f"{shared}: {hub.num_source_entities} source x {hub.num_target_entities} target entities"


def format_hubs(hubs: Sequence[Hub], entity_displays: dict[Side, EntityDisplay]) -> Outline:
    if not hubs:
        return []
    largest = sorted(hubs, key=lambda hub: hub.num_source_entities * hub.num_target_entities, reverse=True)
    return [(
        f"{len(hubs)} hubs (too many entity pairs, so they give no candidates), the largest:",
        [format_hub(hub, entity_displays) for hub in largest[:NUM_SHOWN_HUBS]],
    )]


def format_match(match: Match, own_side: Side, entity_displays: dict[Side, EntityDisplay]) -> str:
    """One match of a candidate, the entity's own value first."""
    source_role, target_role = match.source_rows[0].role_name, match.target_rows[0].role_name
    own_role, candidate_role = (source_role, target_role) if own_side == Side.SOURCE else (target_role, source_role)
    roles = f"({own_role} - {candidate_role})"
    match match:
        case KeyMatch():
            return f'key "{match.key}" on normalization level {match.level} {roles}'
        case VectorMatch():
            own_text, candidate_text = ((match.source_text, match.target_text) if own_side == Side.SOURCE
                                        else (match.target_text, match.source_text))
            return f'"{own_text}" ~ "{candidate_text}", cosine {match.cosine:.2f} {roles}'
        case NeighbourMatch():
            source_witness, target_witness = match.witness
            witness = (f"{display_entity(entity_displays, Side.SOURCE, source_witness)} = "
                       f"{display_entity(entity_displays, Side.TARGET, target_witness)}")
            bucket = match.num_source_entities * match.num_target_entities
            shared = (f", shared by {match.num_source_entities} source x {match.num_target_entities} "
                      f"target entities" if bucket > 1 else "")
            return f"witness {witness} over {match.relation_key}{shared}"
        case _:
            raise NotImplementedError(f"format_match does not support {type(match).__name__}")


def format_candidate(
        candidate: Candidate,
        mutual_candidates: frozenset[IRI],
        unique_candidate: IRI | None,
        entity_displays: dict[Side, EntityDisplay],
        ) -> OutlineItem:
    """The candidate entity with its marks and its best matches."""
    candidate_side = Side.other_side(candidate.own_side)
    marks = [
        mark for mark, applies in [
            ("unique", candidate.candidate_entity == unique_candidate),
            ("mutual", candidate.candidate_entity in mutual_candidates),
            ("already matched", candidate.candidate_already_matched),
        ] if applies
    ]
    groups = ", ".join(sorted(candidate.entity_pair.entity_groups()[candidate_side]))
    text = f"{display_entity(entity_displays, candidate_side, candidate.candidate_entity)} (group {groups})"
    if marks:
        text += f" [{', '.join(marks)}]"
    entity_pair = candidate.entity_pair
    if isinstance(entity_pair, NeighbourEntityPair) and entity_pair.witness_weighting != "count":
        text += f", witness score {entity_pair.witness_score():.2f}"
    matches = entity_pair.matches
    shown: Outline = [format_match(match, candidate.own_side, entity_displays) for match in matches[:NUM_SHOWN_MATCHES]]
    if len(matches) > NUM_SHOWN_MATCHES:
        shown.append(f"+{len(matches) - NUM_SHOWN_MATCHES} more matches")
    return (text, shown)


def format_rivals(entity_candidates: EntityCandidates) -> Outline:
    """The counterparts that are no listed candidates, in summary."""
    outline: Outline = []
    for title, counts in [
        ("competing counterparts outside the population, by group",
         entity_candidates.num_rivals_outside_population_by_group),
        ("counterparts in non-competing partitions, by partition",
         entity_candidates.num_non_competing_rivals_by_partition),
    ]:
        if counts:
            outline.append(f"{title}: " + ", ".join(f"{name} {count}" for name, count in counts.most_common()))
    return outline


def format_selected_candidates(selected: SelectedCandidates, entity_displays: dict[Side, EntityDisplay]) -> tuple[str, Outline]:
    """
    The selection of one entity: its tie groups in rank order (rank 1 = best),
    each with its candidates, then the counterparts that are not listed.
    """
    collected = selected.collected
    outline: Outline = []
    for position, tie_group in enumerate(selected.tie_rank_groups, start=1):
        rivals_outside = collected.num_rivals_outside_by_rank[tie_group[0].rank_key]
        size = len(tie_group) + rivals_outside
        title = f"rank {position}" + (f" (tie of {size}" if size > 1 else "")
        if rivals_outside:
            title += f", {rivals_outside} of them outside the population"
        title += ")" if size > 1 else ""
        outline.append((title, [
            format_candidate(candidate, selected.mutual_candidates, selected.unique_candidate, entity_displays)
            for candidate in tie_group
        ]))
    if selected.dropped_tie_group_rank is not None:
        dropped = (sum(candidate.rank_key == selected.dropped_tie_group_rank for candidate in collected.candidates)
                   + collected.num_rivals_outside_by_rank[selected.dropped_tie_group_rank])
        outline.append(f"next tie group dropped: its {dropped} entities do not fit into top_k")
    outline.extend(format_rivals(collected))
    side = collected.side
    return (f"{side} {display_entity(entity_displays, side, collected.entity)}", outline)


def shown_entities(examples: list[SelectedCandidates], hubs: Sequence[Hub]) -> dict[Side, set[IRI]]:
    """All entities a result may show, per side: the example entities, their
    selected candidates, and the witnesses of the candidates and hubs (more
    than are shown: one query per side instead of one per entity)."""
    entities_by_side: dict[Side, set[IRI]] = {side: set() for side in SIDES}

    def add_witness(witness: tuple[IRI, IRI]) -> None:
        entities_by_side[Side.SOURCE].add(witness[0])
        entities_by_side[Side.TARGET].add(witness[1])

    for selected in examples:
        entities_by_side[selected.collected.side].add(selected.collected.entity)
        for candidate in selected.selected_candidates:
            entities_by_side[Side.other_side(candidate.own_side)].add(candidate.candidate_entity)
            for match in candidate.entity_pair.matches:
                if isinstance(match, NeighbourMatch):
                    add_witness(match.witness)
    for hub in hubs:
        if isinstance(hub, NeighbourHub):
            add_witness(hub.witness)
    return entities_by_side


def prefetch_shown_entities(
        examples: list[SelectedCandidates], hubs: Sequence[Hub], entity_displays: dict[Side, EntityDisplay],
        ) -> None:
    for side, entities in shown_entities(examples, hubs).items():
        entity_displays[side].prefetch(entities)


def choose_examples(result: StrategyResult) -> list[SelectedCandidates]:
    """The example entities of a result: per side a few with a selected candidate."""
    return [
        selected for side in SIDES
        for selected in choose_listed_entities(result, side, CandidateListKind.RANDOM, NUM_EXAMPLE_ENTITIES)[0]
    ]


def format_strategy_result(
        result: StrategyResult,
        examples: list[SelectedCandidates],
        entity_displays: dict[Side, EntityDisplay],
        ) -> Outline:
    """
    The LLM view of a run: the signals per side, the existing equivalences found
    again, the largest hubs, and the example entities with their selection
    (choose_examples; prefetch_shown_entities first). The full result stays in
    the task state (show_candidates).
    """
    outline: Outline = [("signals", [format_signals(result.signals[side]) for side in SIDES])]
    sampled = any(signals.sampled_side is not None for signals in result.signals.values())
    if result.existing_equivalences and sampled:
        # the sample holds no entity with an equivalence: none can be found again
        outline.append("existing equivalences found again: not measured in a trial run on a sample")
    elif result.existing_equivalences:
        outline.append(
            f"finds {len(result.confirmed_equivalences)} of {result.existing_equivalences} "
            f"existing equivalences again (as entity pairs, before the selection)"
        )
    outline.extend(format_hubs(result.hubs, entity_displays))
    example_items: Outline = [format_selected_candidates(selected, entity_displays) for selected in examples]
    outline.append(("examples", example_items) if example_items else "no entity has a candidate")
    return outline


# ---------------------------------------------------------------- start message

NUM_EXAMPLE_EQUIVALENCES = 5


def count_group_entities(
        manager: KgManager, kg_profile: KgProfile, matched_entities: Collection[IRI],
        ) -> dict[str, tuple[int, int]]:
    """Per group of the profile: its entities and those already matched (an
    entity in several groups counts in each). Not for a large KG."""
    grouping = profile_groups_to_grouping(kg_profile)
    sizes = group_sizes(manager, grouping, kg_profile.entity_space)
    matched = (counts_by_group(manager, grouping, kg_profile.entity_space, values_clause("entity", matched_entities))
               if matched_entities else {})
    return {group_name: (size, matched.get(group_name, 0)) for group_name, size in sizes.items()}


def format_group_counts(state: CandidateTaskState, side: Side) -> OutlineItem:
    kg_profile = state.alignment_state.get_profile(side)
    if kg_profile.large:
        return "groups not counted (large KG)"
    matched_entities = {
        correspondence.entity_source if side == Side.SOURCE else correspondence.entity_target
        for correspondence in state.alignment_state.correspondences
    }
    counts = count_group_entities(state.kg_managers[side], kg_profile, matched_entities)
    excluded = set(kg_profile.excluded_group_names())
    return ("entities per group (all / already matched / still to match)", [
        f"{group_name}: {size} / {matched} / {size - matched}" + (" (excluded)" if group_name in excluded else "")
        for group_name, (size, matched) in sorted(counts.items(), key=lambda item: -item[1][0])
    ])


def format_knowledge_graph(state: CandidateTaskState, side: Side) -> str:
    kg_profile = state.alignment_state.get_profile(side)
    manager = state.kg_managers[side]
    profile_notes: Outline = list(describe_profile(kg_profile, findings=False))
    outline: Outline = [("profile", profile_notes), format_group_counts(state, side)]
    text = f"{manager.description}\n\n" if manager.description else ""
    return format_section(f"{manager.kg} ({side})", text + format_outline(outline), level=3)


def format_alignment_status(state: CandidateTaskState) -> str:
    correspondences = state.alignment_state.correspondences
    if not correspondences:
        return format_section("Equivalences found so far", "none", level=2)
    partitions = state.partition_names_by_entity
    by_partition_pair = Counter(
        (partitions[Side.SOURCE].get(correspondence.entity_source, "?"),
         partitions[Side.TARGET].get(correspondence.entity_target, "?"))
        for correspondence in correspondences
    )
    examples = random.Random(0).sample(correspondences, min(NUM_EXAMPLE_EQUIVALENCES, len(correspondences)))
    for side in SIDES:
        state.entity_displays[side].prefetch(
            correspondence.entity_source if side == Side.SOURCE else correspondence.entity_target
            for correspondence in examples
        )
    outline: Outline = [
        (f"{len(correspondences)} equivalences, by partition pair (source - target)", [
            f"{source_partition} - {target_partition}: {count}"
            for (source_partition, target_partition), count in by_partition_pair.most_common()
        ]),
        ("examples", [
            f"{state.entity_displays[Side.SOURCE].show(correspondence.entity_source)} = "
            f"{state.entity_displays[Side.TARGET].show(correspondence.entity_target)}"
            for correspondence in examples
        ]),
    ]
    return format_section("Equivalences found so far", format_outline(outline), level=2)


def format_start_message(state: CandidateTaskState) -> str:
    knowledge_graphs = "\n\n".join(format_knowledge_graph(state, side) for side in SIDES)
    return (
        format_section("Knowledge graphs", knowledge_graphs, level=2)
        + "\n\n" + format_alignment_status(state)
        + "\n\nNo strategy has been tried yet."
    )


# ---------------------------------------------------------------- function definitions

GROUP_REFS_SCHEMA = {
    "type": "array",
    "items": {"anyOf": [
        {"type": "string"},
        {"type": "object", "properties": {
            "name": {"type": "string"},
            "membership_sparql": {"type": "string", "description": "SPARQL group pattern over ?entity"},
        }, "required": ["name", "membership_sparql"]},
    ]},
    "description": (
        f"Groups of the profile by name, {REST!r} for all groups that are not excluded, or a group "
        "defined inline for this strategy (a new name and a SPARQL pattern over ?entity)."
    ),
}

VALUE_SOURCE_SCHEMA = {
    "anyOf": [
        {"type": "array", "items": {"type": "string"},
         "description": "Role names of the profile (names, synonyms, definitions, codes, cross_references) "
                        "or names of structure relations"},
        {"type": "object", "properties": {
            "predicates": {"type": "array", "items": {"type": "string"}},
            "patterns": {"type": "array", "items": {"type": "string"},
                         "description": "SPARQL patterns over ?s (the entity) and ?o (the value)"},
        }},
    ],
}

SELECTION_SCHEMA = {
    "type": "object",
    "description": "From the collected candidates to the candidate list of an entity.",
    "properties": {
        "top_k": {"type": "integer", "minimum": 1,
                  "description": "Places of a list; tied candidates are taken whole or not at all. Default 5."},
        "mutual": {"type": "boolean", "description": "Keep only candidates whose own list has the entity."},
        "unique": {"type": "boolean", "description": "Keep only the unique candidate (alone in front from both sides)."},
    },
}

GENERATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "population": {
            "type": "object",
            "description": "The entities compared. Entities that already have an equivalence get no "
                           "candidates. Default: rest on both sides.",
            "properties": {"source_group_refs": GROUP_REFS_SCHEMA, "target_group_refs": GROUP_REFS_SCHEMA},
        },
        "strategy": {
            "type": "object",
            "description": (
                "key: candidates share a normalized value. vector: candidates have the most similar "
                "values by cosine. neighbours: candidates are related to the entities of an existing "
                "equivalence by the same kind of structure relation (e.g. subclasses of matched classes)."
            ),
            "properties": {
                "kind": {"type": "string", "enum": [str(kind) for kind in CandidateStrategyType]},
                "values": {
                    "type": "object",
                    "description": "key and vector: the values compared. Default: names and synonyms.",
                    "properties": {
                        "source": VALUE_SOURCE_SCHEMA,
                        "target": {**VALUE_SOURCE_SCHEMA, "description": "Omit: like the source."},
                    },
                },
                "normalization": {
                    "type": "object",
                    "description": "key and vector: how values are normalized before they are compared.",
                    "properties": {
                        "transforms": {"type": "array", "items": {"type": "object", "properties": {
                            "operation": {"type": "string", "enum": list(TRANSFORM_OPERATIONS)},
                            "pattern": {"type": "string", "description": "regex_replace only"},
                            "replacement": {"type": "string"},
                        }, "required": ["operation"]}},
                        "levels": {
                            "type": "array", "items": {"type": "string", "enum": list(NORMALIZATION_LEVELS)},
                            "description": "Levels from fine to coarse, in this order; a key pair ranks by "
                                           "its finest shared level. vector: one level, default words.",
                        },
                    },
                },
                "vectorizer": {
                    "type": "string", "enum": list(get_args(Vectorizer.__value__)),
                    "description": "vector: idf_tokens (rare words count more), char_3grams (spelling "
                                   "variants), embedding (meaning, multilingual). Default idf_tokens.",
                },
                "relation_kinds": {
                    "type": "array", "items": {"type": "string", "enum": [str(kind) for kind in StructureRelationKind]},
                    "description": "neighbours: the structure relations of these kinds of the profiles, "
                                   "in both directions. Default subsumption, partonomy, membership.",
                },
                "relations": {
                    "type": "array", "items": {"type": "string"},
                    "description": "neighbours: structure relations by name instead of relation_kinds "
                                   "('^name': the other direction).",
                },
                "min_witnesses": {"type": "integer", "minimum": 1,
                                  "description": "neighbours: equivalences a pair needs at least. Default 1."},
                "witness_weighting": {
                    "type": "string", "enum": ["count", "inverse_degree"],
                    "description": "neighbours: count (default), or inverse_degree for hubs (a witness "
                                   "shared by many entities counts less).",
                },
            },
            "required": ["kind"],
        },
        "selection": SELECTION_SCHEMA,
    },
}

FINISH_DEFINITION = {
    "name": TaskFunctionName.FINISH,
    "description": "Ends the task: selects the tried runs whose strategies are run in full later "
                   "(on all entities, not a sample). Call it exactly once, at the end.",
    "parameters": {
        "type": "object",
        "properties": {
            "run_ids": {"type": "array", "items": {"type": "string"},
                        "description": "The selected runs; empty if no strategy is good."},
            "rationale": {"type": "string",
                          "description": "Which strategies for which groups, why, and which candidates are risky."},
        },
        "required": ["run_ids", "rationale"],
    },
}


def candidate_functions() -> list[dict]:
    return [
        {
            "name": TaskFunctionName.TRY_STRATEGY,
            "description": (
                "Tries a candidate generator (population, strategy, selection) in a trial run, on a sample "
                "of the larger side if it is large, and returns the signals per side, the existing "
                "equivalences it finds again, its hubs (values shared by too many entities, which give "
                "no candidates) and example candidate lists. The run gets an id (run-N)."
            ),
            "parameters": {"type": "object", "properties": {"generator": GENERATOR_SCHEMA}, "required": ["generator"]},
        },
        {
            "name": TaskFunctionName.RESELECT,
            "description": "Selects the candidates of a tried run again with another selection, without "
                           "running the strategy again; gives a new run.",
            "parameters": {"type": "object", "properties": {
                "run_id": {"type": "string"}, "selection": SELECTION_SCHEMA,
            }, "required": ["run_id", "selection"]},
        },
        {
            "name": TaskFunctionName.SHOW_CANDIDATES,
            "description": "Shows more candidate lists of a run: of a kind matching the signals, or of "
                           "given entities.",
            "parameters": {"type": "object", "properties": {
                "run_id": {"type": "string"},
                "side": {"type": "string", "enum": [str(side) for side in SIDES]},
                "kind": {"type": "string", "enum": [str(kind) for kind in CandidateListKind],
                         "description": "random (default), ambiguous, unique, already_matched (the best rank "
                                        "holds an entity that already has an equivalence) or dropped "
                                        "(a tie group did not fit into top_k)."},
                "n": {"type": "integer", "minimum": 1, "maximum": MAX_SHOWN_LISTS},
                "entities": {"type": "array", "items": {"type": "string"},
                             "description": "IRIs of entities of the side, instead of kind and n."},
            }, "required": ["run_id"]},
        },
        FINISH_DEFINITION,
    ]


def system_information(config: GraspConfig) -> str:
    info = load_prompt("candidate_task.txt")
    info = add_equivalence_condition(info, config)
    task_instructions = load_task_instructions(config.task_kwargs.get("om", {}))
    if task_instructions:
        info += "\n\n" + format_section("Task-specific instructions", task_instructions)
    return info


def rules() -> list[str]:
    return []


class OMCandidateTask(GraspTask):
    """
    The candidate stage: the LLM tries candidate generators and selects those
    that are run in full later. Input: a dict with source_kg, target_kg (names
    of the configured KGs) and alignment_state (a serialized AlignmentState with
    the profiles and the equivalences found so far).
    """
    name = "om-candidates"
    state: CandidateTaskState

    def setup(self, input: Any) -> str:
        source_manager, target_manager = resolve_managers(self, input)
        kg_managers = {Side.SOURCE: source_manager, Side.TARGET: target_manager}
        alignment_state = resolve_alignment_state(input)
        self.state = CandidateTaskState(
            kg_managers=kg_managers,
            alignment_state=alignment_state,
            partition_names_by_entity=retrieve_partition_names(kg_managers, alignment_state),
            entity_displays={
                side: EntityDisplay.of_profile(kg_managers[side], alignment_state.get_profile(side))
                for side in SIDES
            },
        )
        return format_start_message(self.state)

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return [
            "Try every strategy you want to select with try_strategy; you can only select tried runs.",
            "Look at the examples and at more lists with show_candidates before you judge a strategy; "
            "use reselect to change only the selection of a run.",
            "You can execute multiple tool calls in one step.",
            f"You have at most {self.config.max_steps} steps; one step is one response of yours, with "
            "all tool calls in it. Call finish exactly once, well before the limit.",
        ]

    def function_definitions(self) -> list[dict]:
        return candidate_functions()

    def final_step_functions(self) -> list[dict] | None:
        return [FINISH_DEFINITION]

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        function = func_reg.Registry.get(fn_name)  # type: ignore[call-overload]
        if function is None:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return function(self.state, fn_args)

    def done(self, fn_name: str) -> bool:
        return fn_name == TaskFunctionName.FINISH and self.state.selection is not None

    @property
    def default_input_field(self) -> str | None:
        return None

    def output(self, messages: list[Message]) -> dict:
        state = self.state
        return CandidateStageResult(
            source_kg=state.kg_managers[Side.SOURCE].kg,
            target_kg=state.kg_managers[Side.TARGET].kg,
            finished=state.selection is not None,
            rationale=state.rationale,
            selected_run_ids=state.selection or [],
            runs={
                run_id: TriedRun(
                    generator=generator,
                    signals=result.signals,
                    confirmed_equivalences=len(result.confirmed_equivalences),
                    existing_equivalences=result.existing_equivalences,
                    num_hubs=len(result.hubs),
                )
                for run_id, (generator, result) in state.runs.items()
            },
        ).model_dump(mode="json")
