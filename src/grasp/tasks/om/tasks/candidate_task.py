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
whole other side (draw_trial_sample). The sample is consistent across runs
(draw_entity_sample): an entity is drawn when its fixed sample position lies
below the sampling share, so an entity in the populations of two runs is in
both samples if it is in the one with the smaller share, and the trial runs of
an ensemble can be merged on the same entities. Only one side is sampled: sampling both
would lose the true counterparts and distort every signal.
"""
from collections import Counter
import dataclasses
from enum import StrEnum, auto
import hashlib
from itertools import chain
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Iterable, Iterator, Sequence, get_args

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidates import (
    CandidateStrategy,
    CandidateStrategyType,
    Generator,
    KeyStrategy,
    NeighbourStrategy,
    Selection,
    VectorStrategy,
    CandidateStageResult,
    EnsembleEvaluation,
    MergedCandidates,
    RunEnsemble,
    RunRole,
    SelectedCandidates,
    StrategyResult,
    StrategySignals,
    TriedRun,
    Candidate,
    EntityCandidates,
    Hub,
    KeyHub,
    KeyMatch,
    Match,
    NeighbourEntityPair,
    NeighbourHub,
    NeighbourMatch,
    VectorHub,
    VectorMatch,
    POPULATION_REST,
    GroupRef,
    Population,
    ValuePatternSource,
    ValueRow,
)
from grasp.tasks.om.data_models.profile import (
    StructureRelationKind,
    GroupDefinition,
    KgProfile,
    describe_profile,
)
from grasp.tasks.om.foundation.configuration import load_task_instructions
from grasp.tasks.om.foundation.rendering import (
    Item,
    Section,
    SectionContent,
    Table,
    format_display_content,
)
from grasp.tasks.om.foundation.text_normalization import (
    NORMALIZATION_LEVELS,
    REGEX_REPLACE,
    TRANSFORM_OPERATIONS,
    NormalizationLevel,
    normalization_level_order,
    words,
)
from grasp.tasks.om.foundation.text_vectors import Vectorizer
from grasp.tasks.om.foundation.types import IRI, SIDES, Side
from grasp.tasks.om.kg_operations.candidate_selection import (
    NAME_ROLES, compares_names, evaluate_merged_candidates, flatten_layers, merge_candidate_runs, select_candidates, summarize_candidate_selection,
)
from grasp.tasks.om.kg_operations.entity_rendering import SHORT_VIEW, EntityDisplay
from grasp.tasks.om.kg_operations.grouping import (
    Grouping, retrieve_groups_by_entity, profile_groups_to_grouping,
)
from grasp.tasks.om.kg_operations.candidate_generation import (
    NeighbourIndex,
    ValueIndex,
    build_side_indices,
    build_side_witness_index,
    compare_value_indices_by_vectors,
    compare_witness_indices,
    find_confirmed_equivalences,
    intersect_level_indices,
    key_entity_pairs_to_candidates,
    neighbour_entity_pairs_to_candidates,
    get_normalization_examples,
    split_name_key_runs,
    vector_entity_pairs_to_candidates,
    retrieve_partitions_by_entity,
    retrieve_value_rows,
)
from grasp.tasks.om.tasks.function_registry import FunctionRegistry
from grasp.tasks.om.tasks.prompting import (
    BLANK_NODE_RULE,
    add_equivalence_condition,
    format_validation_error,
    load_prompt,
)
from grasp.tasks.om.tasks.task_input import managers_by_side, resolve_alignment_state
from grasp.utils import FunctionCallException, format_section
from pydantic import ValidationError


# TODO: Move to lower level
def group_refs_of_side(population: Population, side: Side) -> list[GroupRef]:
    return population.source_group_refs if side == Side.SOURCE else population.target_group_refs


def inline_groups(population: Population, side: Side) -> list[GroupDefinition]:
    """The groups a strategy defines inline for one side (not in the KgProfile)."""
    return [group_ref for group_ref in group_refs_of_side(population, side) if isinstance(group_ref, GroupDefinition)]


def population_group_names(kg_profile: KgProfile, population: Population, side: Side) -> set[str]:
    """
    The names of the groups of one side's population. REST stands for all groups
    of the profile that are not excluded (for now; later: only the entities that
    have no equivalence yet). Raises FunctionCallException for a name that is
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
        elif group_ref == POPULATION_REST:
            names |= set(kg_profile.group_names()) - set(kg_profile.excluded_group_names())
        elif kg_profile.has_group(group_ref):
            names.add(group_ref)
        else:
            raise FunctionCallException(
                f"{group_ref} is not a group of {kg_profile.kg_name}; groups: {kg_profile.group_names()}, "
                f"or {POPULATION_REST!r} for all of them, or define a group inline")
    return names


def generate_group_grouping(kg_profile: KgProfile, population: Population, side: Side) -> Grouping:
    """
    The groups whose values are retrieved for one side: in a small KG all groups of
    the profile and the inline groups (all are needed to count rivals outside the
    population by group), without the entities of excluded groups.
    """
    return profile_groups_to_grouping(kg_profile, inline_groups(population, side))




def generate_population_groups_grouping(kg_profile: KgProfile, population: Population, side: Side) -> Grouping:
    """
    The groups of one side's population only 
    (no rivals), without the entities
    of excluded groups.
    """
    all_groups = profile_groups_to_grouping(kg_profile, inline_groups(population, side))
    group_names = population_group_names(kg_profile, population, side)
    return Grouping(
        tuple(member for member in all_groups.members if member[0] in group_names),
        exclusive=False, minus=all_groups.minus,
    )


def retrieve_side_population(
        manager: KgManager,
        kg_profile: KgProfile,
        population: Population,
        side: Side,
        leave_out_entities: Collection[IRI] = ()
) -> frozenset[IRI]:
    """
    The entities of one side's population, each once (also those without any
    value of the strategy: the denominator of the signals), without the
    leave_out_entities (those with an equivalence: they get no candidates).
    """
    groups_by_entity = retrieve_groups_by_entity(
        manager,
        generate_population_groups_grouping(kg_profile, population, side),
        kg_profile.entity_space
        )
    return frozenset(groups_by_entity) - frozenset(leave_out_entities)


# trial runs: the most entities of the larger population that are compared, per
# vectorizer (in a small KG the key strategy is cheap and runs on the whole population)
TRIAL_ENTITY_LIMITS: dict[str, int] = {
    "idf_tokens": 5000,
    "char_3grams": 5000,
    "embedding": 5_000,
}


@dataclass(frozen=True)
class EntitySample:
    """
    The entities of one side's population that a trial run compares; the
    other side is compared whole.
    """
    side: Side
    entities: frozenset[IRI]


def trial_entity_limit(strategy: CandidateStrategy) -> int | None:
    """The trial limit of a strategy; None: no sampling."""
    if isinstance(strategy, VectorStrategy):
        return TRIAL_ENTITY_LIMITS.get(strategy.vectorizer)
    return None


def choose_sampled_side(population_sizes: dict[Side, int], limit: int | None) -> Side | None:
    """The side to sample in a trial run: the one with the larger population, if it is above the limit (the source on equal
    sizes); None: compare both whole."""
    larger_side = max(SIDES, key=lambda side: population_sizes[side])  # first maximum: the source
    if limit is None or population_sizes[larger_side] <= limit:
        return None
    return larger_side


def sample_position(entity: IRI, seed: int) -> float:
    """The fixed position of an entity in [0, 1) for a seed: a stable hash of
    both (not Python's hash, which changes between processes), so it is the
    same in every run and every population."""
    digest = hashlib.blake2b(f"{seed}|{entity}".encode(), digest_size=8).digest()
    return int.from_bytes(digest) / 2**64


def draw_entity_sample(entities: Iterable[IRI], size: int, seed: int) -> frozenset[IRI]:
    """
    A random sample of about `size` entities: those whose sample position
    lies below the share size / number of entities. Consistent across runs:
    the same entities and seed give the same sample, and of two populations
    an entity in both is in both samples if it is in the one with the smaller
    share (nested samples, the most overlap two samples can have).
    """
    distinct_entities = set(entities)
    if not distinct_entities:
        return frozenset()
    sampling_share = size / len(distinct_entities)
    return frozenset(entity for entity in distinct_entities if sample_position(entity, seed) < sampling_share)


def draw_trial_sample(
        generator: Generator,
        populations_by_side: dict[Side, frozenset[IRI]],
        seed: int,
        ) -> EntitySample | None:
    """
    The sample of a trial run (see the module docstring); None: no sampling
    needed. populations_by_side: the populations, already without the
    entities that have an equivalence.
    """
    limit = trial_entity_limit(generator.strategy)
    side = choose_sampled_side(
        {side: len(population) for side, population in populations_by_side.items()},
        limit
        )
    if side is None or limit is None:
        return None
    population = populations_by_side[side]
    return EntitySample(
        side=side,
        entities=draw_entity_sample(population, min(limit, len(population)), seed)
        )


def check_value_pattern_source(
        value_pattern_source: ValuePatternSource, kg_profile: KgProfile
        ):
    """
    A name in the value source is a role, a structure relation or relation
    kind of the profile, or a path of them (each step checked), and has a
    pattern there: a role that is not set, or a kind without a relation, gives
    no values. Checked before querying, for a message the LLM can act on.
    """
    if not isinstance(value_pattern_source, str):
        return  # an own pattern
    try:
        pattern = kg_profile.roles.predicate_pattern_by_name(value_pattern_source)
    except ValueError as error:
        raise FunctionCallException(f"{kg_profile.kg_name}: {error}") from error
    if pattern is None:
        raise FunctionCallException(
            f"{kg_profile.kg_name}: {value_pattern_source!r} gives no values: a role in it is not set in the "
            "profile, or the profile has no structure relation of a kind in it")


def retrieve_strategy_value_rows(
        manager: KgManager,
        kg_profile: KgProfile,
        generator: Generator,
        side: Side,
        entity_sample: EntitySample | None = None,
        object_iris: Collection[IRI] | None = None,
        alternative_value_pattern_source: ValuePatternSource | None = None
        ) -> Iterator[ValueRow]:
    """
    The values a strategy compares on one side (its value_pattern_source: roles,
    or the structure relations of a neighbour strategy): of the entities of all
    groups (rivals outside the population included); on a sampled side only
    those of the sampled entities. object_iris: only values among these (a
    neighbour strategy: the entities of the witnesses).
    alternative_value_pattern_source: these values instead of the strategy's
    (a key run on a name role: the other name roles, for its rivals).
    """
    value_pattern_source = (
        alternative_value_pattern_source if alternative_value_pattern_source is not None
        else generator.strategy.value_pattern_source(side)
        )
    check_value_pattern_source(value_pattern_source, kg_profile)
    side_sample = entity_sample if entity_sample is not None and entity_sample.side == side else None
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


def retrieve_partition_names(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        ) -> dict[Side, dict[IRI, str]]:
    """The partition of every entity, per side. Depends only on the profiles, so it
    is retrieved once per run and reused by every strategy."""
    partition_names_by_entity = {}
    for side in SIDES:
        kg_profile = alignment_state.get_kg_profile(side)
        partition_names_by_entity[side] = retrieve_partitions_by_entity(
            managers[side], kg_profile, kg_profile.entity_space)
    return partition_names_by_entity

# TODO:
# ------------------------------------------------------------------
# From here functions really belong in this file
# ------------------------------------------------------------------


CANDIDATE_BUDGET_PER_ENTITY = 30
MAX_CANDIDATES_IN_FIRST_PARETO_SUBLAYER = 30


def no_values_error(side: Side) -> ValueError:
    """A key or vector run without any value on one side: why, for the LLM."""
    return ValueError(
        f"No values on the {side} side for the compared entities: the role is not set, has no values "
        "in these groups (or in the trial sample), or your pattern binds no ?o (inside a nested { } "
        "group of its own, ?s is unbound)."
        )


def get_key_candidates(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        max_entity_pairs_per_bucket: int | None = None,
        entity_sample: EntitySample | None = None,
        ) -> tuple[
            dict[tuple[Side, IRI], EntityCandidates], list[KeyHub], list[tuple[IRI, IRI]],
            dict[Side, frozenset[frozenset[str]]],
            dict[Side, list[tuple[str, str, str]]] | None
        ]:
    """
    The candidates of a key strategy for every entity in its population, on both
    sides, the keys whose entities would give too many entity pairs (above
    max_entity_pairs_per_bucket), the existing equivalences found again, and per
    side the word sets of all values compared (value_wordsets) and, with
    transforms, a few normalization examples. entity_sample: a trial run
    compares only these entities on their side.
    Not yet ranked or cut by generator.selection.
    """
    strategy = generator.strategy
    if not isinstance(strategy, KeyStrategy):
        raise FunctionCallException(
            f"get_key_candidates needs a key strategy, not {strategy.kind!r}"
            )

    # 1. per side: the values of the entities in the groups -> indices per level
    retrieve_all_name_roles = compares_names(generator)

    indices_by_side = {}
    group_names_by_side = {}
    for side in SIDES:
        kg_profile = alignment_state.get_kg_profile(side)
        group_names_by_side[side] = population_group_names(
            kg_profile, generator.population, side
            )
        if retrieve_all_name_roles:
            # the projection must give values; the rivals come from every name role that is set
            check_value_pattern_source(strategy.values.by_side(side), kg_profile)
            value_rows = chain.from_iterable(
                retrieve_strategy_value_rows(
                    managers[side], kg_profile, generator,
                    side, entity_sample, alternative_value_pattern_source=role)
                for role in sorted(NAME_ROLES)
                if kg_profile.roles.predicate_pattern_by_name(role) is not None
            )
        else:
            value_rows = retrieve_strategy_value_rows(
                managers[side], kg_profile, generator, side, entity_sample
            )
        indices_by_side[side] = build_side_indices(value_rows, strategy.normalization)
        if not any(indices_by_side[side].values()):
            raise no_values_error(side)

    # examples for raw value -> transform -> string normalization for LLM
    normalization_examples = {
        side: get_normalization_examples(
            indices_by_side[side][strategy.normalization.levels[0]],
            strategy.normalization,
            NUM_NORMALIZATION_EXAMPLES
            ) for side in SIDES
        } if strategy.normalization.transforms else None

    # 2. intersect the two sides: entity pairs with their shared keys
    entity_pairs, overfull_keys = intersect_level_indices(
        indices_by_side[Side.SOURCE],
        indices_by_side[Side.TARGET],
        max_entity_pairs_per_bucket
        )

    # 3. per entity in its population without an equivalence: candidates, rivals,
    # non-competing counterparts; and the equivalences found again
    existing_equivalences = alignment_state.existing_equivalences()

    rival_keys: set[tuple[IRI, IRI]] = set()
    if retrieve_all_name_roles:
        entity_pairs, rival_keys = split_name_key_runs(
            entity_pairs, {
                side: strategy.values.by_side(side)
                for side in SIDES
                }
        )

    candidate_pairs = {
        keys: pair for keys, pair in entity_pairs.items()
        if keys not in rival_keys
        }

    candidates_by_entity = key_entity_pairs_to_candidates(
        alignment_state, entity_pairs,
        group_names_by_side, partition_names_by_entity,
        existing_equivalences, rival_keys
        )
    value_wordsets = {side: value_wordsets_of(indices_by_side[side]) for side in SIDES}
    return (
        candidates_by_entity,
        overfull_keys,
        find_confirmed_equivalences(candidate_pairs, existing_equivalences),
        value_wordsets,
        normalization_examples
        )


def value_wordsets_of(indices_by_level: dict[NormalizationLevel, ValueIndex]) -> frozenset[frozenset[str]]:
    """The word sets of all values of one side (each level's index holds all rows)."""
    any_index = next(iter(indices_by_level.values()), {})
    return frozenset(
        wordset for rows in any_index.values() for row in rows
        if (wordset := frozenset(words(row.raw_projection_value)))
        )


def get_vector_candidates(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        generator: Generator,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        max_entity_pairs_per_bucket: int | None = None,
        neighbours_per_text: int = 50,
        entity_sample: EntitySample | None = None,
        ) -> tuple[
            dict[tuple[Side, IRI], EntityCandidates], list[VectorHub], list[tuple[IRI, IRI]],
            dict[Side, list[tuple[str, str, str]]] | None
        ]:
    """
    The candidates of a vector strategy for every entity in its population, on
    both sides, the text pairs whose entities would give too many entity
    pairs (above max_entity_pairs_per_bucket), the existing equivalences found
    again, and with transforms per side a few normalization examples. The strategy compares the texts
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
        kg_profile = alignment_state.get_kg_profile(side)
        group_names_by_side[side] = population_group_names(kg_profile, generator.population, side)
        value_rows = retrieve_strategy_value_rows(managers[side], kg_profile, generator, side, entity_sample)
        # exactly one level (VectorStrategy checks it)
        (index_by_side[side],) = build_side_indices(value_rows, strategy.normalization).values()
        if not any(index_by_side[side].values()):
            raise no_values_error(side)

    normalization_examples = {
        side: get_normalization_examples(
            index_by_side[side],
            strategy.normalization,
            NUM_NORMALIZATION_EXAMPLES
            ) for side in SIDES
        } if strategy.normalization.transforms else None

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
    witnesses_by_entity = alignment_state.existing_equivalences()
    candidates_by_entity = vector_entity_pairs_to_candidates(
        alignment_state, entity_pairs, group_names_by_side, partition_names_by_entity, witnesses_by_entity
        )
    return (candidates_by_entity, vector_hubs, find_confirmed_equivalences(entity_pairs, witnesses_by_entity),
            normalization_examples)


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
    witnesses_by_entity = alignment_state.existing_equivalences()
    if not witnesses_by_entity[Side.SOURCE]:
        raise FunctionCallException(
            "A neighbour strategy needs equivalences found before (its witnesses), "
            "but there are none yet: run a key or vector strategy first."
        )

    # 1. per side: the neighbours that take part in a witness -> witness index
    index_by_side: dict[Side, NeighbourIndex] = {}
    group_names_by_side = {}
    for side in SIDES:
        kg_profile = alignment_state.get_kg_profile(side)
        group_names_by_side[side] = population_group_names(
            kg_profile, generator.population, side
            )
        value_rows = retrieve_strategy_value_rows(
            managers[side], kg_profile, generator,
            side, entity_sample, witnesses_by_entity[side].keys()
            )
        index_by_side[side] = build_side_witness_index(
            witnesses_by_entity[side],
            value_rows,
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


def compared_entities_by_side(
        population_by_side: dict[Side, frozenset[IRI]],
        entity_sample: EntitySample | None,
        ) -> dict[Side, frozenset[IRI]]:
    """
    The entities a run compares per side: on the sampled side the sample,
    otherwise the population.
    """
    compared_entities: dict[Side, frozenset[IRI]] = {}
    for side in SIDES:
        if entity_sample is not None and entity_sample.side == side:
            compared_entities[side] = entity_sample.entities
        else:
            compared_entities[side] = population_by_side[side]
    return compared_entities


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
    consistent across runs (draw_entity_sample), so that trials of different
    strategies see the same entities as far as their populations and shares allow.
    """
    # 0. the populations without the entities that already have an equivalence,
    # and in a trial run the sample of the larger one
    existing_equivalences_by_entity = alignment_state.existing_equivalences()
    population_by_side: dict[Side, frozenset[IRI]] = {
        side: retrieve_side_population(
            managers[side],
            alignment_state.get_kg_profile(side),
            generator.population,
            side,
            existing_equivalences_by_entity[side].keys()
            )
        for side in SIDES
        }
    population_sizes = {side: len(entities) for side, entities in population_by_side.items()}
    entity_sample = draw_trial_sample(generator, population_by_side, sample_seed) if trial else None
    compared_entities = compared_entities_by_side(population_by_side, entity_sample)

    # 1. collect
    value_wordsets = None
    normalization_examples = None
    match generator.strategy.kind:
        case CandidateStrategyType.KEY:
            collected, hubs, confirmed_equivalences, value_wordsets, normalization_examples = get_key_candidates(
                managers, alignment_state, generator,
                partition_names_by_entity, max_entity_pairs_per_bucket,
                entity_sample=entity_sample,
            )
        case CandidateStrategyType.VECTOR:
            collected, hubs, confirmed_equivalences, normalization_examples = get_vector_candidates(
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
        population=population_by_side,
        compared_entities=compared_entities,
        collected=collected,
        selected=selected,
        signals=signals,
        confirmed_equivalences=confirmed_equivalences,
        existing_equivalences=sum(len(witnesses) for witnesses in existing_equivalences_by_entity[Side.SOURCE].values()),
        hubs=hubs,
        value_wordsets=value_wordsets,
        normalization_examples=normalization_examples
        )


def run_full_ensemble(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        run_ensemble: RunEnsemble,
        generators_by_run: dict[str, Generator]
        ) -> MergedCandidates:
    partition_names_by_entity = retrieve_partition_names(
        managers, alignment_state
        )
    strategy_results_by_run = {
        run_name: run_strategy(
            managers,
            alignment_state,
            generators_by_run[run_name],
            partition_names_by_entity
            )
        for run_name in run_ensemble.run_ids()
    }

    return merge_candidate_runs(
        run_ensemble,
        strategy_results_by_run,
        CANDIDATE_BUDGET_PER_ENTITY,
        MAX_CANDIDATES_IN_FIRST_PARETO_SUBLAYER
        )


@dataclass
class CandidateTaskState:
    kg_managers: dict[Side, KgManager]
    alignment_state: AlignmentState  # doesn't change over the task
    partition_names_by_entity: dict[Side, dict[IRI, str]] = field(default_factory=dict)
    runs: dict[str, tuple[Generator, StrategyResult]] = field(default_factory=dict)
    # how entities are shown, per side; keeps the names retrieved so far
    entity_displays: dict[Side, EntityDisplay] = field(default_factory=dict)
    finished: bool = False
    # the runs selected by finish, with roles and families; None: none selected
    run_ensemble: RunEnsemble | None = None
    rationale: str | None = None
    # the last ensemble evaluated (evaluate_run_ensemble)
    merged_candidates: MergedCandidates | None = None

    def add_run(self, generator: Generator, result: StrategyResult) -> str:
        run_id = f"run-{len(self.runs)}"
        self.runs[run_id] = (generator, result)
        return run_id


class CandidateFunctionName(StrEnum):
    TRY_STRATEGY = auto()
    RESELECT = auto()
    SHOW_CANDIDATES = auto()
    EVALUATE_RUN_ENSEMBLE = auto()
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


@func_reg.register(CandidateFunctionName.TRY_STRATEGY)
def try_strategy(state: CandidateTaskState, fn_args: dict) -> str:
    try:
        generator = Generator(**fn_args.get("generator", {}))
    except ValidationError as error:
        raise FunctionCallException(f"Invalid generator: {format_validation_error(error)}") from error

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
    return format_display_content(
        format_strategy_result(f"Result {run_id}", result, examples, state.entity_displays)
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


@func_reg.register(CandidateFunctionName.RESELECT)
def reselect(state: CandidateTaskState, fn_args: dict) -> str:
    run_id = fn_args.get("run_id")
    if run_id not in state.runs:
        raise FunctionCallException(f"Unknown run_id {run_id!r}; existing runs: {list(state.runs)}")
    try:
        selection = Selection(**fn_args.get("selection", {}))
    except ValidationError as error:
        raise FunctionCallException(f"Invalid selection object: {format_validation_error(error)}") from error

    generator, result = state.runs[run_id]
    new_generator = generator.model_copy(update={"selection": selection})
    new_result = reselect_result(result, selection)
    new_run_id = state.add_run(new_generator, new_result)

    examples = choose_examples(new_result)
    prefetch_shown_entities(examples, new_result.hubs, state.entity_displays)
    return format_display_content(
        format_strategy_result(
            f"Result {new_run_id} (run {run_id} with another selection)",
            new_result, examples, state.entity_displays,
            )
        )


def parse_run_ensemble(state: CandidateTaskState, runs: Any) -> RunEnsemble:
    """
    The ensemble of the function arguments, with run ids of tried runs.
    """
    try:
        run_ensemble = RunEnsemble.model_validate({"runs": runs})
    except ValidationError as error:
        raise FunctionCallException(f"Invalid runs: {format_validation_error(error)}") from error
    unknown_run_ids = [run_id for run_id in run_ensemble.run_ids() if run_id not in state.runs]
    if unknown_run_ids:
        raise FunctionCallException(f"Unknown run ids {unknown_run_ids}; existing runs: {list(state.runs)}")
    return run_ensemble


@func_reg.register(CandidateFunctionName.EVALUATE_RUN_ENSEMBLE)
def evaluate_run_ensemble(state: CandidateTaskState, fn_args: dict) -> str:
    """
    Merges the candidate lists of tried runs and shows the merge on the
    entities where it equals a merge of full runs.
    """
    run_ensemble = parse_run_ensemble(state, fn_args.get("runs"))
    merged = merge_candidate_runs(
        run_ensemble,
        {run_id: result for run_id, (_, result) in state.runs.items()},
        CANDIDATE_BUDGET_PER_ENTITY,
        MAX_CANDIDATES_IN_FIRST_PARETO_SUBLAYER,
        )
    state.merged_candidates = merged
    evaluations = {
        side: evaluate_merged_candidates(
            merged, side, NUM_CELL_EXAMPLES, NUM_TOO_AMBIGUOUS_EXAMPLES, NUM_SHOWN_TIED_CANDIDATES
            )
        for side in SIDES
    }
    prefetch_evaluation_entities(evaluations, merged, state.entity_displays)
    return format_display_content(Section(
        "Ensemble Evaluation",
        subsections=[
            format_run_ensemble(run_ensemble),
            *(
                format_ensemble_evaluation_of_side(side, evaluations[side], merged, state.entity_displays)
                for side in SIDES
                )
            ]
        ))


@func_reg.register(CandidateFunctionName.FINISH)
def finish(state: CandidateTaskState, fn_args: dict) -> str:
    """
    Selects the ensemble of runs whose generators are run in full later,
    with a rationale; no runs: no strategy is good.
    """
    runs = fn_args.get("runs")
    rationale = fn_args.get("rationale")
    if not isinstance(runs, list):
        raise FunctionCallException("runs must be a list of runs (it may be empty)")
    if not rationale:
        raise FunctionCallException("A rationale is required")
    state.run_ensemble = parse_run_ensemble(state, runs) if runs else None
    state.rationale = rationale
    state.finished = True
    if state.run_ensemble is None:
        return "Selected runs: none."
    return f"Selected runs: {format_run_ensemble(state.run_ensemble)}."


class CandidateListKind(StrEnum):
    """
    Which candidate lists show_candidates picks, matching the signals.
    """
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
        result: StrategyResult,
        side: Side,
        kind: CandidateListKind,
        num_entities: int,
        ) -> tuple[list[SelectedCandidates], int]:
    """
    A random sample (fixed seed: the same call shows the same entities) of
    the entities of the side whose list is of the kind, and how many there are.
    """
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


@func_reg.register(CandidateFunctionName.SHOW_CANDIDATES)
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
    content: list[SectionContent] = []
    if entities:
        shown, missing = find_listed_entities(result, side, [IRI(entity) for entity in entities])
        title = f"Run {run_id}, the given {side} entities"
        if missing:
            content.append(
                "without a candidate list (no candidate, already matched, or not in the population): "
                + ", ".join(missing))
    else:
        shown, num_matching = choose_listed_entities(result, side, kind, num_entities)
        title = f"Run {run_id}, {side} entities of kind {kind}: {len(shown)} of {num_matching} shown"
        if result.signals[side].sampled_side is not None:
            title += " (trial run on a sample)"

    prefetch_shown_entities(shown, [], state.entity_displays)
    content.extend(format_selected_candidates(selected, state.entity_displays) for selected in shown)
    return format_display_content(Section(title, content or ["no such entity"]))


# ---------------------------------------------------------------- formatting for the LLM
# Every function returns display content (foundation/rendering.py: Section, Item).
NUM_EXAMPLE_ENTITIES = 3  # entities with candidates shown per side in a result
NUM_SHOWN_HUBS = 3        # largest hubs shown in a result
NUM_SHOWN_MATCHES = 2     # matches shown per candidate
NUM_NORMALIZATION_EXAMPLES = 3  # values shown per side after the transforms and the normalization


def display_entity(entity_displays: dict[Side, EntityDisplay], side: Side, entity: IRI) -> str:
    """An entity of the side as one line (its names must have been prefetched,
    prefetch_shown_entities)."""
    return entity_displays[side].show(entity)


def format_share(share: float) -> str:
    return f"{share:.0%}"


def format_found_again(result: StrategyResult) -> str | None:
    """
    The existing equivalences the run finds again. Their entities are not in
    the population: the coverage and the other signals leave them out.
    """
    if not result.existing_equivalences:
        return None
    line = (
        f"Existing equivalences found again: {len(result.confirmed_equivalences)} of "
        f"{result.existing_equivalences}. They are left out of the result: entities that already have an "
        "equivalence get no candidate list and are not counted in the coverage or any other signal, "
        "so a strategy can find many known equivalences and still show a low coverage."
        )
    if result.signals[Side.SOURCE].sampled_side is not None:
        line += (" Not measured on the sampled side: its sample holds only entities without an equivalence, "
                 "so this number is no estimate of recall here.")
    return line


def format_signals(signals: StrategySignals) -> Item:
    children = [
        Item(f"coverage: {format_share(signals.coverage)}"),
        Item(f"mean list length: {signals.mean_list_length:.1f}"),
        Item(f"ambiguous at top rank: {format_share(signals.ambiguous_at_top_rank)}"),
        Item(
            f"unique matches: {format_share(signals.unique)}"
            + (" (upper estimate)" if signals.sampled_side == signals.side else "")
            )
    ]
    if signals.already_matched_at_top_rank:
        children.append(
            Item(
                f"already matched at top rank: {format_share(signals.already_matched_at_top_rank)}"
                )
            )

    if signals.sampled_side == signals.side:
        sample_note = ", random sample"
    elif signals.sampled_side is not None:
        sample_note = "; distorted: sees only the sampled counterparts"
    else:
        sample_note = ""
    return Item(
        f"{signals.side} ({signals.population_size} entities{sample_note}):",
        children
        )


def format_relations(relations: tuple[str, str]) -> str:
    """The projections of a neighbour match or hub: one if both sides use the same."""
    source_relation, target_relation = relations
    return source_relation if source_relation == target_relation else f"{source_relation} ~ {target_relation}"


def format_hub(hub: Hub, entity_displays: dict[Side, EntityDisplay]) -> str:
    """
    A hub with the size of its bucket.
    """
    match hub:
        case KeyHub():
            shared = f'{hub.level} key "{hub.key}"'
        case VectorHub():
            shared = f'texts "{hub.source_text}" ~ "{hub.target_text}"'
        case NeighbourHub():
            source_witness, target_witness = hub.witness
            shared = (
                f"witness {display_entity(entity_displays, Side.SOURCE, source_witness)} = "
                f"{display_entity(entity_displays, Side.TARGET, target_witness)} "
                f"over {format_relations(hub.relations)}"
                )
        case _:
            raise NotImplementedError(f"format_hub does not support {type(hub).__name__}")
    return f"{shared}: {hub.num_source_entities} source x {hub.num_target_entities} target entities"


def format_hubs(
        hubs: Sequence[Hub], entity_displays: dict[Side, EntityDisplay]
        ) -> list[SectionContent] | None:
    if not hubs:
        return
    largest_hubs = sorted(
        hubs,
        key=lambda hub: hub.num_source_entities * hub.num_target_entities,
        reverse=True
        )
    return [
        f"{len(hubs)} hubs (too many entity pairs, so they give no candidates), the largest:",
        *(Item(format_hub(hub, entity_displays)) for hub in largest_hubs[:NUM_SHOWN_HUBS]),
    ]


def format_match(
        match: Match, own_side: Side, entity_displays: dict[Side, EntityDisplay]
        ) -> str:
    """
    One match of a candidate, the entity's own value first.
    """
    source_role = match.source_rows[0].role_name
    target_role = match.target_rows[0].role_name
    own_role, candidate_role = (
        (source_role, target_role) if own_side == Side.SOURCE
        else (target_role, source_role)
        )
    roles = f"roles: ({own_role} - {candidate_role})"
    match match:
        case KeyMatch():
            return f'matched by key "{match.key}" on normalization level "{match.level}", {roles}'
        case VectorMatch():
            own_text, candidate_text = (
                (match.source_text, match.target_text) if own_side == Side.SOURCE
                else (match.target_text, match.source_text)
                )
            return f'matched by texts: "{own_text}" ~ "{candidate_text}", cosine {match.cosine:.2f}, {roles}'
        case NeighbourMatch():
            source_witness, target_witness = match.witness
            witness = (
                f"{display_entity(entity_displays, Side.SOURCE, source_witness)} = "
                f"{display_entity(entity_displays, Side.TARGET, target_witness)}"
                )
            bucket = match.num_source_entities * match.num_target_entities
            witness_shared_by = (
                f", shared by {match.num_source_entities} source x {match.num_target_entities} "
                f"target entities" if bucket > 1 else ""
                )
            return f"matched by witness {witness} over {format_relations(match.relations)}{witness_shared_by}"
        case _:
            raise NotImplementedError(
                f"format_match does not support {type(match).__name__}"
                )


def format_candidate(
        candidate: Candidate,
        mutual_candidates: frozenset[IRI],
        unique_candidate: IRI | None,
        entity_displays: dict[Side, EntityDisplay],
        ) -> Item:
    """
    The candidate entity with its marks and its best matches.
    """
    candidate_side = Side.other_side(candidate.own_side)
    marks = [
        mark for mark, applies in [
            ("unique match", candidate.candidate_entity == unique_candidate),
            ("mutual match", candidate.candidate_entity in mutual_candidates),
            ("already matched before", candidate.candidate_already_matched),
        ] if applies
    ]
    group_names = sorted(candidate.entity_pair.entity_groups()[candidate_side])
    groups = ", ".join(f"'{group_name}'" for group_name in group_names)
    title = f"{display_entity(
        entity_displays, candidate_side, candidate.candidate_entity
        )} from group{'s' if len(group_names) > 1 else ''} {groups}"
    if marks:
        title += f" [{', '.join(marks)}]"
    entity_pair = candidate.entity_pair
    if isinstance(entity_pair, NeighbourEntityPair) and entity_pair.witness_weighting != "count":
        title += f", witness score {entity_pair.witness_score():.2f}"
    matches = filter_informative_matches(entity_pair.matches)
    shown_matches = [
        Item(format_match(match, candidate.own_side, entity_displays))
        for match in matches[:NUM_SHOWN_MATCHES]
        ]
    if len(matches) > NUM_SHOWN_MATCHES:
        shown_matches.append(
            Item(f"+{len(matches) - NUM_SHOWN_MATCHES} more matches")
            )
    return Item(title, shown_matches)


def format_rivals(entity_candidates: EntityCandidates) -> list[Item]:
    """
    The counterparts that are no listed candidates, in summary.
    """
    items: list[Item] = []
    for title, counts in [
        ("competing counterparts outside the population, by group",
         entity_candidates.num_counterparts_outside_population_by_group),
        ("counterparts in non-competing partitions, by partition",
         entity_candidates.num_non_competing_counterparts_by_partition),
    ]:
        if counts:
            items.append(Item(
                f"{title}: " + ", ".join(f"{name} {count}"
                                         for name, count in counts.most_common())
                ))
    return items


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


def format_selected_candidates(
    selected: SelectedCandidates, entity_displays: dict[Side, EntityDisplay]
        ) -> Item:
    """
    The selection of one entity: its tie groups in rank order (rank 1 = best),
    each with its candidates, then the counterparts that are not listed.
    """
    collected = selected.collected
    items: list[Item] = []

    for rank, tie_group in enumerate(selected.tie_rank_groups, start=1):
        num_rivals_outside = collected.num_counterparts_outside_population_by_rank[
            tie_group[0].rank_key
            ]
        size = len(tie_group) + num_rivals_outside
        candidate_items = [
            format_candidate(
                candidate,
                selected.mutual_candidates,
                selected.unique_candidate,
                entity_displays
                )
            for candidate in tie_group
            ]
        if size == 1:
            # one candidate on this rank: the rank in its own line
            items.append(Item(
                f"candidate on rank {rank}: {candidate_items[0].text}",
                candidate_items[0].children
                ))
            continue
        title = f"candidates on rank {rank}" + (f" (tie of {size}")
        if num_rivals_outside:
            title += f", {num_rivals_outside} of them outside the population"
        title += "):"
        items.append(Item(title, candidate_items))
    if selected.dropped_tie_group_rank is not None:
        dropped = (sum(candidate.rank_key == selected.dropped_tie_group_rank for candidate in collected.candidates)
                   + collected.num_counterparts_outside_population_by_rank[selected.dropped_tie_group_rank])
        items.append(Item(f"next tie group dropped: its {dropped} entities do not fit into top_k"))
    items.extend(format_rivals(collected))
    side = collected.side
    return Item(f"{side} {display_entity(entity_displays, side, collected.entity)}", items)


def shown_entities(
        examples: list[SelectedCandidates], hubs: Sequence[Hub]
        ) -> dict[Side, set[IRI]]:
    """
    All entities a result may show, per side: the example entities, their
    selected candidates, and the witnesses of the candidates and hubs (more
    than are shown: one query per side instead of one per entity).
    """
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
        examples: list[SelectedCandidates],
        hubs: Sequence[Hub],
        entity_displays: dict[Side, EntityDisplay]
        ):
    for side, entities in shown_entities(examples, hubs).items():
        entity_displays[side].prefetch(entities)


def choose_examples(result: StrategyResult) -> list[SelectedCandidates]:
    """
    The example entities of a result: per side a few with a selected candidate.
    """
    return [
        selected for side in SIDES
        for selected in choose_listed_entities(
            result, side, CandidateListKind.RANDOM, NUM_EXAMPLE_ENTITIES
            )[0]
    ]


def format_strategy_result(
        title: str,
        result: StrategyResult,
        examples: list[SelectedCandidates],
        entity_displays: dict[Side, EntityDisplay],
        ) -> Section:
    """
    The LLM view of a run: the signals per side, the existing equivalences found
    again, the largest hubs, and the example entities with their selection
    (choose_examples; prefetch_shown_entities first). The full result stays in
    the task state (show_candidates).
    """
    found_again = format_found_again(result)
    subsections = [
        Section(
            "Signals",
            [format_signals(result.signals[side]) for side in SIDES] + ([found_again] if found_again else [])
            )
        ]
    if result.hubs:
        hubs_content = format_hubs(result.hubs, entity_displays)
        if hubs_content:
            subsections.append(Section("Hubs", hubs_content))

    if result.normalization_examples:
        subsections.append(
            Section(
                "Values after the transforms and the normalization "
                "(raw value → transformed → key on the finest level)",
                [
                    Item(
                        f"{side}: " + "; ".join(
                            f"{value!r} → {transformed!r} → {key!r}"
                            for value, transformed, key in side_examples)
                        )
                    for side, side_examples in result.normalization_examples.items()
                    ]
            )
        )

    subsections.append(Section(
        "Example entities",
        [format_selected_candidates(selected, entity_displays) for selected in examples]
        or ["no entity has a candidate"]
        ))
    return Section(title, subsections=subsections)


# ---------------------------------------------------------------- ensemble evaluation

NUM_CELL_EXAMPLES = 8           # largest Venn cells shown with an example pair
NUM_TOO_AMBIGUOUS_EXAMPLES = 5  # too ambiguous entities shown
NUM_SHOWN_TIED_CANDIDATES = 3   # candidates of their first sublayer shown


def format_run_ensemble(run_ensemble: RunEnsemble) -> Section:
    return Section("Runs", [
        Item(f"{run.run_id}: {run.role}s, family {run.family_name}")
        for run in sorted(run_ensemble.runs, key=lambda run: run.run_id)
    ])


def format_venn_cell(cell: frozenset[str]) -> str:
    return "{" + ", ".join(sorted(cell)) + "}"


def evaluation_entities(evaluation: EnsembleEvaluation, merged: MergedCandidates) -> dict[Side, set[IRI]]:
    """The entities an evaluation shows, per side: the example entities, the
    candidates of their lists and the shown tied candidates."""
    other_side = Side.other_side(evaluation.side)
    entities_by_side: dict[Side, set[IRI]] = {side: set() for side in SIDES}
    for _, entity, _ in evaluation.cell_examples:
        entities_by_side[evaluation.side].add(entity)
        entities_by_side[other_side].update(
            flatten_layers(merged.pareto_layers_by_entity[(evaluation.side, entity)]))
    for entity, _, tied_candidates in evaluation.too_ambiguous_examples:
        entities_by_side[evaluation.side].add(entity)
        entities_by_side[other_side].update(tied_candidates)
    return entities_by_side


def prefetch_evaluation_entities(
        evaluations: dict[Side, EnsembleEvaluation | None], merged: MergedCandidates,
        entity_displays: dict[Side, EntityDisplay],
        ) -> None:
    """Retrieves the names of all entities the evaluations show, one query per side."""
    entities_by_side: dict[Side, set[IRI]] = {side: set() for side in SIDES}
    for evaluation in evaluations.values():
        if evaluation is not None:
            for side, entities in evaluation_entities(evaluation, merged).items():
                entities_by_side[side].update(entities)
    for side, entities in entities_by_side.items():
        entity_displays[side].prefetch(entities)


def format_merged_list(
        side: Side,
        entity: IRI,
        merged: MergedCandidates,
        entity_displays: dict[Side, EntityDisplay],
        ) -> list[Item]:
    """
    The merged list of an entity: per candidate its rank in the list (tied
    candidates of a sublayer share it) and its ranks in the runs.
    """
    other_side = Side.other_side(side)
    ranks_by_candidate = merged.candidate_rank_index[(side, entity)]
    items: list[Item] = []
    list_rank = 0
    for layer in merged.pareto_layers_by_entity[(side, entity)]:
        for sublayer in layer:
            list_rank += 1
            for candidate in sublayer:
                run_ranks = ", ".join(
                    f"{run_id}: {rank}"
                    for run_id, rank in sorted(ranks_by_candidate[candidate].items())
                    )
                items.append(Item(
                    f"{list_rank}. {entity_displays[other_side].show(candidate, SHORT_VIEW)} "
                    f"(ranks: {run_ranks})"
                    ))
    return items


def format_ensemble_evaluation_of_side(
        side: Side,
        evaluation: EnsembleEvaluation | None,
        merged: MergedCandidates,
        entity_displays: dict[Side, EntityDisplay],
        ) -> Section:
    """
    The LLM view of the merge on one side (prefetch_evaluation_entities first).
    """
    title = f"Evaluation of {side} side"
    if evaluation is None:
        return Section(title, [
            f"Not measurable: a run compared only a sample of the other side, so the lists "
            f"of the {side} side lack candidates."
            ])
    if evaluation.num_evaluated == 0:
        return Section(title, [
            f"No {side} entity was compared by every run whose population holds it."
            ])

    items: list[Item] = [
        Item(f"with a candidate list: {format_share(evaluation.num_with_list / evaluation.num_evaluated)}"),
        Item(
            f"too ambiguous: {format_share(evaluation.num_too_ambiguous / evaluation.num_evaluated)} "
            f"(more than {MAX_CANDIDATES_IN_FIRST_PARETO_SUBLAYER} candidates tied on the first rank: no list)"
            ),
        Item(f"without candidates: {format_share(evaluation.num_without_candidates / evaluation.num_evaluated)}"),
        Item(
            f"lists cut by the budget of {CANDIDATE_BUDGET_PER_ENTITY}: "
            f"{format_share(evaluation.num_cut_by_budget / evaluation.num_evaluated)}"
            ),
        ]
    if evaluation.num_with_list:
        items.append(Item(
            "list length (share of the lists):",
            [
                Item(f"{name}: {format_share(count / evaluation.num_with_list)}")
                for name, count in evaluation.list_length_counts.items()
                ]
            ))
        items.append(Item(
            "pairs by the families of the runs that selected them (Venn cells)",
            [
                Item(f"{format_venn_cell(cell)}: {num_pairs} pairs")
                for cell, num_pairs in evaluation.pairs_by_cell
                ]
            ))
        items.append(Item(
            "examples, a random pair of each of the largest cells",
            [
                Item(
                    f"{format_venn_cell(cell)}: {entity_displays[side].show(entity, SHORT_VIEW)}",
                    format_merged_list(side, entity, merged, entity_displays)
                    )
                for cell, entity, candidate in evaluation.cell_examples
                ]
            ))
    if evaluation.too_ambiguous_examples:
        other_side = Side.other_side(side)
        items.append(Item(
            "too ambiguous, e.g.",
            [
                Item(
                    f"{entity_displays[side].show(entity, SHORT_VIEW)}: {num_tied} candidates tied on the first "
                    f"rank, e.g. " + ", ".join(entity_displays[other_side].show(candidate, SHORT_VIEW)
                                               for candidate in tied_candidates)
                    )
                for entity, num_tied, tied_candidates in evaluation.too_ambiguous_examples
                ]
            ))
    return Section(title, [
        f"{evaluation.num_evaluated} {side} entities evaluated (those compared by every run "
        f"whose population holds them).",
        *items,
        ])


# ---------------------------------------------------------------- start message

NUM_EXAMPLE_EQUIVALENCES = 5


def count_group_entities(
        manager: KgManager,
        kg_profile: KgProfile,
        matched_entities: Collection[IRI]
        ) -> dict[str, tuple[int, int]]:
    """
    Per group of the profile: its entities and those already matched (an
    entity in several groups counts in each).
    """
    groups_by_entity = retrieve_groups_by_entity(
        manager, profile_groups_to_grouping(kg_profile), kg_profile.entity_space
        )
    group_sizes = Counter(
        name for group_names in groups_by_entity.values() for name in group_names
        )
    matched_entities_per_group = Counter(
        group_name for entity in set(matched_entities) & groups_by_entity.keys()
        for group_name in groups_by_entity[entity]
    )
    return {
        group_name: (group_size, matched_entities_per_group.get(group_name, 0))
        for group_name, group_size in group_sizes.items()
        }


def format_group_counts(state: CandidateTaskState, side: Side) -> list[SectionContent]:
    kg_profile = state.alignment_state.get_kg_profile(side)
    matched_entities = state.alignment_state.aligned_entities[side]
    counts = count_group_entities(
        state.kg_managers[side], kg_profile, matched_entities
        )
    excluded = set(kg_profile.excluded_group_names())
    content: list[SectionContent] = []
    if excluded:
        content.append(
            "Entities of excluded groups are left out everywhere: they get "
            "no candidates and are no candidates or rivals, also when "
            "they belong to other groups too. The counts are without them."
            )
    content.append(Table(
        ("group", "contained entities", "already matched", "not matched", "excluded"),
        [
            (
                group_name, str(size), str(matched), str(size - matched),
                "yes" if group_name in excluded else ""
                )
            for group_name, (size, matched) in sorted(
                counts.items(), key=lambda item: -item[1][0]
                )
            ]
        ))
    return content


def format_knowledge_graph(state: CandidateTaskState, side: Side) -> Section:
    """
    The knowledge graph with its profile, the profile's parts directly under
    the knowledge graph (no heading 'Profile' of its own), then the group counts.
    """
    kg_profile = state.alignment_state.get_kg_profile(side)
    manager = state.kg_managers[side]
    profile = describe_profile(kg_profile, findings=True)
    return Section(
        f"{manager.kg} ({side})",
        [manager.description or "", *profile.content],
        [*profile.subsections, Section("Entities per group", format_group_counts(state, side))],
        )


def format_alignment_status(state: CandidateTaskState) -> Section:
    equivalences = state.alignment_state.equivalences
    if not equivalences:
        return Section("Equivalences found so far", ["none"])
    partitions = state.partition_names_by_entity
    by_partition_pair = Counter(
        (partitions[Side.SOURCE].get(equivalence.entity_source, "?"),
         partitions[Side.TARGET].get(equivalence.entity_target, "?"))
        for equivalence in equivalences
    )
    examples = random.Random(0).sample(equivalences, min(NUM_EXAMPLE_EQUIVALENCES, len(equivalences)))
    for side in SIDES:
        state.entity_displays[side].prefetch(
            equivalence.entity_source if side == Side.SOURCE else equivalence.entity_target
            for equivalence in examples
        )
    return Section("Equivalences found so far", [
        Item(
            f"{len(equivalences)} equivalences, by partition pair (source - target)",
            [
                Item(f"{source_partition} - {target_partition}: {count}")
                for (source_partition, target_partition), count in by_partition_pair.most_common()
                ]
            ),
        Item(
            "examples",
            [
                Item(
                    f"{state.entity_displays[Side.SOURCE].show(equivalence.entity_source)} = "
                    f"{state.entity_displays[Side.TARGET].show(equivalence.entity_target)}"
                    )
                for equivalence in examples
                ]
            ),
        ])


def format_start_message(state: CandidateTaskState) -> str:
    knowledge_graphs = Section(
        "Knowledge graphs",
        subsections=[format_knowledge_graph(state, side) for side in SIDES],
        )
    return (
        format_display_content(knowledge_graphs)
        + "\n\n" + format_display_content(format_alignment_status(state))
    )

# ---------------------------------------------------------------- function definitions


GROUP_REFS_SCHEMA = {
    "type": "array",
    "items": {"anyOf": [
        {"type": "string"},
        {"type": "object", "properties": {
            "name": {"type": "string"},
            "membership_sparql": {"type": "string", "description": "SPARQL group pattern over ?s"},
        }, "required": ["name", "membership_sparql"]},
    ]},
    "description": (
        f"Groups of the profile by name, {POPULATION_REST!r} for all groups that are not excluded, or a group "
        "defined inline for this strategy (a new name and a SPARQL pattern over ?s). "
        "Entities of excluded groups are never compared."
    ),
}

VALUE_SOURCE_SCHEMA = {
    "anyOf": [
        {"type": "string",
         "description": "One name: a role of the profile (names, synonyms, definitions, codes, cross_references), "
                        "a structure relation of the profile, or a relation kind (" + ", ".join(StructureRelationKind)
                        + "; all relations of that kind); '^' before a relation or kind for the other direction; "
                        "or a path of these joined by '/' (e.g. 'subsumption/names': the names of the broader "
                        "entities)."},
        {"type": "object", "properties": {
            "predicates": {"type": "array", "items": {"type": "string"}},
            "patterns": {"type": "array", "items": {"type": "string"},
                         "description": "SPARQL patterns over ?s (the entity) and ?o (the value)"},
        }, "description": "One own pattern instead of a name."},
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
                           "candidates. Default: not yet matched entities inside the name spaces on both sides.",
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
                    "description": "The values compared, one per side: key and vector compare them directly, "
                                   "neighbours compares the entities they lead to by their equivalences. "
                                   "Default: names (key, vector), subsumption (neighbours). For several, try "
                                   "several runs.",
                    "properties": {
                        "source": VALUE_SOURCE_SCHEMA,
                        "target": {**VALUE_SOURCE_SCHEMA, "description": "Omit: like the source."},
                    },
                },
                "normalization": {
                    "type": "object",
                    "description": "key and vector: how values are normalized before they are compared.",
                    "properties": {
                        "transforms": {
                            "type": "array",
                            "description": "String transforms applied to every value before the levels. Use them only for what the "
                            "standard normalization levels cannot do: case, word splitting (camelCase, '-', '_'), word order, plural and "
                            "inflection are all done by 'levels'.",
                            "items": {
                                "type": "object", "properties": {
                                    "operation": {"type": "string", "enum": [*TRANSFORM_OPERATIONS, REGEX_REPLACE]},
                                    "pattern": {"type": "string", "description": "regex_replace only: a Python re pattern"},
                                    "replacement": {
                                        "type": "string",
                                        "description": "regex_replace only: Python re syntax; groups as \\1, not $1."
                                        },
                                    }, "required": ["operation"]
                            }
                        },
                        "levels": {
                            "type": "array", "items": {"type": "string", "enum": list(NORMALIZATION_LEVELS)},
                            "description": "Levels from fine to coarse, in this order; a key pair ranks by "
                                           "its finest shared level. vector: one level, words (default), or "
                                           "plural or inflection for idf_tokens when the KGs inflect names "
                                           "differently.",
                        },
                    },
                },
                "vectorizer": {
                    "type": "string", "enum": list(get_args(Vectorizer.__value__)),
                    "description": "vector: idf_tokens (rare words count more), char_3grams (spelling "
                                   "variants), embedding (meaning, multilingual). Default idf_tokens.",
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

ENSEMBLE_RUNS_SCHEMA = {
    "type": "array",
    "description": "Tried runs whose candidate lists are merged per entity. The candidates of the generate "
                   "runs form the lists (a candidate ranked better than another in every generate run comes "
                   "first); corroborate runs add no candidates, they only order tied candidates. Runs of one "
                   "family count as one kind of evidence when the pairs are grouped (e.g. two key runs on "
                   "different groups).",
    "items": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "role": {"type": "string", "enum": list(get_args(RunRole.__value__)),
                     "description": "generate (default) or corroborate."},
            "family": {"type": "string",
                       "description": "Name of the kind of evidence of the run; runs whose evidence is "
                                      "alike (e.g. on the same names) share one family. Default: the run id."},
        },
        "required": ["run_id"],
    },
}

FINISH_DEFINITION = {
    "name": CandidateFunctionName.FINISH,
    "description": "Ends the task: selects the ensemble of tried runs whose strategies are run in full later "
                   "(on all entities, not a sample) and merged. Call it exactly once, at the end.",
    "parameters": {
        "type": "object",
        "properties": {
            "runs": {**ENSEMBLE_RUNS_SCHEMA, "description": "The selected runs; empty if no strategy is good. "
                     + ENSEMBLE_RUNS_SCHEMA["description"]},
            "rationale": {"type": "string",
                          "description": "Which strategies for which groups, why, and which candidates are risky."},
        },
        "required": ["runs", "rationale"],
    },
}


def candidate_functions() -> list[dict]:
    return [
        {
            "name": CandidateFunctionName.TRY_STRATEGY,
            "description": (
                "Tries a candidate generator (population, strategy, selection) in a trial run, on a sample "
                "of the larger side if it is large, and returns the signals per side, the existing "
                "equivalences it finds again, its hubs (values shared by too many entities, which give "
                "no candidates) and example candidate lists. The run gets an id (run-N)."
            ),
            "parameters": {"type": "object", "properties": {"generator": GENERATOR_SCHEMA}, "required": ["generator"]},
        },
        {
            "name": CandidateFunctionName.RESELECT,
            "description": "Selects the candidates of a tried run again with another selection, without "
                           "running the strategy again; gives a new run.",
            "parameters": {"type": "object", "properties": {
                "run_id": {"type": "string"}, "selection": SELECTION_SCHEMA,
            }, "required": ["run_id", "selection"]},
        },
        {
            "name": CandidateFunctionName.SHOW_CANDIDATES,
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
        {
            "name": CandidateFunctionName.EVALUATE_RUN_ENSEMBLE,
            "description": (
                f"Merges the candidate lists of tried runs per entity (at most {CANDIDATE_BUDGET_PER_ENTITY} "
                f"candidates, more only if more than that are tied on the first rank) and shows the merge "
                f"on the entities every run of their population compared: how many get a list, how long "
                f"the lists are, the pairs per combination of families that selected them, and examples."
            ),
            "parameters": {"type": "object", "properties": {"runs": ENSEMBLE_RUNS_SCHEMA}, "required": ["runs"]},
        },
        FINISH_DEFINITION,
    ]


def system_information(config: GraspConfig) -> str:
    info = load_prompt("candidate_task.txt")
    info = add_equivalence_condition(info, config)
    task_instructions = load_task_instructions(config)
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
        alignment_state = resolve_alignment_state(input)
        kg_managers = managers_by_side(self, alignment_state.source_kg_name, alignment_state.target_kg_name)
        self.state = CandidateTaskState(
            kg_managers=kg_managers,
            alignment_state=alignment_state,
            partition_names_by_entity=retrieve_partition_names(kg_managers, alignment_state),
            entity_displays={
                side: EntityDisplay(kg_managers[side], alignment_state.get_kg_profile(side))
                for side in SIDES
            },
        )
        return format_start_message(self.state) + "\n\nNo strategy has been tried yet."

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return [
            BLANK_NODE_RULE,
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
        function = func_reg.handlers.get(fn_name)
        if function is None:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return function(self.state, fn_args)

    def done(self, fn_name: str) -> bool:
        return fn_name == CandidateFunctionName.FINISH and self.state.finished

    @property
    def default_input_field(self) -> str | None:
        return None

    def output(self, messages: list[Message]) -> dict:
        state = self.state
        return CandidateStageResult(
            source_kg=state.kg_managers[Side.SOURCE].kg,
            target_kg=state.kg_managers[Side.TARGET].kg,
            finished=state.finished,
            rationale=state.rationale,
            run_ensemble=state.run_ensemble,
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
