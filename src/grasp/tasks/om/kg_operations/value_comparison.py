from collections import defaultdict
from typing import Callable, Iterable

from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.entity_pairs import EntityCandidates, EntityKeyPair, KeyCandidate, KeyMatch, OverfullKey
from grasp.tasks.om.data_models.scope import Normalization, StringTransform, apply_string_transforms
from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.text_normalization import NORMALIZATION_LEVELS, NormalizationLevel
from grasp.tasks.om.foundation.types import IRI, SIDES, Side

type ValueIndex = dict[str, list[ValueRow]]
type EntityPairIndex = dict[tuple[IRI, IRI], EntityKeyPair]


def distinct_entities(rows: list[ValueRow]) -> set[IRI]:
    return {row.entity for row in rows}


def group_value_rows_by_entity(
    rows: list[ValueRow]
        ) -> dict[IRI, list[ValueRow]]:
    rows_by_entity: dict[IRI, list[ValueRow]] = {}
    for row in rows:
        rows_by_entity.setdefault(row.entity, []).append(row)
    return rows_by_entity


def string_normalize_index_keys(
        transformed_index: ValueIndex,
        string_normalization_function: Callable[[str], str]
        ) -> ValueIndex:
    """
    Always from the index right after the transforms,
    never from another level: the levels cannot be chained
    (e.g. case would lose the CamelCase that words splits).
    """
    new_index: ValueIndex = {}

    for key, value_rows in transformed_index.items():
        normalized_key = string_normalization_function(key)
        if not normalized_key:
            continue
        new_index.setdefault(normalized_key, []).extend(value_rows)

    return new_index


def build_transformed_value_index(
        value_rows: Iterable[ValueRow],
        transforms: list[StringTransform]
        ) -> ValueIndex:

    index: ValueIndex = {}
    for row in value_rows:
        key = apply_string_transforms(transforms, row.raw_projection_value)
        if not key:
            continue
        index.setdefault(key, []).append(row)

    return index


def build_side_indices(
    group_value_rows: Iterable[ValueRow],
    normalization: Normalization
) -> dict[NormalizationLevel, ValueIndex]:
    """
    Returns: {normalization_level: {normalized_key: [ValueRow, ...]}}
    """
    transformed_index = build_transformed_value_index(group_value_rows, normalization.transforms)
    indices_by_level: dict[NormalizationLevel, ValueIndex] = {}
    for level in normalization.levels:
        indices_by_level[level] = string_normalize_index_keys(transformed_index, NORMALIZATION_LEVELS[level])
    return indices_by_level


def intersect_level_indices(
        source_indices: dict[NormalizationLevel, ValueIndex],
        target_indices: dict[NormalizationLevel, ValueIndex],
        max_bucket_entities: int | None = None
        ) -> tuple[EntityPairIndex, list[OverfullKey]]:
    """
    Pairs of source and target entities sharing a key, on each level present on
    both sides, with all their shared keys (KeyMatch: the rows of just these two
    entities under the key). A key with more distinct entities than
    max_bucket_entities on a side gives no pairs, only an OverfullKey.
    """
    entity_pairs: EntityPairIndex = {}
    overfull_keys: list[OverfullKey] = []

    for level, source_index in source_indices.items():
        target_index = target_indices.get(level)
        if target_index is None:
            continue

        for key, source_value_rows in source_index.items():
            target_value_rows = target_index.get(key)
            if target_value_rows is None:
                continue

            rows_by_source_entity = group_value_rows_by_entity(source_value_rows)
            rows_by_target_entity = group_value_rows_by_entity(target_value_rows)
            if (max_bucket_entities is not None
                    and max(len(rows_by_source_entity), len(rows_by_target_entity)) > max_bucket_entities):
                overfull_keys.append(
                    OverfullKey(
                        level=level,
                        key=key,
                        num_source_entities=len(rows_by_source_entity),
                        num_target_entities=len(rows_by_target_entity)
                    )
                )
                continue

            for source_entity, source_rows in rows_by_source_entity.items():
                for target_entity, target_rows in rows_by_target_entity.items():
                    entity_pairs.setdefault(
                        (source_entity, target_entity),
                        EntityKeyPair(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            key_matches=[]
                            )
                    ).append_key_match(
                        KeyMatch(
                            level=level,
                            key=key,
                            source_rows=tuple(source_rows),
                            target_rows=tuple(target_rows)
                            )
                    )

    return entity_pairs, overfull_keys


# (side, entity, level) -> the entity's counterparts on that level whose
# partition competes with its own (in the population or not)
type CompetingCounterparts = dict[tuple[Side, IRI, NormalizationLevel], set[IRI]]


def entity_pairs_by_entity(
        entity_pairs: EntityPairIndex,
        ) -> dict[tuple[Side, IRI], list[EntityKeyPair]]:
    """
    Returns an index mapping from an entity (with its side: the same IRI may
    occur in both KGs) to all pair-objects it is involved in. The number of
    entries under one entity is the number of entities of the other side it
    matches, on any level.
    """
    entity_pairs_per_entity: dict[tuple[Side, IRI], list[EntityKeyPair]] = {}
    for entity_pair in entity_pairs.values():
        for side in SIDES:
            entity_pairs_per_entity.setdefault(
                (side, entity_pair.get_entity(side)), []
                ).append(entity_pair)
    return entity_pairs_per_entity


def partitions_compete(
        alignment_state: AlignmentState,
        entity_key_pair: EntityKeyPair,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        ) -> bool | None:
    """Whether the partitions of the two entities compete; None if an entity has no
    partition (uncategorized: not aligned)."""
    source_partition = partition_names_by_entity[Side.SOURCE].get(entity_key_pair.source_entity)
    target_partition = partition_names_by_entity[Side.TARGET].get(entity_key_pair.target_entity)
    if source_partition is None or target_partition is None:
        return None
    return alignment_state.competes(Side.SOURCE, source_partition, Side.TARGET, target_partition)


def competing_counterparts_by_level(
        alignment_state: AlignmentState,
        entity_pairs: EntityPairIndex,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        ) -> CompetingCounterparts:
    """
    Counts, for every entity and every level, which entities of the OTHER side share
    a key with it on that level (its counterparts). Only counterparts whose partition
    competes with the entity's partition are counted; whether they are in the
    population does not matter.

    Example: on "case", s1 (source) shares a key with t1, t2 and t3, so the entry
    (source, s1, case) is {t1, t2, t3}.
    """
    counterparts: CompetingCounterparts = defaultdict(set)
    for entity_key_pair in entity_pairs.values():
        if not partitions_compete(alignment_state, entity_key_pair, partition_names_by_entity):
            continue  # does not compete, or an entity without a partition
        for level in entity_key_pair.levels():
            counterparts[(Side.SOURCE, entity_key_pair.source_entity, level)].add(entity_key_pair.target_entity)
            counterparts[(Side.TARGET, entity_key_pair.target_entity, level)].add(entity_key_pair.source_entity)
    return counterparts


def build_key_candidate(
        entity_key_pair: EntityKeyPair,
        side: Side,
        counterparts: CompetingCounterparts,
        ) -> KeyCandidate:
    """
    The candidate of the entity on `side` (the other entity of the pair), complete
    in one go: everything that depends on other pairs comes from `counterparts`.
    """
    other_side = Side.other_side(side)
    entity, candidate = entity_key_pair.get_entity(side), entity_key_pair.get_entity(other_side)
    levels: list[NormalizationLevel] = entity_key_pair.levels()  # fine to coarse
    # other counterparts of the candidate = entities of the entity's own side that
    # also match the candidate (the entity itself is one of its counterparts: - 1)
    rivals_of_candidate_by_level: dict[NormalizationLevel, int] = {
        level: len(counterparts.get((other_side, candidate, level), set())) - 1 for level in levels
    }
    # unique on a level: the entity has no other counterpart there AND the
    # candidate has no other counterpart there
    finest_unique_level: NormalizationLevel | None = next(
        (level for level in levels
         if len(counterparts.get((side, entity, level), set())) == 1 and rivals_of_candidate_by_level[level] == 0),
        None,
    )
    return KeyCandidate(
        side=side,
        entity_key_pair=entity_key_pair,
        finest_unique_level=finest_unique_level,
        rivals_of_candidate_by_level=rivals_of_candidate_by_level,
    )


def key_matches_to_candidates_and_statistics(
        alignment_state: AlignmentState,
        entity_pairs: EntityPairIndex,
        population_group_names: dict[Side, set[str]],
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        ) -> dict[tuple[Side, IRI], EntityCandidates]:
    """
    For each entity: the candidates and the summarized other counterparts in its
    population, on both sides. population_group_names: the names of the groups of
    each side's population, resolved (no REST). Each pair is seen from both sides.
    Seen from an entity in its population, the other entity is
    - not counted at all if an entity has no partition (uncategorized),
    - non-competing (counted by its partition) if the partitions do not compete,
    - a candidate if it is in its own population,
    - otherwise a rival outside the population (counted by its groups).
    Pairs with both entities outside their population are left out.
    """
    counterparts = competing_counterparts_by_level(alignment_state, entity_pairs, partition_names_by_entity)
    candidates_by_entity: dict[tuple[Side, IRI], EntityCandidates] = {}

    for entity_key_pair in entity_pairs.values():
        competing = partitions_compete(alignment_state, entity_key_pair, partition_names_by_entity)
        if competing is None:
            continue
        groups_by_side = entity_key_pair.entity_groups()
        in_population = {side: not population_group_names[side].isdisjoint(groups_by_side[side]) for side in SIDES}

        for side in SIDES:
            if not in_population[side]:
                continue  # only entities in population get candidates
            other_side = Side.other_side(side)
            entity = entity_key_pair.get_entity(side)
            entity_candidates = candidates_by_entity.setdefault(
                (side, entity), EntityCandidates(side=side, entity=entity))
            if not competing:
                other_partition = partition_names_by_entity[other_side][entity_key_pair.get_entity(other_side)]
                entity_candidates.non_competing[other_partition] += 1
            elif in_population[other_side]:
                entity_candidates.candidates.append(build_key_candidate(entity_key_pair, side, counterparts))
            else:
                entity_candidates.rivals_outside_population.update(groups_by_side[other_side])
                entity_candidates.rivals_outside_by_level[entity_key_pair.finest_normalization_level()] += 1

    return candidates_by_entity
