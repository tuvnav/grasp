from collections import defaultdict
from typing import Callable, Iterable
import numpy as np

from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.entity_pairs import (
    Candidate, EntityCandidates, EntityPair, KeyCandidate, KeyEntityPair, KeyMatch, KeyHub, VectorCandidate, VectorEntityPair, VectorHub, VectorMatch,
)
from grasp.tasks.om.data_models.scope import Normalization, StringTransform, apply_string_transforms
from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.text_normalization import NORMALIZATION_LEVELS, NormalizationLevel
from grasp.tasks.om.foundation.text_vectors import (
    TEXT_BATCH_SIZE, TextVectors, Vectorizer, build_similarity_matrix,
    count_text_vectors, vectorize_texts,
)
from grasp.tasks.om.foundation.types import IRI, SIDES, Side

type ValueIndex = dict[str, list[ValueRow]]
# (source entity, target entity) -> their pair; P: the pair kind of the strategy
type EntityPairIndex[P: EntityPair] = dict[tuple[IRI, IRI], P]


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


# This method is for key candidates only
def intersect_level_indices(
        source_indices: dict[NormalizationLevel, ValueIndex],
        target_indices: dict[NormalizationLevel, ValueIndex],
        max_entity_pairs_per_bucket: int | None = None
        ) -> tuple[EntityPairIndex[KeyEntityPair], list[KeyHub]]:
    """
    Pairs of source and target entities sharing a key, on each level present on
    both sides, with all their shared keys (KeyMatch: the rows of just these two
    entities under the key). A key whose entities would give more entity pairs
    (source entities x target entities) than max_entity_pairs_per_bucket gives
    no pairs, only a KeyHub.
    """
    entity_pairs: EntityPairIndex[KeyEntityPair] = {}
    overfull_keys: list[KeyHub] = []

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
            if (max_entity_pairs_per_bucket is not None
                    and len(rows_by_source_entity) * len(rows_by_target_entity) > max_entity_pairs_per_bucket):
                overfull_keys.append(
                    KeyHub(
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
                        KeyEntityPair(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            matches=[]
                            )
                    ).append_match(
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


def text_pairs_to_entity_pairs(
        cosine_by_text_pairs: dict[tuple[int, int], float],
        source_texts: list[str],
        target_texts: list[str],
        source_index: ValueIndex,
        target_index: ValueIndex,
        max_entity_pairs_per_bucket: int | None = None
) -> tuple[EntityPairIndex[VectorEntityPair], list[VectorHub]]:

    entity_pairs: EntityPairIndex[VectorEntityPair] = {}
    hubs: list[VectorHub] = []

    for (i, j), score in cosine_by_text_pairs.items():
        source_text, target_text = source_texts[i], target_texts[j]
        source_value_rows = source_index[source_text]
        target_value_rows = target_index[target_text]
        rows_by_source_entity = group_value_rows_by_entity(source_value_rows)
        rows_by_target_entity = group_value_rows_by_entity(target_value_rows)

        if (max_entity_pairs_per_bucket is not None
                and len(rows_by_source_entity) * len(rows_by_target_entity) > max_entity_pairs_per_bucket):
            hubs.append(
                VectorHub(
                    source_text=source_text,
                    target_text=target_text,
                    num_source_entities=len(rows_by_source_entity),
                    num_target_entities=len(rows_by_target_entity)
                )
            )

            continue

        for source_entity, source_rows in rows_by_source_entity.items():
            for target_entity, target_rows in rows_by_target_entity.items():
                entity_pairs.setdefault(
                    (source_entity, target_entity),
                    VectorEntityPair(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        matches=[]
                        )
                ).append_match(
                    VectorMatch(
                        source_rows=tuple(source_rows),
                        target_rows=tuple(target_rows),
                        source_text=source_text,
                        target_text=target_text,
                        cosine=score
                        )
                )

    return entity_pairs, hubs


def nearest_texts_per_row(
        vectors_a: TextVectors,
        vectors_b: TextVectors,
        neighbours_per_text: int
) -> list[tuple[int, int, float]]:
    """
    Get the nearest <neighbours_per_text> texts
    with cosine > 0. Returns:
    (a_index, b_index, cosine)
    """
    n, m = count_text_vectors(vectors_a), count_text_vectors(vectors_b)
    if n == 0 or m == 0:
        return []
    cut_off = min(m, neighbours_per_text)
    res = []
    for start in range(0, n, TEXT_BATCH_SIZE):
        a_block = vectors_a[start: start + TEXT_BATCH_SIZE]
        similarities = build_similarity_matrix(a_block, vectors_b)
        best_neighbours = np.argpartition(-similarities, cut_off - 1, axis=1)[:, :cut_off]  # unsorted top neighbours
        for a_row in range(similarities.shape[0]):
            for b_row in best_neighbours[a_row]:
                cosine = similarities[a_row, b_row]
                if cosine > 0:
                    res.append((int(start + a_row), int(b_row), float(cosine)))
    return res


def find_nearest_text_pairs(
        source_vectors: TextVectors,
        target_vectors: TextVectors,
        neighbours_per_text: int
        ) -> dict[tuple[int, int], float]:
    """
    Returns a mapping from indices (i, j) -> score.
    """
    cosines_by_text_pairs = {}
    for (i, j, cosine) in nearest_texts_per_row(source_vectors, target_vectors, neighbours_per_text):
        cosines_by_text_pairs[(i, j)] = cosine
    for (j, i, cosine) in nearest_texts_per_row(target_vectors, source_vectors, neighbours_per_text):
        cosines_by_text_pairs[(i, j)] = cosine
    return cosines_by_text_pairs


def compare_value_indices_by_vectors(
        source_index: ValueIndex,
        target_index: ValueIndex,
        vectorizer: Vectorizer,
        model_name: str | None,
        max_entity_pairs_per_bucket: int | None = None,
        neighbours_per_text: int = 50,
        ) -> tuple[EntityPairIndex[VectorEntityPair], list[VectorHub]]:
    source_texts, target_texts = list(source_index), list(target_index)
    source_vectors, target_vectors = vectorize_texts(source_texts, target_texts, vectorizer, model_name)
    text_pairs = find_nearest_text_pairs(source_vectors, target_vectors, neighbours_per_text)  # {(i, j): cosine}
    return text_pairs_to_entity_pairs(text_pairs, source_texts, target_texts, source_index, target_index, max_entity_pairs_per_bucket)


def entity_pairs_by_entity[P: EntityPair](
        entity_pairs: EntityPairIndex[P],
        ) -> dict[tuple[Side, IRI], list[P]]:
    """
    Returns an index mapping from an entity (with its side: the same IRI may
    occur in both KGs) to all pair-objects it is involved in. The number of
    entries under one entity is the number of entities of the other side it
    matches, on any level.
    """
    entity_pairs_per_entity: dict[tuple[Side, IRI], list[P]] = {}
    for entity_pair in entity_pairs.values():
        for side in SIDES:
            entity_pairs_per_entity.setdefault(
                (side, entity_pair.get_entity(side)), []
                ).append(entity_pair)
    return entity_pairs_per_entity


def partitions_compete(
        alignment_state: AlignmentState,
        entity_pair: EntityPair,
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        ) -> bool | None:
    """
    Whether the partitions of the two entities compete;
    None if an entity has no
    partition (uncategorized: not aligned).
    """
    source_partition = partition_names_by_entity[Side.SOURCE].get(entity_pair.source_entity)
    target_partition = partition_names_by_entity[Side.TARGET].get(entity_pair.target_entity)
    if source_partition is None or target_partition is None:
        return None
    return alignment_state.competes(Side.SOURCE, source_partition, Side.TARGET, target_partition)


# Key candidates only
def competing_counterparts_by_level(
        alignment_state: AlignmentState,
        entity_pairs: EntityPairIndex[KeyEntityPair],
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
    for key_entity_pair in entity_pairs.values():
        if not partitions_compete(alignment_state, key_entity_pair, partition_names_by_entity):
            continue  # does not compete, or an entity without a partition
        for level in key_entity_pair.levels():
            counterparts[(Side.SOURCE, key_entity_pair.source_entity, level)].add(key_entity_pair.target_entity)
            counterparts[(Side.TARGET, key_entity_pair.target_entity, level)].add(key_entity_pair.source_entity)
    return counterparts


def build_key_candidate(
        key_entity_pair: KeyEntityPair,
        side: Side,
        counterparts: CompetingCounterparts,
        ) -> KeyCandidate:
    """
    The candidate of the entity on `side` (the other entity of the pair), complete
    in one go: everything that depends on other pairs comes from `counterparts`.
    """
    other_side = Side.other_side(side)
    entity, candidate = key_entity_pair.get_entity(side), key_entity_pair.get_entity(other_side)
    levels: list[NormalizationLevel] = key_entity_pair.levels()  # fine to coarse
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
        entity_pair=key_entity_pair,
        finest_unique_level=finest_unique_level,
        num_rivals_of_candidate_by_level=rivals_of_candidate_by_level,
    )


def build_vector_candidate(
        vector_entity_pair: VectorEntityPair,
        side: Side
) -> VectorCandidate:
    return VectorCandidate(
        side=side,
        entity_pair=vector_entity_pair
    )


# generich method for all candidate types
def entity_pairs_to_candidates[P: EntityPair](
        alignment_state: AlignmentState,
        entity_pairs: EntityPairIndex[P],
        population_group_names: dict[Side, set[str]],
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        build_candidate: Callable[[P, Side], Candidate],
        ) -> dict[tuple[Side, IRI], EntityCandidates]:
    """
    For each entity: the candidates and the summarized other counterparts in its
    population, on both sides, for any strategy. population_group_names: the
    names of the groups of each side's population, resolved (no REST). Each pair
    is seen from both sides. Seen from an entity in its population, the other
    entity is
    - not counted at all if an entity has no partition (uncategorized),
    - non-competing (counted by its partition) if the partitions do not compete,
    - a candidate if it is in its own population (built by build_candidate from
      the pair and the side of the entity: the strategy's own candidate fields),
    - otherwise a rival outside the population (counted by its groups and by the
      rank key of the pair).
    Pairs with both entities outside their population are left out.
    """
    candidates_by_entity: dict[tuple[Side, IRI], EntityCandidates] = {}

    for entity_pair in entity_pairs.values():
        competing = partitions_compete(alignment_state, entity_pair, partition_names_by_entity)
        if competing is None:
            continue
        groups_by_side = entity_pair.entity_groups()
        in_population = {side: not population_group_names[side].isdisjoint(groups_by_side[side]) for side in SIDES}

        for side in SIDES:
            if not in_population[side]:
                continue  # only entities in population get candidates
            other_side = Side.other_side(side)
            entity = entity_pair.get_entity(side)
            entity_candidates = candidates_by_entity.setdefault(
                (side, entity), EntityCandidates(side=side, entity=entity))
            if not competing:
                other_partition = partition_names_by_entity[other_side][entity_pair.get_entity(other_side)]
                entity_candidates.non_competing[other_partition] += 1
            elif in_population[other_side]:
                entity_candidates.candidates.append(build_candidate(entity_pair, side))
            else:
                entity_candidates.rivals_outside_population.update(groups_by_side[other_side])
                entity_candidates.rivals_outside_by_rank[entity_pair.rank_key] += 1

    return candidates_by_entity


def key_entity_pairs_to_candidates(
        alignment_state: AlignmentState,
        entity_pairs: EntityPairIndex[KeyEntityPair],
        population_group_names: dict[Side, set[str]],
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        ) -> dict[tuple[Side, IRI], EntityCandidates]:
    """
    entity_pairs_to_candidates for the key strategy: 
    the key candidates need
    the competing counterparts of all pairs per level.
    """
    counterparts = competing_counterparts_by_level(alignment_state, entity_pairs, partition_names_by_entity)
    return entity_pairs_to_candidates(
        alignment_state, entity_pairs, population_group_names,
        partition_names_by_entity,
        build_candidate=lambda entity_pair, side: build_key_candidate(
            entity_pair, side, counterparts
            ),
    )


def vector_entity_pairs_to_candidates(
        alignment_state: AlignmentState,
        entity_pairs: EntityPairIndex[VectorEntityPair],
        population_group_names: dict[Side, set[str]],
        partition_names_by_entity: dict[Side, dict[IRI, str]],
        ) -> dict[tuple[Side, IRI], EntityCandidates]:
    return entity_pairs_to_candidates(
        alignment_state, entity_pairs, population_group_names,
        partition_names_by_entity, build_vector_candidate
    )
