from typing import Callable, Iterable

from grasp.tasks.om.data_models.scope import Normalization, StringTransform, apply_string_transforms
from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.text_normalization import NORMALIZATION_LEVELS, NormalizationLevel
from grasp.tasks.om.foundation.types import IRI, Side
from grasp.tasks.om.data_models.entity_pairs import EntityKeyPair, KeyMatch, OverfullKey

type ValueIndex = dict[str, list[ValueRow]]


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
    ) -> tuple[dict[tuple[IRI, IRI], EntityKeyPair], list[OverfullKey]]:
    """
    Pairs of source and target entities sharing a key, on each level present on
    both sides, with all their shared keys (KeyMatch: the rows of just these two
    entities under the key). A key with more distinct entities than
    max_bucket_entities on a side gives no pairs, only an OverfullKey.
    """
    entity_pairs: dict[tuple[IRI, IRI], EntityKeyPair] = {}
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
                            matches=[]
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


def group_entity_pairs_by_entity(
    entity_pairs: dict[tuple[IRI, IRI], EntityKeyPair],
    ) -> dict[Side, dict[IRI, list[EntityKeyPair]]]:
    """
    The pairs of each entity, per side: every pair once under its source entity
    and once under its target entity (an entity with several counterparts has
    several pairs, which are no duplicates).
    """
    entity_pairs_per_entity: dict[Side, dict[IRI, list[EntityKeyPair]]] = {
        Side.SOURCE: {},
        Side.TARGET: {}
        }
    for entity_pair in entity_pairs.values():
        entity_pairs_per_entity[Side.SOURCE].setdefault(
            entity_pair.source_entity, []
            ).append(entity_pair)
        entity_pairs_per_entity[Side.TARGET].setdefault(
            entity_pair.target_entity, []
            ).append(entity_pair)
    return entity_pairs_per_entity
