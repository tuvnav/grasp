from typing import Callable, Iterable

from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.types import IRI

type ValueIndex = dict[str, list[ValueRow]]


def build_value_index(
        value_rows: Iterable[ValueRow], string_normalization_function: Callable[[str], str]
        ) -> ValueIndex:
    index = {}
    for row in value_rows:
        key = string_normalization_function(row.value)
        if not key:
            continue
        index.setdefault(key, []).append(row)
    return index


def distinct_entities(rows: list[ValueRow]) -> set[IRI]:
    return {row.entity for row in rows}
