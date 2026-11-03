from dataclasses import dataclass

from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.text_normalization import NormalizationLevel
from grasp.tasks.om.foundation.types import IRI


@dataclass(frozen=True)
class KeyMatch:
    level: NormalizationLevel
    key: str
    source_rows: tuple[ValueRow, ...]
    target_rows: tuple[ValueRow, ...]


@dataclass
class EntityKeyPair:
    source_entity: IRI
    target_entity: IRI
    matches: list[KeyMatch]

    def append_key_match(self, key_match: KeyMatch):
        self.matches.append(key_match)

    def finest_normalization_level(self) -> NormalizationLevel:
        ...

    def role_pairs(self) -> set[tuple[str, str]]:
        ...


@dataclass
class OverfullKey:
    """
    A key shared by too many entities to pair them (above max_ambiguity).
    """
    level: NormalizationLevel
    key: str
    num_source_entities: int
    num_target_entities: int
