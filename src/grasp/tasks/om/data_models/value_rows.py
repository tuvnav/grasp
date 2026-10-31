"""
Values of entities as retrieved for comparison (kg_operations/value_retrieval.py),
one row per value: the common input of the key strategy of the candidate stage
and, after its migration, of the anchor rule (which still reads
anchors.KeyEntityRow).
"""
from dataclasses import dataclass

from grasp.tasks.om.foundation.types import IRI


@dataclass(frozen=True)
class ValueRow:
    """One value of an entity. An entity in several groups of a population, or
    with several values, has several rows."""
    entity: IRI
    group: str             # group of the population the entity was retrieved for
    role: str              # role of the value: "names", "synonyms", ... or "own" (own pattern)
    value: str             # the raw value; normalized keys are derived in key_comparison
    language: str | None   # language tag of a literal, if any (multilingual tracks)

    @property
    def primary(self) -> bool:
        """Whether the value is a name (the others only block ambiguous matches)."""
        return self.role == "names"
