"""
Data model of the anchor stage (tasks/anchor_task.py): the stage result, and
the anchors of an ensemble of key runs with their specificity
(kg_operations/candidate_selection.py: merge_anchors).
"""
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from grasp.tasks.om.data_models.candidate_selection import RunEnsemble, TriedRun
from grasp.tasks.om.foundation.text_normalization import EXTENDED_LEVELS, NormalizationLevel
from grasp.tasks.om.foundation.types import IRI


class AnchorStageResult(BaseModel):
    """The output of the anchor stage: the ensemble of key runs selected for the
    full runs (None: no anchors, e.g. no strategy gives reliable ones), and all
    tried runs. finished: the LLM called finish."""
    source_kg: str
    target_kg: str
    finished: bool
    rationale: str | None
    run_ensemble: RunEnsemble | None
    runs: dict[str, TriedRun]


# --- anchors of key runs (kg_operations/candidate_selection.py: merge_anchors) ---

class AnchorSpecificity(StrEnum):
    """
    Whether the words of an anchor's name are extended by another name of one
    of the KGs ({paper} by {poster, paper}): a head anchor may name a broader
    or narrower thing than its partner (granularity homonymy), a specific one
    hardly. Only for names; for codes and other values not applicable.
    """
    SPECIFIC = "specific"
    HEAD = "head"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Anchor:
    """A pair that is unique in at least one run of the ensemble (the anchor rule)."""
    source_entity: IRI
    target_entity: IRI
    families: frozenset[str]   # families of the runs that found it
    normalization_level: NormalizationLevel  # finest normalization level of its key over these runs
    specificity: AnchorSpecificity
    # lexical information content of its name (Seco 2004), the minimum over both
    # sides; 1: specific, lower: extended by more names; None: not applicable
    information_content: float | None

    @property
    def pair(self) -> tuple[IRI, IRI]:
        return self.source_entity, self.target_entity

    @property
    def extended(self) -> bool:
        """Unique only on a coarse level (word set, plural, inflection)."""
        return self.normalization_level in EXTENDED_LEVELS


@dataclass(frozen=True)
class AnchorMerge:
    """
    The anchors of an ensemble of key runs. Each run gives only 1:1 pairs, but
    two runs may give an entity different partners: these pairs are conflicts,
    left out of the anchors (they become candidates of a later stage).
    """
    anchors: list[Anchor]
    conflicts: list[tuple[IRI, IRI, frozenset[str]]]  # (source, target, families)

