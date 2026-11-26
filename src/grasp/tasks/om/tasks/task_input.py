"""
Resolving the input of a task: the KG managers of the source and target KG
(by name), and the alignment state.
"""
from typing import Any

from pydantic import ValidationError

from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidates import StrategyHistory
from grasp.tasks.om.foundation.types import Side


def managers_by_side(task: GraspTask, source_kg: str, target_kg: str) -> dict[Side, KgManager]:
    """The KG managers of the source and the target KG, by their names (e.g. from the alignment state)."""
    source, _ = find_manager(task.managers, source_kg)  # raises on an unknown KG name
    target, _ = find_manager(task.managers, target_kg)
    return {Side.SOURCE: source, Side.TARGET: target}


def resolve_strategy_history(input: Any) -> StrategyHistory:
    """
    The StrategyHistory serialized under the key 'strategy_history' of the
    input; without it an empty history (the first round).
    """
    try:
        return StrategyHistory.model_validate(input.get("strategy_history") or {})
    except ValidationError as error:
        raise ValueError(
            f"Invalid StrategyHistory under the key 'strategy_history': {error}"
            ) from error


def resolve_alignment_state(input: Any) -> AlignmentState:
    """
    The AlignmentState serialized under the key 'alignment_state' of the input.
    """
    if not isinstance(input, dict) or "alignment_state" not in input:
        raise ValueError(
            "The input of this task must be a dict with a serialized AlignmentState "
            "under the key 'alignment_state'."
            )
    try:
        return AlignmentState.model_validate(input["alignment_state"])
    except ValidationError as error:
        raise ValueError(
            f"Invalid AlignmentState under the key 'alignment_state': {error}"
            ) from error
