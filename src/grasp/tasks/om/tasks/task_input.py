"""
Resolving the input of a task: the KG managers of the source and target KG,
and the alignment state.
"""
from typing import Any

from pydantic import ValidationError

from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.alignment_state import AlignmentState


def resolve_managers(task: GraspTask, input: Any) -> tuple[KgManager, KgManager]:
    if isinstance(input, dict) and "source_kg" in input and "target_kg" in input:
        source, _ = find_manager(task.managers, input["source_kg"])
        target, _ = find_manager(task.managers, input["target_kg"])
        return source, target
    if len(task.managers) != 2:
        raise ValueError(
            "Input must name source_kg and target_kg unless exactly two KGs are configured"
        )
    return task.managers[0], task.managers[1]


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
