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


def resolve_managers(
        task: GraspTask,
        input: Any = None,
        source_kg: str | None = None,
        target_kg: str | None = None
        ) -> tuple[KgManager, KgManager]:
    if (
            not (source_kg and target_kg) and isinstance(input, dict)
            and "source_kg" in input and "target_kg" in input
            ):
        source_kg = str(input.get("source_kg"))
        target_kg = str(input.get("target_kg"))

    if source_kg is None or target_kg is None:
        raise ValueError(
            "No source_kg or no target_kg name was given. "
            "Either specifiy the names explicitly in the args "
            "of this function or as separate fields in the task input."
            )

    source, _ = find_manager(task.managers, source_kg)  # raises when wrong kg name
    target, _ = find_manager(task.managers, target_kg)

    return source, target


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
