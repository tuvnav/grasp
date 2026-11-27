"""
Resolving the input of a task: the KG managers of the source and target KG
(by name), the alignment state, the strategy history and the entity
memberships.
"""
from typing import Any

from pydantic import BaseModel, ValidationError

from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidates import StrategyHistory
from grasp.tasks.om.data_models.profile import EntityMemberships
from grasp.tasks.om.foundation.types import IRI, Side
from grasp.tasks.om.kg_operations.grouping import retrieve_entity_memberships

# the key of the entity memberships in a task input: passed as the object
# itself (large; a task runs in the process of its caller), not serialized
ENTITY_MEMBERSHIPS_KEY = "entity_memberships"


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


def resolve_entity_memberships(
        input: Any, managers: dict[Side, KgManager], alignment_state: AlignmentState
        ) -> EntityMemberships:
    """
    The EntityMemberships under the key ENTITY_MEMBERSHIPS_KEY of the input
    (an object or serialized); without them retrieved from the KGs (a task run
    on its own).
    """
    memberships = input.get(ENTITY_MEMBERSHIPS_KEY) if isinstance(input, dict) else None
    if memberships is None:
        return retrieve_entity_memberships(managers, alignment_state)
    try:
        return EntityMemberships.model_validate(memberships)
    except ValidationError as error:
        raise ValueError(
            f"Invalid EntityMemberships under the key {ENTITY_MEMBERSHIPS_KEY!r}: {error}"
            ) from error


def group_names_by_entity(
        entity_memberships: EntityMemberships | None, side: Side
        ) -> dict[IRI, set[str]] | None:
    """The profile groups of the entities of one side; None without memberships (an EntityDisplay then queries them)."""
    return None if entity_memberships is None else entity_memberships.group_names_by_entity[side]


def to_task_input(task_input: BaseModel) -> dict[str, Any]:
    """
    A task input model as the input of a task run in this process: serialized,
    except the entity memberships, which are passed as they are (not copied).
    """
    serialized = task_input.model_dump(mode="json", exclude={ENTITY_MEMBERSHIPS_KEY})
    memberships = getattr(task_input, ENTITY_MEMBERSHIPS_KEY, None)
    if memberships is not None:
        serialized[ENTITY_MEMBERSHIPS_KEY] = memberships
    return serialized
