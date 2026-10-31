"""
Resolving the input of a task: the KG managers of the source and target KG.
"""
from typing import Any

from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.tasks.base import GraspTask


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
