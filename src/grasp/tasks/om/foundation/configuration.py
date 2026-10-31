"""
Settings of the om task in the GRASP config: task-specific instructions and the configured entity space.
"""
from pathlib import Path
from typing import Any

from grasp.configs import GraspConfig


def load_task_instructions(om_kwargs: dict) -> str | None:
    """
    Free-text instructions specific to the alignment task at hand (e.g. domain
    knowledge needed to decide on the relations), given inline or as the path
    to a .md/.txt file in the `task_instructions` om task kwargs.
    """
    spec = om_kwargs.get("task_instructions")
    if not spec:
        return None
    if "\n" not in spec and Path(spec).suffix in (".md", ".txt"):
        spec = Path(spec).expanduser().read_text(encoding="utf-8")
    return spec.strip() or None


def get_om_task_config(config: GraspConfig) -> dict[str, Any]:
    return config.task_kwargs.get("om", {})


def configured_namespaces(config: GraspConfig, kg: str, side: str | None = None) -> list[str] | None:
    """Entity space configured in om.scope.namespaces, keyed by KG name
    (e.g. mouse: [http://mouse.owl#]) or, for older configs, by side."""
    configured = get_om_task_config(config).get("scope", {}).get("namespaces", {})
    return configured.get(kg) or (configured.get(side) if side else None)
