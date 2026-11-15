"""
Settings of the om task in the GRASP config (task_kwargs -> om) that GRASP
itself reads: the task-specific instructions, whether an entity must be known
before it is used, and the configured entity space. The keys are named only
here; the orchestration validates the whole om section against its model
(OmSettings, which must have these keys as fields).
"""
from pathlib import Path
from typing import Any

from grasp.configs import GraspConfig

TASK_INSTRUCTIONS_KEY = "task_instructions"
KNOW_BEFORE_USE_KEY = "know_before_use"
SCOPE_KEY = "scope"
NAMESPACES_KEY = "namespaces"
# the keys of task_kwargs -> om that GRASP reads
GRASP_OM_KEYS = (TASK_INSTRUCTIONS_KEY, KNOW_BEFORE_USE_KEY, SCOPE_KEY)


def get_om_task_config(config: GraspConfig) -> dict[str, Any]:
    return config.task_kwargs.get("om", {})


def load_task_instructions(config: GraspConfig) -> str | None:
    """
    Free-text instructions specific to the alignment task at hand (e.g. domain
    knowledge needed to decide on equivalences), given inline or as the path
    to a .md/.txt file.
    """
    spec = get_om_task_config(config).get(TASK_INSTRUCTIONS_KEY)
    if not spec:
        return None
    if "\n" not in spec and Path(spec).suffix in (".md", ".txt"):
        spec = Path(spec).expanduser().read_text(encoding="utf-8")
    return spec.strip() or None


def know_before_use(config: GraspConfig) -> bool:
    """Whether an entity must exist in its KG before it is used in an equivalence (default: yes)."""
    return bool(get_om_task_config(config).get(KNOW_BEFORE_USE_KEY, True))


def configured_namespaces(config: GraspConfig, kg: str) -> list[str] | None:
    """The entity space configured for a KG (scope -> namespaces -> KG name)."""
    return get_om_task_config(config).get(SCOPE_KEY, {}).get(NAMESPACES_KEY, {}).get(kg)
