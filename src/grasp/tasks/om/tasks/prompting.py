"""
Prompts of the tasks (prompts/) and the equivalence condition inserted into them.
"""
from pathlib import Path

from grasp.configs import GraspConfig
from grasp.tasks.om.foundation.configuration import load_task_instructions

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()


EQUIVALENCE_PLACEHOLDER = "__EQUIVALENCE_CONDITION__"


def add_equivalence_condition(text: str, config: GraspConfig) -> str:
    """
    Puts the equivalence condition at the placeholder in text, or appends it.
    The default heuristic (equivalence_condition.txt: should a query for the
    source entity return the target entity?) is only added without
    task-specific instructions, since it can contradict their goal (e.g. only
    one direction counts, or the ontologies are merged into one).
    """
    injection = (
        "Whether entities across ontologies are truly "
        "equivalent is ultimately a functional question. "
        "When deciding about entities being equivalent across "
        "ontologies, think primarily about how these entities "
        "are used in real world scenarios and if these entities "
        "are treated as equivalent in the context of "
        "their usage."
    )
    if load_task_instructions(config.task_kwargs.get("om", {})) is not None:
        injection += (
            " For more information about the usage of the ontologies "
            "and the alignment you assist with, see the section "
            "'Task-specific instructions'."
        )
    else:
        injection += " " + load_prompt("equivalence_condition.txt")

    if EQUIVALENCE_PLACEHOLDER in text:
        return text.replace(EQUIVALENCE_PLACEHOLDER, injection)
    return text + "\n\n" + injection
