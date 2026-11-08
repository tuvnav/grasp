"""
Prompts of the tasks (prompts/), the equivalence condition inserted into them,
and the layout of nested texts for the LLM (format_outline).
"""
from pathlib import Path

from grasp.configs import GraspConfig
from grasp.tasks.om.foundation.configuration import load_task_instructions
from grasp.utils import format_list

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


# An outline: items in order, each a line of text or a line with its sub-items.
# The parts of a text for the LLM are built as an outline, and format_outline
# alone decides how levels are shown (bullets, indentation), so no caller joins
# with its own separators.
type OutlineItem = str | tuple[str, Outline]
type Outline = list[OutlineItem]

OUTLINE_INDENT = 2  # spaces per level


def format_outline(outline: Outline, indent: int = 0) -> str:
    """
    An outline as a nested bullet list, one line per item, sub-items indented
    by OUTLINE_INDENT. E.g. ["a", ("b", ["c"])] ->
        - a
        - b
          - c
    """
    lines: list[str] = []
    for item in outline:
        text, sub_items = (item, []) if isinstance(item, str) else item
        lines.append(format_list([text], indent))
        if sub_items:
            lines.append(format_outline(sub_items, indent + OUTLINE_INDENT))
    return "\n".join(lines)
