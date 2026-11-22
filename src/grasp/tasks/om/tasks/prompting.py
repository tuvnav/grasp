from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

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
    if load_task_instructions(config) is not None:
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


def inline_schema(
        model: type[BaseModel], drop: frozenset[str] = frozenset()
        ) -> dict:
    """
    JSON schema of a model with its $defs inlined (not every model API
    resolves $ref), without titles, without the given properties and without
    discriminator mappings (they point into $defs; the const kind of each
    alternative of a oneOf is enough).
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            node = {k: resolve(v) for k, v in node.items() if k not in ("title", "discriminator")}
            if "properties" in node:
                node["properties"] = {k: v for k, v in node["properties"].items() if k not in drop}
                if "required" in node:
                    node["required"] = [r for r in node["required"] if r not in drop]
            return node
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)
