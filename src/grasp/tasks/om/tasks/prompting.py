from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from grasp.configs import GraspConfig
from grasp.tasks.om.foundation.configuration import load_task_instructions

PROMPT_DIR = Path(__file__).parent / "prompts"

# a rule of every task with SPARQL functions: models copy blank node labels from
# results into their queries, where they name no node
BLANK_NODE_RULE = (
    "Blank nodes (shown as _:bn…) cannot be named in a query; reach them through the "
    "triple that links them to an entity (e.g. ?s ?p ?node . ?node ?q ?value)."
    )


def load_prompt(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()


# a placeholder in a prompt file, e.g. __TRY_PATTERN__
PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__")


def fill_function_name_placeholders(text: str, function_names: type[StrEnum]) -> str:
    """
    Puts the function names at their placeholders: __MEMBER__ becomes the
    value of function_names.MEMBER (e.g. __TRY_PATTERN__ -> try_pattern).
    Fails if a placeholder is left (a typo, or a renamed or removed
    function), so call it after the other placeholders are filled.
    """
    for member in function_names:
        text = text.replace(f"__{member.name}__", member.value)
    if unfilled := sorted(set(PLACEHOLDER.findall(text))):
        raise ValueError(f"Prompt placeholders without a function name: {unfilled}")
    return text


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
                # keep what stands beside the $ref (description, default)
                siblings = {k: resolve(v) for k, v in node.items() if k not in ("$ref", "title", "discriminator")}
                return {**resolve(defs[node["$ref"].rsplit("/", 1)[-1]]), **siblings}
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


def format_validation_error(error: ValidationError) -> str:
    """
    One line per error: where (dotted path of field names and list positions)
    and what, without pydantic's links and its internal names of the
    alternatives of a type (e.g. "list[str]", "function-after[...]").
    """
    def path(location: tuple) -> str:
        parts = [str(part) for part in location if isinstance(part, int) or str(part).isidentifier()]
        return ".".join(parts) or "input"

    return "; ".join(f"{path(detail['loc'])}: {detail['msg']}" for detail in error.errors())
