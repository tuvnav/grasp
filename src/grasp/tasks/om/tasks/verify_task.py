import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Optional

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.verification import Verdict, VerdictOption, VerifyInput
from grasp.tasks.om.foundation.configuration import (
    load_task_instructions,
)
from grasp.tasks.om.foundation.types import IRI, Side
from grasp.tasks.om.kg_operations.entity_rendering import LONG_VIEW, EntityDisplay
from grasp.tasks.om.tasks.function_registry import FunctionRegistry
from grasp.tasks.om.tasks.prompting import add_equivalence_condition
from grasp.tasks.om.tasks.task_input import managers_by_side
from grasp.utils import FunctionCallException, format_section


@dataclass
class VerifyState:
    source_kg: str
    target_kg: str
    kg_managers: dict[Side, KgManager]
    source_iri: IRI
    target_iri: IRI
    entity_displays: dict[Side, EntityDisplay]
    agentic: bool
    competitors_by_side: dict[Side, list[IRI]]
    verdict: Optional[Verdict] = None

    @property
    def competitors(self) -> bool:
        """Competitors on at least one side: the judge sees a group, not a pair."""
        return any(self.competitors_by_side.get(side) for side in (Side.SOURCE, Side.TARGET))

    @property
    def presented_source(self) -> list[IRI]:
        return [self.source_iri] + self.competitors_by_side.get(Side.SOURCE, [])

    @property
    def presented_target(self) -> list[IRI]:
        return [self.target_iri] + self.competitors_by_side.get(Side.TARGET, [])


class VerifyFunction(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNSURE = "unsure"
    SET_VERDICT = "set_verdict"
    STOP = "stop"


func_reg = FunctionRegistry()


def judge_focus_pair(state: VerifyState, fn_args: dict, function_name: VerifyFunction,
                     verdict: VerdictOption, judged_as: str) -> str:
    """The verdict on the presented pair (accept, reject, unsure), with the
    rationale; judged_as completes the confirmation for the model."""
    rationale = fn_args.get("rationale")
    if rationale is None:
        raise FunctionCallException(f"Argument 'rationale' required when calling '{function_name}'")
    state.verdict = Verdict(
        source_iri=state.source_iri,
        target_iri=state.target_iri,
        verdict=verdict,
        rationale=rationale,
    )
    return f"You judged {state.source_iri} and {state.target_iri} {judged_as}."


@func_reg.register(VerifyFunction.ACCEPT)
def accept_anchor(state: VerifyState, fn_args: dict, known: set[str]) -> str:
    return judge_focus_pair(state, fn_args, VerifyFunction.ACCEPT, VerdictOption.ACCEPT,
                            "to be truly equivalent in the context of the task")


@func_reg.register(VerifyFunction.REJECT)
def reject_anchor(state: VerifyState, fn_args: dict, known: set[str]) -> str:
    return judge_focus_pair(state, fn_args, VerifyFunction.REJECT, VerdictOption.REJECT,
                            "to be not equivalent in the context of the task")


@func_reg.register(VerifyFunction.UNSURE)
def unsure_anchor(state: VerifyState, fn_args: dict, known: set[str]) -> str:
    return judge_focus_pair(state, fn_args, VerifyFunction.UNSURE, VerdictOption.UNSURE,
                            "as unsure: you cannot decide whether they are equivalent in the context of the task")


@func_reg.register(VerifyFunction.SET_VERDICT)
def set_verdict(state: VerifyState, fn_args: dict, known: set[str]) -> str:
    mappings = fn_args.get("mappings")
    if mappings is None:
        raise FunctionCallException(
            f"Mappings must be provided when calling {VerifyFunction.SET_VERDICT}. "
            "If you don't want to set any equivalence, call the function "
            "with an empty list."
            )
    rationale = fn_args.get("rationale")
    if rationale is None:
        raise FunctionCallException(
            f"Argument 'rationale' required when calling '{VerifyFunction.SET_VERDICT}'"
            )
    unsure = bool(fn_args.get("unsure", False))

    # all pairs must consist of presented entities; an entity may occur in several
    # pairs (entities can be equivalent within an ontology), frequent use of this
    # may also hint at an unreliable judge
    for mapping in mappings:
        iri_source, iri_target = mapping["source"], mapping["target"]
        if iri_source not in state.presented_source:
            raise FunctionCallException(
                f"{iri_source} is not one of the presented source entities: {state.presented_source}"
            )
        if iri_target not in state.presented_target:
            raise FunctionCallException(
                f"{iri_target} is not one of the presented target entities: {state.presented_target}"
            )

    anchor_accepted = any(
        m["source"] == state.source_iri and m["target"] == state.target_iri for m in mappings
    )
    if unsure:
        verdict = VerdictOption.UNSURE
    else:
        verdict = VerdictOption.ACCEPT if anchor_accepted else VerdictOption.REJECT
    # all mappings among the presented entities, also those between competitors
    # (by-products for later stages); unsure: the mappings are ignored
    state.verdict = Verdict(
        source_iri=state.source_iri,
        target_iri=state.target_iri,
        verdict=verdict,
        rationale=rationale,
        mappings=None if unsure else [(mapping["source"], mapping["target"]) for mapping in mappings],
        )

    if unsure:
        return "You confirmed you are unsure which of the presented entities are equivalent."
    if not mappings:
        return (
            "You confirmed there are no equivalence relationships "
            "between any source and target entities presented "
            "to you in this task.")
    elif len(mappings) == 1:
        return (
            f"You judged {mappings[0]["source"]} and {mappings[0]["target"]} to be "
            "truly equivalent in the context of the task."
        )
    else:
        return (
            f"You judged {
                ", ".join(
                    (mapping["source"] + " ↔ " + mapping["target"]
                     for mapping in mappings[:-1])
                )
            } "
            f"and {mappings[-1]["source"]} ↔ {mappings[-1]["target"]} "
            "to be truly equivalent in the context of the task."
            )


@func_reg.register(VerifyFunction.STOP)
def stop(state: VerifyState, fn_args: dict, known: set[str]) -> str:
    return "Stopping"


def function_definitions(function_names: list[VerifyFunction]) -> list[dict]:
    rationale_property = {
        "type": "string",
        "description": "Short explanation of your decision.",
    }

    def verdict_function(name: VerifyFunction, description: str) -> dict:
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"rationale": rationale_property},
                "required": ["rationale"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    res = []
    for func_name in function_names:
        match func_name:
            case VerifyFunction.STOP:
                res.append(
                    {
                        "name": VerifyFunction.STOP,
                        "description": "Finish the task after you have established your decision.",
                    }
                )
            case VerifyFunction.SET_VERDICT:
                res.append(
                    {
                        "name": VerifyFunction.SET_VERDICT,
                        "description": (
                            "Establish which pairs of a source and a target entity among "
                            "those presented to you are truly equivalent, if any. "
                            "Call with an empty list if none of the presented "
                            "entities are equivalent, and with unsure set to true if you "
                            "cannot decide."
                            ),
                        "parameters": {
                            "type": "object",
                                    "properties": {
                                        "mappings": {
                                            "type": "array",
                                            "description": "The source-target pairs you judge to be equivalent.",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "The exact IRI of the source entity.",
                                                    },
                                                    "target": {
                                                        "type": "string",
                                                        "description": "The exact IRI of the target entity.",
                                                    },
                                                },
                                                "required": ["source", "target"],
                                                "additionalProperties": False,
                                            },
                                        },
                                        "unsure": {
                                            "type": "boolean",
                                            "description": "True if you cannot decide which of the presented entities are equivalent; the mappings are then ignored.",
                                        },
                                        "rationale": rationale_property,
                                    },
                            "required": ["mappings", "rationale"],
                            "additionalProperties": False,
                            },
                        "strict": False,
                        }
                )
            case VerifyFunction.ACCEPT:
                res.append(
                    verdict_function(
                        VerifyFunction.ACCEPT,
                        "Establish that the source and the target entity "
                        "are truly equivalent in the context of the task."
                    )
                )
            case VerifyFunction.REJECT:
                res.append(
                    verdict_function(
                        VerifyFunction.REJECT,
                        "Establish that the source and the target entity "
                        "are not equivalent in the context of the task."
                    )
                )
            case VerifyFunction.UNSURE:
                res.append(
                    verdict_function(
                        VerifyFunction.UNSURE,
                        "Confirm that you cannot decide whether the source and the "
                        "target entity are equivalent in the context of the task."
                        )
                    )
    return res


def system_prompt_completion(competitors: bool, agentic: bool, config: GraspConfig) -> tuple[str, str, str]:
    """Returns an 'task outline', 'steps to follow' and 'special notes' text"""
    steps = (
        "Follow these steps:\n"
        "- Consider the given context for each entity "
        "and think about how these entities are going "
        "to be used in real world applications of the "
        "ontologies.\n"
        )
    if agentic:
        steps += (
            "- Use the built in functions to explore "
            "the involved ontologies if you regard the "
            "provided context as insufficient.\n"
        )
        notes = (
            f"Note that you only have {config.max_steps} "
            "steps to take your decision. After this, the program "
            "loop will terminate automatically, so make sure "
            "to establish your decision after up to "
            f"{config.max_steps * 2 // 3} steps."
            )
        if config.parallel_tool_calls:
            notes += (" You can invoke multiple tool calls in one step.")
    else:
        notes = (
            "Note that you have to come to a decision in a "
            "single step and that you have to call one of the "
            "functions to establish it instead of writing it "
            "down in the chat."
        )

    if competitors:
        outline = (
            "You are presented a group of similar entities, "
            "coming from two distinct ontologies. Each entity "
            "is presented with its relevant context within "
            "its ontology. Decide which pairs of a source and "
            "a target entity are equivalent, if any."
        )
        steps += "- Call 'set_verdict' to establish your decision."

    else:
        outline = (
            "You are presented a pair of entities from "
            "two distinct ontologies with their relevant context "
            "within their ontologies. Your task is to verify whether "
            "they constitute a true equivalence."
        )
        steps += (
            "- establish your decision by calling 'accept', "
            "'reject' or 'unsure'."
        )
    return outline, steps, notes


def input_instructions(
        state: VerifyState,
        source_entities: list[IRI],
        target_entities: list[IRI]
        ) -> str:
    input = "Consider the following entities:\n\n"
    if not source_entities or not target_entities:
        raise FunctionCallException(
            "OMVerifyTask needs at least one source entity and at least one target entity."
        )

    if len(source_entities) == 1:
        input += "**Source entity**: "
    else:
        input += "**Source entities:**\n"

    for side, entities in ((Side.SOURCE, source_entities), (Side.TARGET, target_entities)):
        state.entity_displays[side].prefetch(entities)
    for src_ent in source_entities:
        input += state.entity_displays[Side.SOURCE].show(src_ent) + "\n\n"

    if len(target_entities) == 1:
        input += "**Target entity**: "
    else:
        input += "**Target entities:**\n"

    for tgt_ent in target_entities:
        input += state.entity_displays[Side.TARGET].show(tgt_ent) + "\n\n"

    return input


class OMVerifyTask(GraspTask):
    name = "om-verify"
    state: VerifyState

    def system_information(self) -> str:
        prompt = (
            "You are an ontology matching assistant. "
            "The goal is 1:1 equivalences between the entities of "
            "a source and a target ontology."
        )
        prompt += "\n\n" + "\n\n".join(
            system_prompt_completion(
                self.state.competitors, self.state.agentic, self.config
                )
            )

        prompt = add_equivalence_condition(prompt, self.config)

        task_instructions = load_task_instructions(self.config)
        if task_instructions:
            prompt += "\n\n" + format_section(
                "Task-specific instructions", task_instructions
                )

        return prompt

    def setup(self, input: Any) -> str:
        """
        Input must be a serialized VerifyInput
        """
        try:
            verify_input = VerifyInput(**input)
        except Exception as e:
            raise ValueError(
                f"Input must be a serialized VerifyInput object: {e}"
                ) from e

        managers = managers_by_side(self, verify_input.source_kg, verify_input.target_kg)
        self.state = VerifyState(
            source_kg=verify_input.source_kg,
            target_kg=verify_input.target_kg,
            kg_managers=managers,
            source_iri=verify_input.source_iri,
            target_iri=verify_input.target_iri,
            entity_displays={
                Side.SOURCE: EntityDisplay(managers[Side.SOURCE], verify_input.source_profile, LONG_VIEW),
                Side.TARGET: EntityDisplay(managers[Side.TARGET], verify_input.target_profile, LONG_VIEW),
            },
            agentic=verify_input.agentic,
            competitors_by_side=verify_input.competitors
            )

        src_iris = list(self.state.presented_source)
        tgt_iris = list(self.state.presented_target)

        rng = random.Random(f"{verify_input.source_iri}|{verify_input.target_iri}")
        rng.shuffle(src_iris)
        rng.shuffle(tgt_iris)

        return input_instructions(self.state, src_iris, tgt_iris)

    def rules(self) -> list[str]:
        return []

    def offered_functions(self) -> list[VerifyFunction]:
        if self.state.competitors:
            function_names = [VerifyFunction.SET_VERDICT]
        else:
            function_names = [VerifyFunction.ACCEPT, VerifyFunction.REJECT, VerifyFunction.UNSURE]

        if self.state.agentic:
            function_names.append(VerifyFunction.STOP)
        return function_names

    def function_definitions(self) -> list[dict]:
        return function_definitions(self.offered_functions())

    def verdict_functions(self) -> list[VerifyFunction]:
        return [f for f in self.offered_functions() if f != VerifyFunction.STOP]

    def final_step_functions(self) -> list[dict] | None:
        # the last allowed step must submit a verdict
        return function_definitions(self.verdict_functions())

    def budget_message(self, steps_left: int, final: bool) -> str:
        names = " or ".join(f"'{f.value}'" for f in self.verdict_functions())
        if final:
            return (
                "This is your last step. Decide now based on what you have found "
                f"so far and call {names}."
            )
        return (
            f"You have {steps_left} steps left, including this one. Make sure to "
            f"call {names} before you run out of steps."
        )

    def call_function(
            self,
            fn_name: str,
            fn_args: dict,
            known: set[str],
            example_indices: dict | None) -> str:
        fn: Optional[Callable] = func_reg.handlers.get(fn_name)
        if fn is None or fn_name not in self.offered_functions():
            raise FunctionCallException(
                f"Unknown function {fn_name}, use one of {[f.value for f in self.offered_functions()]}"
            )
        return fn(self.state, fn_args, known)

    def done(self, fn_name: str) -> bool:
        return (
            fn_name in (
                VerifyFunction.ACCEPT, VerifyFunction.REJECT,
                VerifyFunction.UNSURE, VerifyFunction.SET_VERDICT,
                VerifyFunction.STOP,
            )
            and self.state.verdict is not None
        )

    def output(self, messages: list[Message]) -> dict | None:
        verdict = self.state.verdict
        if verdict is None:
            return None
        else:
            return verdict.model_dump(mode="json")
