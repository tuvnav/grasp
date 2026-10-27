from dataclasses import dataclass, field
from enum import StrEnum
import random
from typing import Any, Callable, Optional

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.entity_context import entity_context
from grasp.tasks.om.model import (
    IRI, AnchorRecord, EntityContextConfig, Verdict, VerdictOption, VerifyInput,
    add_equivalence_condition, get_om_task_config, load_task_instructions, local_name, namespace_of, FunctionRegistry, resolve_managers
)
from grasp.utils import FunctionCallException, format_section


@dataclass
class VerifyState:
    source_kg: str
    target_kg: str
    source_manager: KgManager
    target_manager: KgManager
    anchor: AnchorRecord
    verdict: Optional[Verdict] = None
    # entities shown to the LLM (with competitors: the pair plus similar entities)
    presented_source: list[IRI] = field(default_factory=list)
    presented_target: list[IRI] = field(default_factory=list)


class VerifyFunction(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNSURE = "unsure"
    SET_VERDICT = "set_verdict"
    STOP = "stop"


func_reg = FunctionRegistry()


@func_reg.register(VerifyFunction.ACCEPT)
def accept_anchor(state: VerifyState, fn_args: dict, known: set[str]) -> str:
    anchor = state.anchor
    rationale = fn_args.get("rationale")
    if rationale is None:
        raise FunctionCallException(
            f"Argument 'rationale' required when calling '{VerifyFunction.ACCEPT}'"
            )
    state.verdict = Verdict(
        **anchor.model_dump(),
        verdict=VerdictOption.ACCEPT,
        rationale=rationale
    )

    return (
        f"You judged {anchor.source_iri} and {anchor.target_iri} to be "
        "truly equivalent in the context of the task."
        )


@func_reg.register(VerifyFunction.REJECT)
def reject_anchor(state: VerifyState, fn_args: dict, known: set[str]) -> str:
    anchor = state.anchor
    rationale = fn_args.get("rationale")
    if rationale is None:
        raise FunctionCallException(
            f"Argument 'rationale' required when calling '{VerifyFunction.REJECT}'"
            )
    state.verdict = Verdict(
        **anchor.model_dump(),
        verdict=VerdictOption.REJECT,
        rationale=rationale
    )
    return (
        f"You judged {anchor.source_iri} and {anchor.target_iri} to be "
        "not equivalent in the context of the task."
        )


@func_reg.register(VerifyFunction.UNSURE)
def unsure_anchor(state: VerifyState, fn_args: dict, known: set[str]) -> str:
    anchor = state.anchor
    rationale = fn_args.get("rationale")
    if rationale is None:
        raise FunctionCallException(
            f"Argument 'rationale' required when calling '{VerifyFunction.UNSURE}'"
            )
    state.verdict = Verdict(
        **anchor.model_dump(),
        verdict=VerdictOption.UNSURE,
        rationale=rationale
    )
    return (
        f"You confirmed you are unsure about {anchor.source_iri} "
        f"and {anchor.target_iri} being equivalent in the context "
        "of the task."
        )


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
    anchor = state.anchor

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
        m["source"] == anchor.source_iri and m["target"] == anchor.target_iri for m in mappings
    )
    if unsure:
        verdict = VerdictOption.UNSURE
    else:
        verdict = VerdictOption.ACCEPT if anchor_accepted else VerdictOption.REJECT
    state.verdict = Verdict(**anchor.model_dump(), verdict=verdict, rationale=rationale)

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
                    (mapping["source"] + " → " + mapping["target"]
                     for mapping in mappings[:-1])
                )
            } "
            f"and {mappings[-1]["source"]} → {mappings[-1]["target"]} "
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

    cfg = EntityContextConfig()
    for src_ent in source_entities:
        input += entity_context(state.source_manager, src_ent, cfg) + "\n\n"

    if len(target_entities) == 1:
        input += "**Target entity**: "
    else:
        input += "**Target entities:**\n"

    for tgt_ent in target_entities:
        input += entity_context(state.target_manager, tgt_ent, cfg) + "\n\n"

    return input


def retrieve_competitors(
        entity: IRI,
        manager: KgManager,
        k: int = 4,
        entity_space: tuple[str, ...] | None = None,
        ) -> list[IRI]:
    """IRIs of up to k entities from the KG's entity index most similar to
    `entity`, excluding the entity itself and entities outside the entity space
    (default: the namespace of `entity`, which keeps out e.g. vocabulary
    entities like oboInOwl:Synonym). Used to present the LLM with plausible,
    confusable alternatives instead of just the pair."""
    space = entity_space or (namespace_of(entity),)
    label = manager.get_label(entity, "entities") or local_name(entity)
    # over-fetch, since the filter may remove some alternatives
    alternatives = manager.search_index("entities", query=label, k=3 * k + 1)
    return [
        IRI(alt.identifier) for alt in alternatives
        if alt.identifier != entity and alt.identifier.startswith(space)
    ][:k]


def get_verify_task_config(config: GraspConfig) -> dict[str, Any]:
    return get_om_task_config(config).get("om_verify", {})


class OMVerifyTask(GraspTask):
    name = "om-verify"
    state: VerifyState

    def system_information(self) -> str:
        task_kwargs = get_verify_task_config(self.config)
        competitors = bool(task_kwargs.get("competitors", False))
        agentic = task_kwargs.get("mode") == "agentic"

        prompt = (
            "You are an ontology matching assistant. "
            "The goal is 1:1 equivalences between the entities of "
            "a source and a target ontology."
        )
        prompt += "\n\n" + "\n\n".join(
            system_prompt_completion(competitors, agentic, self.config)
            )

        prompt = add_equivalence_condition(prompt, self.config)

        task_instructions = load_task_instructions(get_om_task_config(self.config))
        if task_instructions:
            prompt += "\n\n" + format_section("Task-specific instructions", task_instructions)

        return prompt

    def setup(self, input: Any) -> str:
        """Input must be a serialized VerifyInput"""
        try:
            input_as_class = VerifyInput(**input)
        except Exception as e:
            raise ValueError(f"Input must be a serialized VerifyInput object: {e}") from e

        source_manager, target_manager = resolve_managers(self, input)
        self.state = VerifyState(
            source_kg=input_as_class.source_kg,
            target_kg=input_as_class.target_kg,
            source_manager=source_manager,
            target_manager=target_manager,
            # the model itself, not model_dump(): the verdict functions need an AnchorRecord
            anchor=input_as_class.anchor,
            )

        task_kwargs = get_verify_task_config(self.config)
        competitors = bool(task_kwargs.get("competitors", False))
        k_competitors = int(task_kwargs.get("k_competitors", 4))
        # entity space per KG name (pipeline parameter), else the entity's namespace
        entity_spaces = get_om_task_config(self.config).get("entity_space", {})

        src_iri = input_as_class.anchor.source_iri
        tgt_iri = input_as_class.anchor.target_iri

        src_iris, tgt_iris = [src_iri], [tgt_iri]

        if competitors:
            comps_src = retrieve_competitors(
                src_iri, self.state.source_manager, k_competitors,
                tuple(entity_spaces.get(input_as_class.source_kg, ())) or None,
                )
            comps_tgt = retrieve_competitors(
                tgt_iri, self.state.target_manager, k_competitors,
                tuple(entity_spaces.get(input_as_class.target_kg, ())) or None,
                )
            src_iris.extend(comps_src)
            tgt_iris.extend(comps_tgt)

            rng = random.Random(f"{src_iri}|{tgt_iri}")
            rng.shuffle(src_iris)
            rng.shuffle(tgt_iris)

        self.state.presented_source = src_iris
        self.state.presented_target = tgt_iris
        return input_instructions(self.state, src_iris, tgt_iris)

    def rules(self) -> list[str]:
        return []

    def offered_functions(self) -> list[VerifyFunction]:
        task_kwargs = get_verify_task_config(self.config)
        agentic = task_kwargs.get("mode") == "agentic"
        competitors = bool(task_kwargs.get("competitors", False))

        if competitors:
            function_names = [VerifyFunction.SET_VERDICT]
        else:
            function_names = [VerifyFunction.ACCEPT, VerifyFunction.REJECT, VerifyFunction.UNSURE]

        if agentic:
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

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        fn: Optional[Callable] = func_reg.Registry.get(fn_name)
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
