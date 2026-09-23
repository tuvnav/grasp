import random
from typing import Any
from enum import StrEnum

from pydantic import BaseModel
from universal_ml_utils.table import generate_table

from grasp.configs import GraspConfig, ShapeConfig
from grasp.functions import (
    find_manager,
    update_known_from_shape_iris,
    SHAPE_CAVEAT,
)
from grasp.manager import KgManager, format_kgs
from grasp.model import Message
from grasp.sparql.types import AskResult
from grasp.sparql.utils import prepare_identifier_for_sparql
from grasp.tasks.base import GraspTask
from grasp.tasks.entities import Entity, prepare_entity
from grasp.tasks.om import (
    format_entity, Relation, Correspondence,
    PotentialCorrespondences, AlignmentTaskInput, AlignmentState,
    entity_exists_in_kg, ensure_known, get_entity_shape_text
    )
from grasp.utils import FunctionCallException, format_list, format_notes
from grasp.build.shapes import collect_iris, compute_shape, emit_pseudo_shex


def functions(managers: list[KgManager]) -> list[dict]:
    fns = [
        {
            "name": "confirm_equivalence",
            "description": """\
Confirms a 1:1 equivalence correspondence between an entity from the source \
ontology and one from the target ontology suggested by the prematcher.\
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "The exact IRI of the source entity",
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "The exact IRI of the target entity",
                    },
                },
                "required": ["source_entity", "target_entity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_equivalence",
            "description": "Delete a formerly confirmed equivalence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "The full IRI of the source entity whose correspondence should be removed",
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "The full IRI of the target entity whose correspondence should be removed",
                    },
                },
                "required": ["source_entity", "target_entity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_correspondences",
            "description": "Displays the currently confirmed mappings. If 'num' is provided, "
            "it shows a random sample; otherwise, it lists all correspondences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num": {
                        "type": "integer",
                        "description": "The number of random mappings to display for review",
                        },
                    },
                "required": [],
                "additionalProperties": False,
                },
            "strict": False,
        },
        {
            "name": "stop",
            "description": "Finalize your mappings and stop the alignment process. May be "
            "rejected if there is still an unresolved logical conflict in the current "
            "alignment - resolve it and call this again.",
        },
    ]
    return fns


def rules() -> list[str]:
    return []


def confirm_equivalence(
        managers: list[KgManager],
        kg_source: str,
        kg_target: str,
        entity_source: str,
        entity_target: str,
        state: AlignmentState,
        known: set[str],
        know_before_use: bool = True,
        ) -> str:

    try:
        manager_source, _ = find_manager(managers, kg_source)
        manager_target, _ = find_manager(managers, kg_target)
        entity_object_source = prepare_entity(manager_source, entity_source)
        entity_object_target = prepare_entity(manager_target, entity_target)
        full_iri_source = entity_object_source.identifier
        full_iri_target = entity_object_target.identifier

        if know_before_use:
            ensure_known(manager_source, full_iri_source, known)
            ensure_known(manager_target, full_iri_target, known)

        correspondence = Correspondence(entity_source=entity_object_source, entity_target=entity_object_target)
        state.add_1_to_1_correspondence(correspondence)
        return f"You confirmed {entity_source} from {kg_source} is equivalent to {entity_target} from {kg_target}."

    except ValueError as e:
        raise FunctionCallException(str(e)) from e


def delete_correspondence(
        entity1: str,
        entity2: str,
        state: AlignmentState,
        ) -> str:
    correspondence1 = state.get_correspondence(entity1)
    correspondence2 = state.get_correspondence(entity2)
    if correspondence1 is None or correspondence1 is not correspondence2:
        raise FunctionCallException(
            f"There is no Correspondence aligning {entity1} to {entity2} yet. "
            "Notice you have to use the full IRI as entity reference."
            )
    state.remove_correspondence(entity1, entity2)
    return f"Deleted correspondence between {entity1} and {entity2}"


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: AlignmentState | None = None,
    example_indices: dict | None = None,
) -> str:
    assert isinstance(state, AlignmentState), (
        "Alignments must be provided as state for OM task"
    )
    assert not example_indices, "Example indices are not supported for OM task"

    om_oracle_kwargs = config.task_kwargs.get("om_oracle", {})

    if fn_name == "confirm_equivalence":
        if state is None:
            return "No AlignmentState was provided. Cannot set correspondences without AlignmentState"
        kg1 = state.task_input.source_kg  # type: ignore
        kg2 = state.task_input.target_kg  # type: ignore

        ent1, ent2 = fn_args["source_entity"], fn_args["target_entity"]
        know_before_use = om_oracle_kwargs.get("know_before_use", True)

        return confirm_equivalence(
            managers, kg1, kg2, ent1, ent2, state, known,
            know_before_use
        )

    elif fn_name == "delete_correspondence":
        return delete_correspondence(
            fn_args["source_entity"],
            fn_args["target_entity"],
            state
            )

    elif fn_name == "show_correspondences":
        return state.format(num=fn_args.get("num"))

    elif fn_name == "stop":
        return "Stopping"

    else:
        raise ValueError(f"Unknown function {fn_name}")


def input_instructions(
    task_input: AlignmentTaskInput,
    state: AlignmentState,
    managers: list[KgManager],
    known: set[str],
    agentic: bool = True  # for outside of GRASP use of `input_instructions`
) -> str:
    instructions = ""
    source_manager, _ = find_manager(managers, task_input.source_kg)
    target_manager, _ = find_manager(managers, task_input.target_kg)
    shape_shown = False
    block_num = 0

    def entity_block(entity: Entity, manager: KgManager, label: str) -> str:
        nonlocal shape_shown
        block = f"**{label}**: {format_entity(entity)}\n"
        shape_text = get_entity_shape_text(manager, entity.identifier, known)
        if shape_text is not None and "no properties found for this class" not in shape_text:
            shape_shown = True
            block += f"\n```shape: \n{shape_text}\n```\n"
        return block

    len_pc = len(task_input.potential_correspondences)
    if len_pc == 0:
        raise FunctionCallException("Input instructions for LLM must contain candidates for equivalence")

    quantity_corrs = "candidate equivalences"
    filler_1 = "each"
    filler_2 = " for each "
    if len_pc == 1:
        quantity_corrs = "a candidate equivalence"
        filler_1 = "it"
        filler_2 = " "
    instructions += f"You are given {quantity_corrs} found by a prematcher."
    if agentic:
        instructions += f"""\
Verify whether {filler_1} is an actual equivalence, and call \
'confirm_equivalence'{filler_2}if so.\
"""
    else:
        if len_pc > 1:
            instructions += "Do these equivalences hold? Answer with 'True' or 'False'"
        else:
            instructions += "Does this equivalence hold? Answer with 'True' or 'False'"

    for corr in task_input.potential_correspondences:
        if len_pc > 1:
            block_num += 1
            instructions += f"### Entity {block_num}\n"
        instructions += entity_block(corr.source_entity, source_manager, "Source")
        if len(corr.candidates) == 1:
            instructions += entity_block(corr.candidates[0], target_manager, "Target")
        else:
            for j, c in enumerate(corr.candidates, start=1):
                instructions += entity_block(c, target_manager, f"Target Candidate {j}")
        instructions += "\n"

    if shape_shown:
        instructions += f"\n{SHAPE_CAVEAT}\n"

    return instructions


def input_and_state(
    input: Any,
    config: GraspConfig,
    managers: list[KgManager],
    known: set[str],
) -> tuple[str, AlignmentState]:
    try:
        task_input = AlignmentTaskInput(**input)
    except Exception as e:
        raise ValueError("OM task input must match AlignmentTaskInput schema") from e

    state = AlignmentState()
    state.task_input = task_input
    for correspondence in state.task_input.input_alignment:
        state.add_1_to_1_correspondence(correspondence)

    instructions = input_instructions(task_input, state, managers, known)
    return instructions, state


class OmOracleTask(GraspTask):
    name = "om-oracle"

    def setup(self, input: Any) -> str:
        instructions, self.state = input_and_state(
            input, self.config, self.managers, self.known
        )
        return instructions

    def system_information(self) -> str:
        output = ""
        prompt_file = self.config.task_kwargs["om_oracle"]["prompt_file"]
        if prompt_file is not None:
            with open(prompt_file, "r") as f:
                output = "\n".join(f.readlines())
        return output

    def rules(self) -> list[str]:
        return []

    def function_definitions(self) -> list[dict]:
        return functions(self.managers)

    def output(self, messages: list[Message]) -> dict:
        return self.state.to_dict()

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
            ) -> str:
        return call_function(
            self.config,
            self.managers,
            fn_name,
            fn_args,
            known,
            self.state,
            example_indices,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    @property
    def default_input_field(self) -> str | None:
        return None
