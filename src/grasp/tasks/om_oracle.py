import random
from typing import Any

from grasp.utils import FunctionCallException
from pydantic import BaseModel
from universal_ml_utils.table import generate_table

from grasp.configs import GraspConfig
from grasp.functions import (
    find_manager,
    SHAPE_CAVEAT,
)
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.entities import Entity, prepare_entity
from grasp.tasks.om import (
    format_entity, AlignmentTaskInput,
    ensure_known, get_entity_shape_text
    )


class CandidateEquivalence(BaseModel):
    entity_source: Entity
    entity_target: Entity


class OracleTaskInput(BaseModel):
    candidate_equivalences: list[CandidateEquivalence]
    source_kg: str
    target_kg: str


class Verdict(BaseModel):
    entity_source: Entity
    entity_target: Entity
    equivalent: bool


class OracleState:
    def __init__(self):
        self.task_input: OracleTaskInput | None = None
        self.verdicts: dict[tuple[str, str], Verdict] = {}

    def add_verdict(self, verdict: Verdict):
        self.verdicts[(
            verdict.entity_source.identifier, verdict.entity_target.identifier
        )] = verdict

    def undo_verdict(self, entity_source, entity_target):
        self.verdicts.pop((entity_source, entity_target), None)

    def get_verdict(self, entity_source, entity_target) -> Verdict | None:
        return self.verdicts.get((entity_source, entity_target))

    def to_dict(self) -> dict:
        verdicts = list(self.verdicts.values())
        return {
            "formatted": self.format(),
            "verdicts": [v.model_dump() for v in verdicts]
        }

    def format(self, num: int | None = None) -> str:
        verdicts = list(self.verdicts.values())

        if not verdicts:
            return "No verdicts yet."

        if num is not None and num < len(verdicts):
            verdicts = random.sample(verdicts, num)

        data = [
            [v.entity_source.identifier, v.entity_target.identifier, str(v.equivalent)]
            for v in verdicts
        ]
        headers = [["Source Entity (Ontology 1)", "Target Entity (Ontology 2)", "Equivalent?"]]

        return generate_table(data=data, headers=headers)


def functions(managers: list[KgManager]) -> list[dict]:
    fns = [
        {
            "name": "confirm_equivalence",
            "description": """\
Confirms a 1:1 equivalence correspondence between an entity from the source \
ontology and an entity from the target ontology. This candidate correspondence \
was suggested by an upstream matcher (the prematcher).\
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
            "name": "reject_equivalence",
            "description": """\
Rejects a candidate 1:1 equivalence between an entity from the source \
ontology and an entity from the target ontology. This candidate correspondence \
was suggested by an upstream matcher (the prematcher).""",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "The full IRI of the source entity",
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "The full IRI of the target entity",
                    },
                },
                "required": ["source_entity", "target_entity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "undo_verdict",
            "description": """\
Revokes a previous confirmation or rejection of a candidate equivalence.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "The full IRI of the source entity",
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "The full IRI of the target entity",
                    },
                },
                "required": ["source_entity", "target_entity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_verdicts",
            "description": "Displays the currently confirmed and rejected mappings. "
            "If 'num' is provided, "
            "it shows a random sample; otherwise, it lists all verdicts.",
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
            "description": "Finalize your verdicts and stop the process.",
        },
    ]
    return fns


def add_verdict(
        managers: list[KgManager],
        kg_source: str,
        kg_target: str,
        entity_source: str,
        entity_target: str,
        equivalent: bool,
        state: OracleState,
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

        message = ""

        existing_with_source = [v for (s, t), v in state.verdicts.items()
                                if s == full_iri_source and t != full_iri_target]
        existing_with_target = [v for (s, t), v in state.verdicts.items()
                                if t == full_iri_target and s != full_iri_source]

        verdict = Verdict(
            entity_source=entity_object_source,
            entity_target=entity_object_target,
            equivalent=equivalent
        )
        state.add_verdict(verdict)

        if equivalent:
            message += f"You confirmed {entity_source} from {kg_source} is equivalent to {entity_target} from {kg_target}."
        else:
            message += f"You rejected an equivalence between {entity_source} from {kg_source} and {entity_target} from {kg_target}."

        if existing_with_source or existing_with_target:
            message += (
                "\nNotice that your verdict concerns an entity "
                "which is already involved in another or multiple other verdicts:")

        if existing_with_source:
            for v in existing_with_source:
                message += (
                    f"\n{v.entity_source.identifier} has been judged as "
                    f"{'' if v.equivalent else 'not '}equivalent to "
                    f"{v.entity_target.identifier}.")

        if existing_with_target:
            for v in existing_with_target:
                message += (
                    f"\n{v.entity_source.identifier} has been judged as "
                    f"{'' if v.equivalent else 'not '}equivalent to "
                    f"{v.entity_target.identifier}.")

        if existing_with_source or existing_with_target:
            message += (
                "\nCheck whether this verdict conflicts with the other one(s). "
                "If so, use 'undo_verdict' to revise either.")

        return message

    except ValueError as e:
        raise FunctionCallException(str(e)) from e


def undo_verdict(
        entity1: str,
        entity2: str,
        state: OracleState,
        ) -> str:
    verdict = state.get_verdict(entity1, entity2)
    if verdict is None:
        raise FunctionCallException(
            f"There is no verdict concerning a potential equivalence between "
            f"{entity1} and {entity2} yet. "
            "Note: use the full IRI as the entity reference."
            )
    state.undo_verdict(entity1, entity2)
    if verdict.equivalent:
        return f"{entity1} and {entity2} are no longer considered equivalent"
    else:
        return f"{entity1} and {entity2} are no longer considered non-equivalent"


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: OracleState | None = None,
    example_indices: dict | None = None,
) -> str:
    assert isinstance(state, OracleState), (
        "Alignments must be provided as state for OM Oracle task"
    )
    assert not example_indices, "Example indices are not supported for OM Oracle task"

    om_oracle_kwargs = config.task_kwargs.get("om_oracle", {})

    if fn_name == "confirm_equivalence":
        if state is None:
            return "No OracleState was provided. Cannot record verdicts without an OracleState"
        kg1 = state.task_input.source_kg  # type: ignore
        kg2 = state.task_input.target_kg  # type: ignore

        ent1, ent2 = fn_args["source_entity"], fn_args["target_entity"]

        know_before_use = om_oracle_kwargs.get("know_before_use", True)

        return add_verdict(
            managers, kg1, kg2, ent1, ent2, True, state, known,
            know_before_use
        )

    elif fn_name == "reject_equivalence":
        if state is None:
            return "No OracleState was provided. Cannot record verdicts without an OracleState"
        kg1 = state.task_input.source_kg  # type: ignore
        kg2 = state.task_input.target_kg  # type: ignore

        ent1, ent2 = fn_args["source_entity"], fn_args["target_entity"]
        know_before_use = om_oracle_kwargs.get("know_before_use", True)

        return add_verdict(
            managers, kg1, kg2, ent1, ent2, False, state, known,
            know_before_use
        )

    elif fn_name == "show_verdicts":
        return state.format(num=fn_args.get("num"))

    elif fn_name == "undo_verdict":
        ent1, ent2 = fn_args["source_entity"], fn_args["target_entity"]
        return undo_verdict(ent1, ent2, state)

    elif fn_name == "stop":
        return "Stopping"

    else:
        raise ValueError(f"Unknown function {fn_name}")


def input_instructions(
    task_input: OracleTaskInput,
    state: OracleState,
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

    len_qe = len(task_input.candidate_equivalences)
    if len_qe == 0:
        raise FunctionCallException("Input instructions for LLM must contain candidates for equivalence.")

    quantifier_qe = "candidate equivalences"
    if len_qe == 1:
        quantifier_qe = "a candidate equivalence"
    instructions += f"You are given {quantifier_qe} found by a prematcher."
    if agentic:
        if len_qe > 1:
            instructions += (
                " Verify whether each is an actual equivalence. Call "
                "'confirm_equivalence' for each one that holds, and "
                "'reject_equivalence' for each one that does not. You may "
                "call these functions multiple times within the same turn "
                "(once per candidate) instead of one at a time, once you "
                "have gathered enough information to judge several of them."
            )
        else:
            instructions += (
                " Verify whether it is an actual equivalence. Call "
                "'confirm_equivalence' if it holds, and 'reject_equivalence' "
                "if it does not."
            )
    else:
        instructions += "Does this equivalence hold? Answer with exactly one word: 'True' or 'False'"

    for qe in task_input.candidate_equivalences:
        if len_qe > 1:
            block_num += 1
            instructions += f"\n### Entity {block_num}\n"
        instructions += entity_block(qe.entity_source, source_manager, "Source")
        instructions += entity_block(qe.entity_target, target_manager, "Target")
        instructions += "\n"

    if shape_shown:
        instructions += f"\n{SHAPE_CAVEAT}\n"

    return instructions


def input_and_state(
    input: Any,
    config: GraspConfig,
    managers: list[KgManager],
    known: set[str],
) -> tuple[str, OracleState]:
    try:
        task_input = OracleTaskInput(**input)
    except Exception as e:
        raise ValueError("OM Oracle task input must match OracleTaskInput schema") from e

    state = OracleState()
    state.task_input = task_input
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
        om_oracle_kwargs = self.config.task_kwargs.get("om_oracle", {})
        prompt_file = om_oracle_kwargs.get("prompt_file")
        if prompt_file is not None:
            with open(prompt_file, "r") as f:
                output = "\n".join(f.readlines())
        return output

    def rules(self) -> list[str]:
        return [
            "Before calling 'stop', make sure every candidate equivalence you "
            "were given has been judged via 'confirm_equivalence' or "
            "'reject_equivalence'. If any are still missing a verdict when "
            "you try to stop, you will be asked to judge them first.",
        ]

    def function_definitions(self) -> list[dict]:
        return functions(self.managers)

    def output(self, messages: list[Message]) -> dict:
        return self.state.to_dict()

    def incomplete_reason(self) -> str | None:
        task_input = self.state.task_input
        if task_input is None:
            return None

        all_pairs = [
            (ce.entity_source.identifier, ce.entity_target.identifier)
            for ce in task_input.candidate_equivalences
        ]
        missing = [pair for pair in all_pairs if pair not in self.state.verdicts]
        if not missing:
            return None

        lines = "\n".join(f"- {source} <-> {target}" for source, target in missing)
        return (
            f"You have not yet judged {len(missing)} of the "
            f"{len(all_pairs)} candidate equivalence(s) you were given. "
            "Call 'confirm_equivalence' or 'reject_equivalence' for each of "
            f"the following before stopping:\n{lines}"
        )

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
