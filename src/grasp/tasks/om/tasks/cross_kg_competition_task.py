"""
The cross-KG competition (om-cross-kg-competition): which partitions of the
source KG and of the target KG do not compete, i.e. an entity of one is never
a candidate or a rival for an entity of the other. Runs on the alignment state
after the profiles of both KGs; its output are the non-competing pairs, which
the orchestration puts into the alignment state.

Default: as within a KG, a partition named property does not compete with the
other partitions, but only if both KGs have one. KGs whose partitions are cut
otherwise (e.g. items and properties of a generic KG under names of its own)
get no default: there the names say nothing about where the properties are.
"""
from typing import Any, Callable

from grasp.functions import find_manager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.alignment_state import (
    AlignmentState,
    CrossKgCompetitionOutput,
    CrossKgNonCompetingPartitionPair,
)
from grasp.tasks.om.data_models.profile import KgProfile
from grasp.tasks.om.foundation.configuration import load_task_instructions
from grasp.tasks.om.foundation.types import Side
from grasp.tasks.om.tasks.profile_stage import PartitionEditor
from grasp.tasks.om.tasks.prompting import load_prompt
from grasp.utils import FunctionCallException, format_section

FINISH = "finish"
PROPERTY_PARTITION = "property"
# set_by of the pairs
DEFAULT_SET_BY = "default"
TASK_SET_BY = "cross-kg-competition"
SIDES = (Side.SOURCE, Side.TARGET)


def default_cross_kg_non_competing_pairs(
        source_profile: KgProfile,
        target_profile: KgProfile
        ) -> list[CrossKgNonCompetingPartitionPair]:
    """The property partition of each KG does not compete with the other
    partitions of the other KG, if both KGs have a property partition; else none."""
    source_names, target_names = source_profile.partition_names(), target_profile.partition_names()
    if PROPERTY_PARTITION not in source_names or PROPERTY_PARTITION not in target_names:
        return []
    return [
        CrossKgNonCompetingPartitionPair(
            source_partition=source_name, target_partition=target_name,
            reason="properties and entities are not equivalent", set_by=DEFAULT_SET_BY,
            )
        for source_name in source_names for target_name in target_names
        if (source_name == PROPERTY_PARTITION) != (target_name == PROPERTY_PARTITION)
        ]


class CrossKgCompetitionTask(GraspTask):
    """
    Input: a dict with alignment_state (both profiles; pairs across the KGs
    set already are kept, otherwise the default). The partitions of both KGs
    are shown with sizes, features and examples (read-only partition editors).
    """
    name = "om-cross-kg-competition"
    prompt_file = "cross_kg_competition.txt"
    alignment_state: AlignmentState
    partition_editors: dict[Side, PartitionEditor]  # read only: sizes, features, examples
    finished: bool
    rationale: str | None

    def setup(self, input: Any) -> str:
        if not isinstance(input, dict) or "alignment_state" not in input:
            raise ValueError("Input must be a dict with the alignment_state")
        alignment_state = AlignmentState.model_validate(input["alignment_state"])
        if not alignment_state.cross_kg_non_competing_pairs:
            alignment_state = alignment_state.with_cross_kg_non_competing_pairs(
                default_cross_kg_non_competing_pairs(
                    alignment_state.source_kg_profile, alignment_state.target_kg_profile
                    )
                )
        self.alignment_state = alignment_state
        self.partition_editors = {side: self.read_partitions(side) for side in SIDES}
        self.finished, self.rationale = False, None
        return self.input_text()

    def read_partitions(self, side: Side) -> PartitionEditor:
        """The partitions of one KG with their statistics."""
        profile = self.alignment_state.get_kg_profile(side)
        manager, _ = find_manager(self.managers, profile.kg_name)
        return PartitionEditor.from_kg_profile(manager, profile, create_missing=False)

    # --- input ---

    def input_text(self) -> str:
        sections = []
        for side in SIDES:
            profile = self.alignment_state.get_kg_profile(side)
            editor = self.partition_editors[side]
            manager, _ = find_manager(self.managers, profile.kg_name)
            parts = [f"Knowledge graph: {profile.kg_name}"
                     + (f" ({manager.description})" if manager.description else ""),
                     editor.input_for_llm()]
            parts += [editor.describe_member(name, f"Partition {name}")
                      for name in editor.names() if editor.group_sizes.get(name)]
            sections.append(format_section(f"Partitions of the {side.value} knowledge graph", "\n\n".join(parts)))
        sections.append(format_section("Partitions of the two knowledge graphs that do not compete",
                                       self.format_cross_kg_pairs()))
        return "\n\n".join(sections)

    def format_cross_kg_pairs(self) -> str:
        pairs = self.alignment_state.cross_kg_non_competing_pairs
        if pairs:
            return "Source partition - target partition:\n" + "\n".join(
                f"  {pair.source_partition} - {pair.target_partition}: {pair.reason} [{pair.set_by}]" for pair in pairs)
        source_names = self.alignment_state.source_kg_profile.partition_names()
        target_names = self.alignment_state.target_kg_profile.partition_names()
        if PROPERTY_PARTITION in source_names and PROPERTY_PARTITION in target_names:
            return "None: all partitions of the two knowledge graphs compete."
        return ("None: all partitions of the two knowledge graphs compete. There is no default, as the "
                f"partitions are cut differently (not both knowledge graphs have a partition {PROPERTY_PARTITION}).")

    # --- functions ---

    def set_competition_definition(self) -> dict:
        partition_names = {side: self.alignment_state.get_kg_profile(side).partition_names() for side in SIDES}
        return {
            "name": "set_competition",
            "description": (
                "Sets whether the entities of a partition of the source knowledge graph and of a partition of "
                "the target knowledge graph compete with each other: if not, an entity of one is never a "
                "candidate or a rival for an entity of the other. Shows the pairs that do not compete afterwards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_partition": {"type": "string", "enum": partition_names[Side.SOURCE]},
                    "target_partition": {"type": "string", "enum": partition_names[Side.TARGET]},
                    "compete": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["source_partition", "target_partition", "compete", "reason"],
                "additionalProperties": False,
            },
        }

    def finish_definition(self) -> dict:
        return {
            "name": FINISH,
            "description": "Submits the pairs of partitions that do not compete.",
            "parameters": {
                "type": "object",
                "properties": {"rationale": {"type": "string", "description": "What you decided and why."}},
                "required": ["rationale"],
                "additionalProperties": False,
            },
        }

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return [(self.set_competition_definition(), self.set_competition),
                (self.finish_definition(), self.finish)]

    def set_competition(self, fn_args: dict) -> str:
        source_partition, target_partition = fn_args.get("source_partition"), fn_args.get("target_partition")
        if not isinstance(source_partition, str) or not isinstance(target_partition, str):
            raise FunctionCallException("source_partition and target_partition are required")
        try:
            if fn_args.get("compete"):
                self.alignment_state.allow_competition(Side.SOURCE, source_partition, Side.TARGET, target_partition)
            else:
                self.alignment_state.forbid_competition(
                    Side.SOURCE, source_partition, Side.TARGET, target_partition,
                    reason=fn_args.get("reason") or "", set_by=TASK_SET_BY,
                    )
        except ValueError as e:
            raise FunctionCallException(str(e)) from e
        return self.format_cross_kg_pairs()

    def finish(self, fn_args: dict) -> str:
        self.finished, self.rationale = True, fn_args.get("rationale")
        return "Submitted. " + self.format_cross_kg_pairs()

    # --- GraspTask ---

    def system_information(self) -> str:
        info = load_prompt(self.prompt_file)
        task_instructions = load_task_instructions(self.config)
        if task_instructions:
            info += "\n\n" + format_section("Task-specific instructions", task_instructions)
        return info

    def rules(self) -> list[str]:
        return [
            "Inspect the knowledge graphs with the SPARQL functions where it is unclear what a partition holds.",
            "You can execute multiple tool calls in one step.",
            "Call finish exactly once, at the end.",
            f"You have at most {self.config.max_steps} steps; one step is one response of yours, with all "
            "tool calls in it.",
        ]

    def function_definitions(self) -> list[dict]:
        return [definition for definition, _ in self.functions()]

    def final_step_functions(self) -> list[dict] | None:
        return [self.finish_definition()]

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        handlers = {definition["name"]: handler for definition, handler in self.functions()}
        if fn_name not in handlers:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return handlers[fn_name](fn_args)

    def done(self, fn_name: str) -> bool:
        return fn_name == FINISH and self.finished

    def output(self, messages: list[Message]) -> dict[str, Any]:
        return CrossKgCompetitionOutput(
            finished=self.finished,
            rationale=self.rationale,
            non_competing_pairs=self.alignment_state.cross_kg_non_competing_pairs,
        ).model_dump(mode="json")
