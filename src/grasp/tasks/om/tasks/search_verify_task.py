"""
The search-verify judge (om-search-verify): an agent gets a unit (focus
entities with their candidate lists, tasks/candidate_task.py and
kg_operations/candidate_selection.py: build_judging_units), judges the
candidates and searches beyond the lists with the KG functions; it sets
equivalences and finishes.

The variant (CandidateJudgeVariant) decides how the unit is shown: focused
(the focus entities are named, each with its list) or clusters (two lists of
entities, the focus not named), with or without the signals that ranked the
candidates, in rank order or shuffled.

set_equivalence checks that both entities exist (know_before_use), lie in the
entity space and in no excluded group, and warns, without setting it, if
- an entity is matched already (by the alignment state, or in this run),
  with the equivalences the new one would imply within each ontology,
- the pair was rejected before,
- the partitions of the two entities do not compete.
Calling it again with the same arguments after a warning sets it anyway (a
warning left unanswered at finish means: not set). Only equivalences set in
this run can be deleted; one kept against an earlier equivalence becomes a
conflict in the ledger, checked later.
"""
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from grasp.manager import KgManager
from grasp.model import Message
from grasp.sparql.types import AskResult
from grasp.sparql.utils import prepare_identifier_for_sparql
from grasp.tasks.base import GraspTask
from grasp.tasks.entities import prepare_entity
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.verification import (
    CandidateJudgeVariant,
    FocusEntity,
    JudgingUnit,
    ListedCandidate,
    SearchVerifyInput,
    SearchVerifyResult,
)
from grasp.tasks.om.foundation.configuration import know_before_use, load_task_instructions
from grasp.tasks.om.foundation.types import IRI, EntityKey, Side
from grasp.tasks.om.kg_operations.entity_rendering import LONG_VIEW, EntityDisplay
from grasp.tasks.om.sparql_access import ensure_known
from grasp.tasks.om.tasks.function_registry import FunctionRegistry
from grasp.tasks.om.tasks.prompting import add_equivalence_condition, load_prompt
from grasp.tasks.om.tasks.task_input import managers_by_side
from grasp.utils import FunctionCallException, format_section

SIDES = (Side.SOURCE, Side.TARGET)
type EquivalencePair = tuple[IRI, IRI]  # (source entity, target entity)


class SearchVerifyFunction(StrEnum):
    SET_EQUIVALENCE = "set_equivalence"
    DELETE_EQUIVALENCE = "delete_equivalence"
    FINISH = "finish"


@dataclass
class SearchVerifyState:
    kg_managers: dict[Side, KgManager]
    alignment_state: AlignmentState
    unit: JudgingUnit
    variant: CandidateJudgeVariant
    entity_displays: dict[Side, EntityDisplay]
    know_before_use: bool
    # the entities shown, per side, in the order shown
    presented: dict[Side, list[IRI]] = field(default_factory=dict)
    equivalences: list[EquivalencePair] = field(default_factory=list)  # set in this run
    # pair -> what its last warning was about; a repeated call sets the pair if
    # nothing new turned up since
    pending_warnings: dict[EquivalencePair, frozenset[str]] = field(default_factory=dict)
    # (side, entity) -> its partition (None: in none), the excluded group it is in (None: in none)
    partition_cache: dict[EntityKey, str | None] = field(default_factory=dict)
    excluded_group_cache: dict[EntityKey, str | None] = field(default_factory=dict)
    finished: bool = False
    rationale: str = ""
    unsure: list[EntityKey] = field(default_factory=list)

    @property
    def kg_names(self) -> dict[Side, str]:
        return {side: self.alignment_state.get_kg_profile(side).kg_name for side in SIDES}


func_reg = FunctionRegistry()


# ---------------------------------------------------------------- what is shown

def ordered_candidates(focus: FocusEntity, variant: CandidateJudgeVariant, seed: str) -> list[ListedCandidate]:
    """
    The candidates of a focus entity in rank order, or shuffled (seeded by the unit).
    """
    candidates = list(focus.candidates)
    if variant.candidate_order == "random":
        random.Random(f"{seed}|{focus.entity}").shuffle(candidates)
    return candidates


def presented_entities(unit: JudgingUnit, variant: CandidateJudgeVariant) -> dict[Side, list[IRI]]:
    """
    Per side the entities shown: the focus entities of the side and the
    candidates of the other side's focus entities, each once.
    """
    presented: dict[Side, list[IRI]] = {side: [] for side in SIDES}
    for focus in unit.focus:
        side, other_side = Side(focus.side), Side.other_side(Side(focus.side))
        presented[side].append(focus.entity)
        presented[other_side] += [c.entity for c in ordered_candidates(focus, variant, unit.key)]
    return {side: list(dict.fromkeys(entities)) for side, entities in presented.items()}


def format_signals(candidate: ListedCandidate) -> str:
    """
    e.g. 'Pareto layer 1; key: rank 1, vector: rank 3'; for a candidate
    outside the round's list: 'found by a strategy, not in this round's ranked list'.
    """
    if candidate.pareto_layer is None:
        return "found by a strategy, not in this round's ranked list"
    ranks = ", ".join(f"{signal.family}: rank {signal.rank}" for signal in candidate.signals)
    return f"Pareto layer {candidate.pareto_layer}" + (f"; {ranks}" if ranks else "")


def focused_input(state: SearchVerifyState) -> str:
    """
    Each focus entity with its candidates.
    """
    kg_names = state.kg_names
    sections = []
    with_candidates = any(focus.candidates for focus in state.unit.focus)
    for number, focus in enumerate(state.unit.focus, start=1):
        side, other_side = Side(focus.side), Side.other_side(Side(focus.side))
        body = state.entity_displays[side].show(focus.entity)
        candidates = ordered_candidates(focus, state.variant, state.unit.key)
        if candidates:
            lines = []
            for candidate_number, candidate in enumerate(candidates, start=1):
                line = f"{candidate_number}. {state.entity_displays[other_side].show(candidate.entity)}"
                if state.variant.show_signals:
                    line += f"\n   ({format_signals(candidate)})"
                lines.append(line)
            body += f"\n\nCandidates in the {other_side.value} ontology {kg_names[other_side]}:\n" + "\n\n".join(lines)
        sections.append(format_section(
            f"Focus entity {number}, from the {side.value} ontology {kg_names[side]}", body, level=3))
    if with_candidates:
        instruction = (
            "Find the equivalent of each focus entity in the other ontology, or verify that it has none. "
            "Examine its candidates closely first; search the other ontology beyond them only if none of "
            "them fits or doubts remain."
            )
    else:
        instruction = "Find the equivalent of each focus entity in the other ontology, or verify that it has none."
    return instruction + "\n\n" + "\n\n".join(sections)


def clusters_input(state: SearchVerifyState) -> str:
    """The entities of both sides as two lists, without naming the focus."""
    kg_names = state.kg_names
    sections = []
    for side in SIDES:
        entities = state.presented[side]
        body = "\n\n".join(f"{number}. {state.entity_displays[side].show(entity)}"
                           for number, entity in enumerate(entities, start=1))
        sections.append(format_section(
            f"Entities of the {side.value} ontology {kg_names[side]}", body or "none", level=3))
    instruction = (
        "Find the equivalences between the listed entities of the two ontologies. Examine the listed "
        "entities closely against each other first; search the ontologies beyond the lists only if a "
        "listed entity has no fitting counterpart among them or doubts remain."
        )
    return instruction + "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------- checks of set_equivalence

def ask(manager: KgManager, sparql: str) -> bool:
    result = manager.execute_sparql(manager.fix_prefixes(sparql))
    assert isinstance(result, AskResult), "expected an ASK query"
    return result.boolean


def matches_membership(manager: KgManager, entity: IRI, membership_sparql: str) -> bool:
    return ask(manager, f"ASK {{ VALUES ?s {{ {prepare_identifier_for_sparql(entity, manager.iri_literal_parser)} }} {membership_sparql} }}")


def check_entity_space(state: SearchVerifyState, side: Side, entity: IRI) -> None:
    profile = state.alignment_state.get_kg_profile(side)
    if profile.entity_space and not entity.startswith(tuple(profile.entity_space)):
        raise FunctionCallException(
            f"The entity {entity} is not among the entities to be aligned in the {side.value} ontology "
            f"{profile.kg_name}: its IRI starts with none of {profile.entity_space}."
            )


def excluded_group_of(state: SearchVerifyState, side: Side, entity: IRI) -> str | None:
    """The first excluded group the entity is in (None: in none); cached."""
    key = (side, entity)
    if key not in state.excluded_group_cache:
        profile = state.alignment_state.get_kg_profile(side)
        state.excluded_group_cache[key] = next(
            (name for name in profile.excluded_group_names()
             if matches_membership(state.kg_managers[side], entity, profile.get_group(name).definition.membership_sparql)),
            None,
            )
    return state.excluded_group_cache[key]


def check_not_excluded(state: SearchVerifyState, side: Side, entity: IRI) -> None:
    group = excluded_group_of(state, side, entity)
    if group is not None:
        profile = state.alignment_state.get_kg_profile(side)
        reason = profile.get_group(group).exclusion_reason
        raise FunctionCallException(
            f"The entity {entity} is not to be aligned: it belongs to the group {group} of the "
            f"{side.value} ontology {profile.kg_name}, which is excluded from the alignment"
            + (f" ({reason})." if reason else ".")
            )


def partition_of(state: SearchVerifyState, side: Side, entity: IRI) -> str | None:
    """The partition of the entity: the first whose membership it matches (None: in none); cached."""
    key = (side, entity)
    if key not in state.partition_cache:
        profile = state.alignment_state.get_kg_profile(side)
        state.partition_cache[key] = next(
            (partition.name for partition in profile.partitions
             if matches_membership(state.kg_managers[side], entity, partition.membership_sparql)),
            None,
            )
    return state.partition_cache[key]


def non_competing_reason(state: SearchVerifyState, source_entity: IRI, target_entity: IRI) -> str | None:
    """Why the partitions of the two entities do not compete; None: they compete (or one has none)."""
    source_partition = partition_of(state, Side.SOURCE, source_entity)
    target_partition = partition_of(state, Side.TARGET, target_entity)
    if source_partition is None or target_partition is None:
        return None
    if state.alignment_state.competes(Side.SOURCE, source_partition, Side.TARGET, target_partition):
        return None
    reason = next((other.reason for other in state.alignment_state.non_competing_partitions(Side.SOURCE, source_partition)
                   if Side(other.side) == Side.TARGET and other.partition == target_partition), "")
    return (f"The source entity is in the partition {source_partition} and the target entity in the partition "
            f"{target_partition}, which do not compete with each other" + (f": {reason}." if reason else "."))


# ---------------------------------------------------------------- collisions

def partners_in_state(state: SearchVerifyState, side: Side, entity: IRI) -> list[IRI]:
    partners = (equivalence.partner_of(side, entity) for equivalence in state.alignment_state.equivalences_of((side, entity)))
    return [partner for partner in partners if partner is not None]


def partners_in_run(state: SearchVerifyState, side: Side, entity: IRI) -> list[IRI]:
    return [target if side == Side.SOURCE else source
            for source, target in state.equivalences
            if (source if side == Side.SOURCE else target) == entity]


def format_pair(pair: EquivalencePair) -> str:
    return f"'{pair[0]}' = '{pair[1]}'"


def collision_warning(state: SearchVerifyState, pair: EquivalencePair) -> tuple[str, frozenset[str]] | None:
    """The warning for a new pair (None: nothing to warn about), and what it is
    about, to recognize a repeated call."""
    source_entity, target_entity = pair
    kg_names = state.kg_names
    earlier: list[EquivalencePair] = []
    own: list[EquivalencePair] = []
    for partner in partners_in_state(state, Side.SOURCE, source_entity):
        earlier.append((source_entity, partner))
    for partner in partners_in_state(state, Side.TARGET, target_entity):
        earlier.append((partner, target_entity))
    for partner in partners_in_run(state, Side.SOURCE, source_entity):
        own.append((source_entity, partner))
    for partner in partners_in_run(state, Side.TARGET, target_entity):
        own.append((partner, target_entity))
    rejected = target_entity in state.alignment_state.rejected_partners((Side.SOURCE, source_entity))
    non_competing = non_competing_reason(state, source_entity, target_entity)
    if not (earlier or own or rejected or non_competing):
        return None

    message = f"Warning for {format_pair(pair)}:\n"
    if rejected:
        message += "\nThis pair was rejected by an earlier judgment.\n"
    if non_competing:
        message += f"\n{non_competing}\n"
    if earlier or own:
        message += "\nThe entities are already matched:\n"
        message += "".join(f"- {format_pair(p)} (set earlier, not by you)\n" for p in earlier)
        message += "".join(f"- {format_pair(p)} (set by you in this task)\n" for p in own)
        implied = []
        for other_source, other_target in earlier + own:
            if other_target != target_entity:
                implied.append(f"- '{target_entity}' = '{other_target}' (within the target ontology {kg_names[Side.TARGET]})\n")
            if other_source != source_entity:
                implied.append(f"- '{source_entity}' = '{other_source}' (within the source ontology {kg_names[Side.SOURCE]})\n")
        message += "Together with them, the new equivalence would imply:\n" + "".join(dict.fromkeys(implied))
    message += (
        "\nThe new equivalence has NOT been set yet. Compare all involved entities closely and choose "
        "deliberately:\n"
        "- Do not set it: no further action needed.\n"
    )
    if own:
        message += ("- Replace an equivalence you set in this task: delete it with delete_equivalence, then "
                    "call set_equivalence again with the same arguments.\n")
    message += "- Set it anyway: call set_equivalence again with the same arguments."
    if earlier:
        message += (" Equivalences set earlier cannot be deleted here; if you set the new one in addition, "
                    "the conflict is checked later.")
    about = frozenset(
        [f"earlier:{p}" for p in earlier] + [f"own:{p}" for p in own]
        + (["rejected"] if rejected else []) + (["non_competing"] if non_competing else [])
        )
    return message, about


# ---------------------------------------------------------------- functions

def resolve_entity(state: SearchVerifyState, side: Side, iri: Any, known: set[str]) -> IRI:
    """The full IRI of an entity of the side, checked: it exists (know_before_use),
    is in the entity space and in no excluded group."""
    if not isinstance(iri, str) or not iri:
        raise FunctionCallException(f"The {side.value} entity is missing")
    manager = state.kg_managers[side]
    try:
        full_iri = IRI(prepare_entity(manager, iri).identifier)
    except ValueError as error:
        raise FunctionCallException(str(error)) from error
    if state.know_before_use:
        ensure_known(manager, full_iri, known)
    check_entity_space(state, side, full_iri)
    check_not_excluded(state, side, full_iri)
    return full_iri


@func_reg.register(SearchVerifyFunction.SET_EQUIVALENCE)
def set_equivalence(state: SearchVerifyState, fn_args: dict, known: set[str]) -> str:
    pair = (resolve_entity(state, Side.SOURCE, fn_args.get("source_entity"), known),
            resolve_entity(state, Side.TARGET, fn_args.get("target_entity"), known))
    if pair in state.equivalences:
        return f"{format_pair(pair)} is set already."
    warning = collision_warning(state, pair)
    if warning is not None:
        message, about = warning
        warned = state.pending_warnings.get(pair)
        if warned is None or not about <= warned:
            state.pending_warnings[pair] = about
            return message
    state.pending_warnings.pop(pair, None)
    state.equivalences.append(pair)
    return f"Set {format_pair(pair)}." + (" It is set in addition to the equivalences from the warning."
                                          if warning is not None else "")


@func_reg.register(SearchVerifyFunction.DELETE_EQUIVALENCE)
def delete_equivalence(state: SearchVerifyState, fn_args: dict, known: set[str]) -> str:
    source_entity, target_entity = fn_args.get("source_entity"), fn_args.get("target_entity")
    pair = next((p for p in state.equivalences if p == (source_entity, target_entity)), None)
    if pair is None:
        raise FunctionCallException(
            f"You did not set '{source_entity}' = '{target_entity}' in this task; only equivalences you set "
            "here can be deleted (use the full IRIs)."
            )
    state.equivalences.remove(pair)
    return f"Deleted {format_pair(pair)}."


@func_reg.register(SearchVerifyFunction.FINISH)
def finish(state: SearchVerifyState, fn_args: dict, known: set[str]) -> str:
    rationale = fn_args.get("rationale")
    if not rationale:
        raise FunctionCallException("A rationale is required")
    unsure_iris = fn_args.get("unsure") or []
    if not isinstance(unsure_iris, list):
        raise FunctionCallException("unsure must be a list of entity IRIs (it may be empty)")
    # focused: the focus entities can be unsure; clusters: every presented entity
    if state.variant.presentation == "focused":
        allowed = {(Side(focus.side), focus.entity) for focus in state.unit.focus}
    else:
        allowed = {(side, entity) for side in SIDES for entity in state.presented[side]}
    unsure: list[EntityKey] = []
    for iri in unsure_iris:
        keys = [key for key in allowed if key[1] == iri]
        if not keys:
            raise FunctionCallException(
                f"{iri} cannot be marked unsure: only " + ("the focus entities" if state.variant.presentation == "focused"
                                                            else "the listed entities") + " can.")
        unsure += keys
    state.unsure = sorted(set(unsure))
    state.rationale = rationale
    state.finished = True
    return f"Finished with {len(state.equivalences)} equivalences."


def function_definitions(function_names: list[SearchVerifyFunction]) -> list[dict]:
    entity_properties = {
        "source_entity": {"type": "string", "description": "The exact IRI of the source entity."},
        "target_entity": {"type": "string", "description": "The exact IRI of the target entity."},
    }
    definitions = {
        SearchVerifyFunction.SET_EQUIVALENCE: {
            "name": SearchVerifyFunction.SET_EQUIVALENCE,
            "description": "Sets an equivalence between a source and a target entity.",
            "parameters": {"type": "object", "properties": entity_properties,
                           "required": list(entity_properties), "additionalProperties": False},
            "strict": True,
        },
        SearchVerifyFunction.DELETE_EQUIVALENCE: {
            "name": SearchVerifyFunction.DELETE_EQUIVALENCE,
            "description": "Deletes an equivalence you set in this task.",
            "parameters": {"type": "object", "properties": entity_properties,
                           "required": list(entity_properties), "additionalProperties": False},
            "strict": True,
        },
        SearchVerifyFunction.FINISH: {
            "name": SearchVerifyFunction.FINISH,
            "description": "Finishes the task after you have set all equivalences you found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string", "description": "Short explanation of your decisions."},
                    "unsure": {"type": "array", "items": {"type": "string"},
                               "description": "IRIs of the entities you could not decide on; empty if none."},
                },
                "required": ["rationale", "unsure"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    return [definitions[name] for name in function_names]


# ---------------------------------------------------------------- the task

class SearchVerifyTask(GraspTask):
    """Input: a serialized SearchVerifyInput. Output: a SearchVerifyResult."""
    name = "om-search-verify"
    state: SearchVerifyState

    def setup(self, input: Any) -> str:
        try:
            search_verify_input = SearchVerifyInput.model_validate(input)
        except Exception as error:
            raise ValueError(f"Input must be a serialized SearchVerifyInput: {error}") from error
        alignment_state = search_verify_input.alignment_state
        managers = managers_by_side(self, alignment_state.source_kg_name, alignment_state.target_kg_name)
        self.state = SearchVerifyState(
            kg_managers=managers,
            alignment_state=alignment_state,
            unit=search_verify_input.unit,
            variant=search_verify_input.variant,
            entity_displays={
                side: EntityDisplay(managers[side], alignment_state.get_kg_profile(side), LONG_VIEW)
                for side in SIDES
            },
            know_before_use=know_before_use(self.config),
        )
        state = self.state
        state.presented = presented_entities(state.unit, state.variant)
        for side in SIDES:
            display = state.entity_displays[side]
            display.prefetch(state.presented[side])
            # what the model sees is known: the entities and the IRIs in their contexts
            for entity in state.presented[side]:
                self.known.add(entity)
                if entity in display.contexts_by_entity:
                    self.known.update(display.shown_iris(entity))
        if state.variant.presentation == "focused":
            return focused_input(state)
        return clusters_input(state)

    def system_information(self) -> str:
        info = add_equivalence_condition(load_prompt("search_verify_task.txt"), self.config)
        task_instructions = load_task_instructions(self.config)
        if task_instructions:
            info += "\n\n" + format_section("Task-specific instructions", task_instructions)
        return info

    def rules(self) -> list[str]:
        return [
            "Use the exact IRIs of the entities, as shown, in set_equivalence.",
            f"You have at most {self.config.max_steps} steps; one step is one response of yours, with all "
            "tool calls in it. Call finish well before the limit.",
        ]

    def function_definitions(self) -> list[dict]:
        return function_definitions(list(SearchVerifyFunction))

    def final_step_functions(self) -> list[dict] | None:
        # the last step can still set a last equivalence, and must finish
        return function_definitions([SearchVerifyFunction.SET_EQUIVALENCE, SearchVerifyFunction.FINISH])

    def budget_message(self, steps_left: int, final: bool) -> str:
        if final:
            return ("This is your last step. Set the equivalences you are sure of now and call finish, with "
                    "the entities you could not decide on as unsure.")
        return f"You have {steps_left} steps left, including this one. Make sure to call finish before you run out."

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        function: Callable | None = func_reg.handlers.get(fn_name)
        if function is None:
            raise FunctionCallException(
                f"Unknown function {fn_name}, use one of {[name.value for name in SearchVerifyFunction]}")
        return function(self.state, fn_args, known)

    def done(self, fn_name: str) -> bool:
        return fn_name == SearchVerifyFunction.FINISH and self.state.finished

    def output(self, messages: list[Message]) -> dict:
        state = self.state
        return SearchVerifyResult(
            equivalences=list(state.equivalences),
            unsure=state.unsure,
            finished=state.finished,
            rationale=state.rationale,
        ).model_dump(mode="json")
