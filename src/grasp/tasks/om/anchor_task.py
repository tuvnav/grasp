import random
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable
from grasp.manager import KgManager
from grasp.model.base import Message
from grasp.tasks.base import GraspTask
from grasp.functions import find_manager
from grasp.tasks.om.agentic_search_verify import Correspondence, load_task_instructions
from grasp.tasks.om.compare import key_anchors, stratum
from grasp.tasks.om.model import (
    AnchorStrategyType, CandidateStrategyType, FunctionRegistry, KeyAnchor, KeyAnchorParams,
    KeyAnchorResult)
from grasp.tasks.om.retrieve import retrieve_key_rows
from grasp.utils import FunctionCallException, format_section
from pydantic import BaseModel, ValidationError


ANCHOR_STRATEGIES = [
    {
        "name": AnchorStrategyType.KEY_ANCHORS,
        "description": (
            "Each entity gets one or more keys; pairs of "
            "entities that share a key which is unique within both ontologies "
            "become anchors. By default the keys are the entity's names (labels "
            "and synonyms; local names if no label exists), normalized for "
            "case and word boundaries; optional normalization levels: plural, "
            "inflection, or exact (for codes). You can define other keys with "
            "one SPARQL query per ontology, `SELECT ?entity ?key [?role]`, e.g. "
            "a scientific name, a code, a notation, a name under a specific "
            "property, a shared cross-reference, or a composite key such as "
            "name + rank. ?role 'secondary' marks keys that only block "
            "ambiguity (e.g. synonyms) but do not create anchors themselves."
        ),
    },
    {
        "name": AnchorStrategyType.NO_ANCHORS,
        "description": "Only if the ontologies clearly do not describe the same kind of things.",
    },
]

CANDIDATE_STRATEGIES = [
    {
        "name": CandidateStrategyType.EMBEDDING_CANDIDATES,
        "description": (
            "Names are embedded with a multilingual sentence embedding model; pairs that are each other's nearest "
            "neighbour and more similar to each other than to any other entity of their own ontology become "
            "candidates. Works across languages and finds synonyms, but often pairs related, broader or sibling "
            "concepts, so expect medium precision. Fits ontologies without a common language or with few name "
            "anchors."
            ),
        }
]


def render_strategies(strategies: list[dict]) -> str:
    return "\n".join(f"- {s['name']}: {s['description']}" for s in strategies)


class StrategyStatistics(BaseModel):
    ...


class AnchorStrategy(BaseModel):
    type: AnchorStrategyType
    statistics: StrategyStatistics
    correspondences: list[Correspondence]


class AnchorFunctionName(StrEnum):
    TRY_KEY = "try_key_strategy"
    FINISH = "finish"


func_reg = FunctionRegistry()

PROMPT_DIR = Path(__file__).parent / "prompts"
# example anchors per stratum in the LLM view of a result
NUM_EXAMPLES = 5


@dataclass
class AnchorTaskState:
    source: KgManager
    target: KgManager
    # run id -> parameters and result of a tried strategy; "run-0" is the default
    runs: dict[str, tuple[KeyAnchorParams, KeyAnchorResult]] = field(default_factory=dict)
    # set by finish: [(run id, selected group labels)]
    selection: list[tuple[str, list[str]]] | None = None
    rationale: str | None = None

    def add_run(self, params: KeyAnchorParams, result: KeyAnchorResult) -> str:
        run_id = f"run-{len(self.runs)}"
        self.runs[run_id] = (params, result)
        return run_id


def run_key_anchors(
        manager_source: KgManager,
        manager_target: KgManager,
        params: KeyAnchorParams,
        ) -> KeyAnchorResult:
    """Computes the key anchors; used directly by the pipeline (default run)."""
    source_rows = retrieve_key_rows(manager_source, params.source)
    target_rows = retrieve_key_rows(manager_target, params.target)
    return key_anchors(source_rows, target_rows, params.extended, params.pairing)


def format_anchor(anchor: KeyAnchor, state: AnchorTaskState) -> str:
    text = (
        f"{state.source.format_iri(anchor.source_iri)} \"{anchor.source_key}\" = "
        f"{state.target.format_iri(anchor.target_iri)} \"{anchor.target_key}\""
    )
    if anchor.head:
        text += f" (information content {anchor.information_content:.2f})"
    return text


def format_key_anchor_result(run_id: str, result: KeyAnchorResult, state: AnchorTaskState) -> str:
    """LLM view of a result: statistics per partition group and a few example
    anchors per stratum; the full anchor lists stay in the task state."""
    lines = [f"Result {run_id}:"]
    by_stratum: dict[tuple[str, str], list[KeyAnchor]] = {}
    for anchor in result.anchors:
        by_stratum.setdefault(stratum(anchor), []).append(anchor)

    for group, s in result.group_statistics.items():
        lines.append(
            f"Group {group.label}: {s.n_source} source / {s.n_target} target entities, "
            f"{s.anchors} anchors (specific {s.specific}, head {s.head}, extended {s.extended}), "
            f"anchor rate {s.anchor_rate:.2f}, token overlap {s.token_overlap:.2f}"
        )
        for level, ls in s.levels.items():
            lines.append(
                f"  level {level}: {ls.new_anchors} new anchors, new internal collisions "
                f"{ls.new_collisions_source} source / {ls.new_collisions_target} target, "
                f"ambiguous {ls.ambiguous_source} source / {ls.ambiguous_target} target"
            )
        for (label, name), anchors in by_stratum.items():
            if label != group.label:
                continue
            # fixed seed: the same result always shows the same examples
            examples = random.Random(0).sample(anchors, min(NUM_EXAMPLES, len(anchors)))
            lines.append(f"  examples {name} ({len(anchors)} anchors):")
            lines.extend(f"    {format_anchor(a, state)}" for a in examples)

    if not result.group_statistics:
        lines.append("No partition groups with entities on both sides.")
    for name, partitions in [
        ("Unpaired source partitions", result.unpaired_source),
        ("Unpaired target partitions", result.unpaired_target),
        ("Pairing names unknown source partitions", result.missing_source),
        ("Pairing names unknown target partitions", result.missing_target),
    ]:
        if partitions:
            lines.append(f"{name}: {', '.join(partitions)}")
    return "\n".join(lines)


@func_reg.register(AnchorFunctionName.TRY_KEY)
def try_key_strategy(state: AnchorTaskState, fn_args: dict) -> str:
    """Tool wrapper: LLM arguments (JSON) -> KeyAnchorParams -> result -> text."""
    try:
        params = KeyAnchorParams.model_validate(fn_args)
    except ValidationError as e:
        raise FunctionCallException(f"Invalid arguments for {AnchorFunctionName.TRY_KEY}:\n{e}") from e
    try:
        result = run_key_anchors(state.source, state.target, params)
    except (ValueError, RuntimeError) as e:
        # e.g. overlapping partition groups, SPARQL syntax errors (GRASP's parser),
        # conflicting prefixes (merge_prefixes) or errors of the endpoint
        raise FunctionCallException(str(e)) from e
    run_id = state.add_run(params, result)
    return format_key_anchor_result(run_id, result, state)


class Selection(BaseModel):
    run_id: str
    groups: list[str] | None = None


class FinishArgs(BaseModel):
    selected: list[Selection]
    rationale: str


@func_reg.register(AnchorFunctionName.FINISH)
def finish(state: AnchorTaskState, fn_args: dict) -> str:
    try:
        args = FinishArgs.model_validate(fn_args)
    except ValidationError as e:
        raise FunctionCallException(f"Invalid arguments for {AnchorFunctionName.FINISH}:\n{e}") from e

    selection = []
    for sel in args.selected:
        if sel.run_id not in state.runs:
            raise FunctionCallException(
                f"Unknown result {sel.run_id}, must be one of {list(state.runs)}"
            )
        _, result = state.runs[sel.run_id]
        labels = [g.label for g in result.group_statistics]
        groups = labels if sel.groups is None else sel.groups
        unknown = [g for g in groups if g not in labels]
        if unknown:
            raise FunctionCallException(
                f"Result {sel.run_id} has no groups {unknown}, its groups are {labels}"
            )
        selection.append((sel.run_id, groups))

    state.selection = selection
    state.rationale = args.rationale
    n = sum(
        1 for run_id, groups in selection
        for a in state.runs[run_id][1].anchors if a.group.label in groups
    )
    return f"Selected {n} anchors from {len(selection)} result(s)."


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def system_information(task_instructions: str | None) -> str:
    info = load_prompt("anchor_strategies.txt").replace(
        "__EQUIVALENCE_CONDITION__", load_prompt("equivalence_condition.txt")
    )
    if task_instructions:
        info += "\n\n" + format_section("Task-specific instructions", task_instructions)
    return info


def anchor_to_dict(anchor: KeyAnchor) -> dict:
    return {
        "source": anchor.source_iri,
        "target": anchor.target_iri,
        "group": anchor.group.label,
        "stratum": stratum(anchor)[1],
        "level": anchor.string_normalization_level,
        "evidence": anchor.evidence,
        "information_content": round(anchor.information_content, 3),
    }


SIDE_QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {
            "type": "string",
            "description": (
                "SPARQL SELECT query with the variables ?entity and ?key, optionally ?role. "
                "Each row gives one key of an entity. ?role \"secondary\" marks keys that only "
                "block ambiguous anchors but do not create anchors themselves (e.g. synonyms); "
                "rows without ?role are primary. Omit to use the default name keys: "
                "rdfs:label and skos:prefLabel as primary keys, skos:altLabel and oboInOwl "
                "synonyms as secondary keys, the local name of the IRI as primary key for "
                "entities without a label."
            ),
        },
        "scope": {
            "type": ["string", "null"],
            "description": (
                "SPARQL SELECT query with the variable ?entity. Only these entities take part. "
                "Omit or null to use all entities of the key query."
            ),
        },
        "partition": {
            "type": ["string", "null"],
            "description": (
                "SPARQL SELECT query with the variables ?entity and ?partition (a literal or an "
                "IRI). The anchor rule runs separately per partition group, and anchors are "
                "certified per partition group. An entity may be in several partitions; "
                "entities without a partition become \"uncategorized\", which is only compared "
                "if a pairing group names it. Omit to use the default partitions by kind: "
                "class, property, instance, concept (SKOS), from declared types and usage. "
                "null puts all entities into one partition \"all\"."
            ),
        },
    },
    "additionalProperties": False,
}


def anchor_functions() -> list[dict]:
    return [
        {
            "name": AnchorFunctionName.TRY_KEY,
            "description": (
                "Computes key anchors between the source and the target ontology and returns "
                "statistics per partition group. An anchor is a pair of entities that share a "
                "primary key which is unique among all keys of its partition group on both "
                "sides; an entity with unique keys towards several partners gets no anchor. "
                "Keys are compared on increasingly coarse normalization levels (raw; case; "
                "word boundaries such as CamelCase, _ and -; spaceless). Each anchor gets the "
                "finest level on which it is unique; coarser levels only add entities without "
                "an anchor. Anchors whose key is extended by another name of the same "
                "ontology (e.g. 'paper' vs. 'poster paper') are counted as head anchors, "
                "since their scope may differ between the ontologies. Call without arguments "
                "for the default strategy (name keys, partitions by kind)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        **SIDE_QUERIES_SCHEMA,
                        "description": "Queries for the source ontology. Omitted fields use the defaults.",
                    },
                    "target": {
                        **SIDE_QUERIES_SCHEMA,
                        "description": (
                            "Queries for the target ontology. Starts as a copy of the source "
                            "queries; only the fields given here replace them. Omit if the same "
                            "queries apply to both ontologies."
                        ),
                    },
                    "extended": {
                        "type": "boolean",
                        "description": (
                            "Also compare keys on the extended normalization levels: word order, "
                            "singular/plural and verb inflection. Their anchors form a separate "
                            "stratum. Default false."
                        ),
                    },
                    "pairing": {
                        "type": ["array", "null"],
                        "description": (
                            "Which source partitions are compared with which target partitions. "
                            "Uniqueness holds within the union of the partitions of each side of "
                            "a group. Each partition may occur in at most one group per side. "
                            "Omit to compare each partition with the partition of the same name, "
                            "e.g. use [{\"source\": [\"class\"], \"target\": [\"class\", "
                            "\"concept\"]}] to compare OWL classes with SKOS concepts."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "array", "items": {"type": "string"}},
                                "target": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["source", "target"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            # optional nested fields are not possible in strict mode
            "strict": False,
        },
        {
            "name": AnchorFunctionName.FINISH,
            "description": (
                "Submit your decision and end the task. Select the results (by their id, e.g. "
                "run-0 for the default strategy) and optionally only some of their partition "
                "groups whose anchors should be used. Every selected group is verified "
                "afterwards by sampling per stratum. Select nothing if no anchor set is "
                "plausibly mostly correct."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selected": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "run_id": {"type": "string", "description": "Id of a tried result, e.g. run-0"},
                                "groups": {
                                    "type": ["array", "null"],
                                    "items": {"type": "string"},
                                    "description": (
                                        "Labels of the partition groups to use, as shown in the "
                                        "result (e.g. \"class <-> class\"); omit or null for all groups"
                                    ),
                                },
                            },
                            "required": ["run_id"],
                            "additionalProperties": False,
                        },
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Short explanation of the selection: which keys and partitions, why, "
                            "and which anchors you expect to be risky."
                        ),
                    },
                },
                "required": ["selected", "rationale"],
                "additionalProperties": False,
            },
            "strict": False,
        },
    ]


class OMAnchorTask(GraspTask):
    """First stage of the orchestrator: choose anchor strategies. Input: a dict
    with source_kg and target_kg (names of the configured knowledge graphs); by
    default the first configured KG is the source, the second the target."""
    name = "om-anchors"

    def setup(self, input: Any) -> str:
        source, target = self.resolve_managers(input)
        self.state = AnchorTaskState(source, target)
        params = KeyAnchorParams()
        result = run_key_anchors(source, target, params)
        run_id = self.state.add_run(params, result)
        return (
            f"Source ontology: {source.kg}"
            + (f" ({source.description})" if source.description else "")
            + f"\nTarget ontology: {target.kg}"
            + (f" ({target.description})" if target.description else "")
            + "\n\nThe default strategy (name keys, partitions by kind) has already been run:\n\n"
            + format_key_anchor_result(run_id, result, self.state)
        )

    def resolve_managers(self, input: Any) -> tuple[KgManager, KgManager]:
        if isinstance(input, dict) and "source_kg" in input and "target_kg" in input:
            source, _ = find_manager(self.managers, input["source_kg"])
            target, _ = find_manager(self.managers, input["target_kg"])
            return source, target
        if len(self.managers) != 2:
            raise ValueError(
                "Input must name source_kg and target_kg unless exactly two KGs are configured"
            )
        return self.managers[0], self.managers[1]

    def system_information(self) -> str:
        return system_information(
            load_task_instructions(self.config.task_kwargs.get("om", {}))
        )

    def rules(self) -> list[str]:
        return [
            "Look at the default result and its examples first. Inspect the ontologies with "
            "the SPARQL and search functions before you define your own key, scope or "
            "partition queries.",
            "Test every strategy you want to select with try_key_strategy; you can only "
            "select results that you have tried.",
            "Call finish exactly once, at the end.",
        ]

    def function_definitions(self) -> list[dict]:
        return anchor_functions()

    def output(self, messages: list[Message]) -> dict:
        state: AnchorTaskState = self.state
        selected = []
        anchors = []
        for run_id, groups in state.selection or []:
            params, result = state.runs[run_id]
            selected.append({
                "run_id": run_id,
                "groups": groups,
                "params": params.model_dump(),
                "statistics": {
                    g.label: {k: v for k, v in vars(s).items() if k != "levels"}
                    for g, s in result.group_statistics.items() if g.label in groups
                },
            })
            anchors.extend(
                {"run_id": run_id, **anchor_to_dict(a)}
                for a in result.anchors if a.group.label in groups
            )
        return {
            "source_kg": state.source.kg,
            "target_kg": state.target.kg,
            "finished": state.selection is not None,
            "rationale": state.rationale,
            "runs_tried": len(state.runs) - 1,
            "selected": selected,
            "anchors": anchors,
        }

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        fn: Callable | None = func_reg.Registry.get(fn_name)
        if fn is None:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return fn(self.state, fn_args)

    def done(self, fn_name: str) -> bool:
        return fn_name == AnchorFunctionName.FINISH and self.state.selection is not None

    @property
    def default_input_field(self) -> str | None:
        return None
