import random
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable
from grasp.manager import KgManager
from grasp.model.base import Message
from grasp.tasks.base import GraspTask
from grasp.functions import find_manager
from grasp.tasks.om.compare import key_anchors, stratum
from grasp.tasks.om.model import (
    IRI, PROMPT_DIR, UNCATEGORIZED, AnchorConflict,
    AnchorRecord, AnchorStageResult, FunctionRegistry,
    KeyAnchor, KeyAnchorParams, SelectedRun, 
    TryKeyAnchorResult, SIDES, load_prompt, load_task_instructions, resolve_managers)
from grasp.tasks.om.retrieve import namespace_counts, retrieve_key_rows
from grasp.utils import FunctionCallException, format_section
from pydantic import BaseModel, ValidationError


# TODO: Move this to central place
# example anchors per stratum in the LLM view of a result
NUM_EXAMPLES = 5
# examples per kind of difference in the comparison with another result
NUM_DIFF_EXAMPLES = 3


@dataclass
class AnchorTaskState:
    source: KgManager
    target: KgManager
    # run id -> parameters and result of a tried strategy; "run-0" is the default
    runs: dict[str, tuple[KeyAnchorParams, TryKeyAnchorResult]] = field(default_factory=dict)
    # set by finish: [(run id, selected group labels)]
    selection: list[tuple[str, list[str]]] | None = None
    rationale: str | None = None
    # per side: IRI prefixes of the entities to align; all runs are restricted to them
    entity_space: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # per side: entities per namespace in the default run, before the restriction
    namespaces: dict[str, Counter[str]] = field(default_factory=dict)
    # set by finish: the selected anchors merged into Venn cells, and conflicts
    merged: "MergedSelection | None" = None

    def add_run(self, params: KeyAnchorParams, result: TryKeyAnchorResult) -> str:
        run_id = f"run-{len(self.runs)}"
        self.runs[run_id] = (params, result)
        return run_id


def run_key_anchors(
        manager_source: KgManager,
        manager_target: KgManager,
        params: KeyAnchorParams,
        source_space: tuple[str, ...] | None = None,
        target_space: tuple[str, ...] | None = None,
        ) -> TryKeyAnchorResult:
    """Computes the key anchors; used directly by the pipeline (default run)."""
    source_rows = retrieve_key_rows(manager_source, params.source, source_space)
    target_rows = retrieve_key_rows(manager_target, params.target, target_space)
    return key_anchors(source_rows, target_rows, params.extended, params.pairing)


def determine_entity_space(counts: Counter[str], configured: list[str] | None) -> tuple[str, ...]:
    """Configured IRI prefixes, otherwise the namespace with the most entities."""
    if configured:
        return tuple(configured)
    if not counts:
        return ()
    return (counts.most_common(1)[0][0],)


def format_entity_space(state: AnchorTaskState) -> str:
    lines = ["Entity space (only these entities are aligned, in all strategies):"]
    for side in SIDES:
        manager: KgManager = getattr(state, side)
        space = state.entity_space[side]
        counts = ", ".join(
            f"{ns}: {n}" + (" (included)" if ns.startswith(space) or any(p.startswith(ns) for p in space) else " (excluded)")
            for ns, n in state.namespaces[side].most_common()
        )
        lines.append(f"  {side} {manager.kg}: {', '.join(space)}; entities per namespace: {counts}")
    return "\n".join(lines)


def diff_results(
        reference: TryKeyAnchorResult,
        result: TryKeyAnchorResult,
        ) -> tuple[list, list, list, list, list]:
    """Pairs kept, new, with a changed partner, lost with an entity no longer
    compared, and lost although both entities are still compared."""
    old = {(a.source_iri, a.target_iri): a for a in reference.anchors}
    new = {(a.source_iri, a.target_iri): a for a in result.anchors}
    old_by_src = {a.source_iri: a for a in old.values()}
    old_by_tgt = {a.target_iri: a for a in old.values()}

    kept = [new[p] for p in new.keys() & old.keys()]
    added, changed = [], []
    replaced: set[tuple[str, str]] = set()
    for pair in sorted(new.keys() - old.keys()):
        before = old_by_src.get(pair[0]) or old_by_tgt.get(pair[1])
        if before is None:
            added.append(new[pair])
        else:
            changed.append((before, new[pair]))
            replaced.add((before.source_iri, before.target_iri))
    lost_gone, lost_present = [], []
    for pair in sorted(old.keys() - new.keys() - replaced):
        if pair[0] in result.source_entities and pair[1] in result.target_entities:
            lost_present.append(old[pair])
        else:
            lost_gone.append(old[pair])
    return kept, added, changed, lost_gone, lost_present


def format_diff(ref_id: str, reference: TryKeyAnchorResult, result: TryKeyAnchorResult, state: AnchorTaskState) -> str:
    kept, added, changed, lost_gone, lost_present = diff_results(reference, result)
    lines = [
        f"Compared to {ref_id}: {len(kept)} anchors kept, {len(added)} new, {len(changed)} with "
        f"another partner, {len(lost_gone) + len(lost_present)} lost ({len(lost_gone)} because an "
        f"entity is no longer compared, {len(lost_present)} although both entities are still "
        f"compared, e.g. through new collisions or ambiguities)"
    ]
    rng = random.Random(0)

    def sample(items: list) -> list:
        return rng.sample(items, min(NUM_DIFF_EXAMPLES, len(items)))

    for name, items in [("new", added), ("lost, entity no longer compared", lost_gone),
                        ("lost, entities still compared", lost_present)]:
        if items:
            lines.append(f"  examples {name}:")
            lines.extend(f"    {format_anchor(a, state)}" for a in sample(items))
    if changed:
        lines.append(f"  examples with another partner ({ref_id} -> this result):")
        lines.extend(
            f"    {format_anchor(before, state)}  ->  {format_anchor(after, state)}"
            for before, after in sample(changed)
        )
    return "\n".join(lines)


def format_anchor(anchor: KeyAnchor, state: AnchorTaskState) -> str:
    text = (
        f"{state.source.format_iri(anchor.source_iri)} \"{anchor.source_key}\" = "
        f"{state.target.format_iri(anchor.target_iri)} \"{anchor.target_key}\""
    )
    if anchor.head:
        text += f" (information content {anchor.information_content:.2f})"
    return text


def format_key_anchor_result(run_id: str, result: TryKeyAnchorResult, state: AnchorTaskState) -> str:
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


class AnchorFunctionName(StrEnum):
    TRY_KEY = "try_key_strategy"
    FINISH = "finish"


func_reg = FunctionRegistry()


@func_reg.register(AnchorFunctionName.TRY_KEY)
def try_key_strategy(state: AnchorTaskState, fn_args: dict) -> str:
    """Tool wrapper: LLM arguments (JSON) -> KeyAnchorParams -> result -> text."""
    # display option, not part of the strategy
    fn_args = dict(fn_args)
    ref_id = fn_args.pop("compare_to", None) or "run-0"
    if ref_id not in state.runs:
        raise FunctionCallException(f"Unknown result {ref_id}, must be one of {list(state.runs)}")
    try:
        params = KeyAnchorParams.model_validate(fn_args)
    except ValidationError as e:
        raise FunctionCallException(f"Invalid arguments for {AnchorFunctionName.TRY_KEY}:\n{e}") from e
    try:
        result = run_key_anchors(
            state.source, state.target, params,
            state.entity_space.get("source"), state.entity_space.get("target"),
        )
    except (ValueError, RuntimeError) as e:
        # e.g. overlapping partition groups, SPARQL syntax errors (GRASP's parser),
        # conflicting prefixes (merge_prefixes) or errors of the endpoint
        raise FunctionCallException(str(e)) from e
    run_id = state.add_run(params, result)
    return (
        format_key_anchor_result(run_id, result, state)
        + "\n\n" + format_diff(ref_id, state.runs[ref_id][1], result, state)
    )


class Selection(BaseModel):
    run_id: str
    groups: list[str] | None = None


class FinishArgs(BaseModel):
    selected: list[Selection]
    rationale: str


# stratum types from least to most risky; a pair in several results gets the riskiest
STRATUM_RISK = {"standard_specific": 0, "standard_head": 1, "extended": 2}


@dataclass
class VennCell:
    """Venn cell of the selection: the pairs found by exactly the same selected
    results and groups, split by their riskiest stratum type. Cells are disjoint
    and are the strata of the certification."""
    memberships: tuple[tuple[str, str], ...]  # (run id, group label), sorted
    stratum_type: str
    anchors: list[tuple[str, KeyAnchor]] = field(default_factory=list)  # (run id, anchor)

    @property
    def label(self) -> str:
        runs = " & ".join(f"{run_id} {group}" for run_id, group in self.memberships)
        return f"[{runs}] {self.stratum_type}"


@dataclass
class MergedSelection:
    cells: list[VennCell]
    # pairs whose source or target has another partner in the selection;
    # (pair, run ids) - no anchors, to be checked individually
    conflicts: list[tuple[tuple[str, str], list[str]]]


# TODO: 1:1-restriction is to strict
def merge_selection(state: AnchorTaskState, selection: list[tuple[str, list[str]]]) -> MergedSelection:
    """Splits the selected anchors into Venn cells by the results that found them,
    instead of prioritizing one result over another. Pairs in 1:1 conflict across
    the selection become no anchors."""
    found: dict[tuple[str, str], list[tuple[str, KeyAnchor]]] = {}
    for run_id, groups in selection:
        for anchor in state.runs[run_id][1].anchors:
            if anchor.group.label in groups:
                found.setdefault((anchor.source_iri, anchor.target_iri), []).append((run_id, anchor))

    partners_src: dict[str, set[str]] = {}
    partners_tgt: dict[str, set[str]] = {}
    for s, t in found:
        partners_src.setdefault(s, set()).add(t)
        partners_tgt.setdefault(t, set()).add(s)

    cells: dict[tuple, VennCell] = {}
    conflicts = []
    for pair, hits in sorted(found.items()):
        if len(partners_src[pair[0]]) > 1 or len(partners_tgt[pair[1]]) > 1:
            conflicts.append((pair, sorted({run_id for run_id, _ in hits})))
            continue
        memberships = tuple(sorted({(run_id, a.group.label) for run_id, a in hits}))
        stratum_type = max((stratum(a)[1] for _, a in hits), key=STRATUM_RISK.__getitem__)
        key = (memberships, stratum_type)
        cells.setdefault(key, VennCell(memberships, stratum_type)).anchors.extend(hits[:1])
    return MergedSelection(
        cells=sorted(cells.values(), key=lambda c: (-len(c.anchors), c.label)),
        conflicts=conflicts,
    )


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

    # the default result is always verified with all its groups, in addition to
    # the selection; a result selected several times: union of its groups
    merged_groups: dict[str, list[str]] = {}
    if "run-0" in state.runs:
        merged_groups["run-0"] = [g.label for g in state.runs["run-0"][1].group_statistics]
    for run_id, groups in selection:
        merged_groups.setdefault(run_id, [])
        merged_groups[run_id] += [g for g in groups if g not in merged_groups[run_id]]
    selection = list(merged_groups.items())

    state.selection = selection
    state.rationale = args.rationale
    state.merged = merge_selection(state, selection)

    n = sum(len(c.anchors) for c in state.merged.cells)
    lines = [
        f"Selected {n} anchors from {len(selection)} result(s), in {len(state.merged.cells)} "
        f"disjoint strata (by the results that found them and the riskiest stratum type), "
        f"each verified separately:"
    ]
    lines.extend(f"  {c.label}: {len(c.anchors)}" for c in state.merged.cells)
    if state.merged.conflicts:
        lines.append(
            f"{len(state.merged.conflicts)} pairs are in conflict across the results (an entity "
            f"with different partners); they are not used as anchors but checked individually."
        )
    return "\n".join(lines)


def system_information(task_instructions: str | None) -> str:
    info = load_prompt("anchor_task.txt").replace(
        "__EQUIVALENCE_CONDITION__", load_prompt("equivalence_condition.txt")
    )
    if task_instructions:
        info += "\n\n" + format_section("Task-specific instructions", task_instructions)
    return info


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
                    "compare_to": {
                        "type": "string",
                        "description": (
                            "Id of a tried result to compare the new result with (kept, new, "
                            "changed and lost anchors). Default run-0, the default strategy."
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
                "Submit your decision and end the task. The default result run-0 is always used "
                "with all its groups; select the results you want to use in addition (by their "
                "id) and optionally only some of their partition groups. The selected anchors are split into "
                "disjoint strata by the results that found them (e.g. pairs found by two "
                "results, pairs found by only one of them) and by stratum type; each stratum is "
                "verified separately by sampling. Pairs whose entities have different partners "
                "in different results are not used as anchors. Select nothing if no anchor set "
                "is plausibly mostly correct, apart from run-0."
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
    state: AnchorTaskState

    def setup(self, input: Any) -> str:
        source, target = resolve_managers(self, input)
        self.state = state = AnchorTaskState(source=source, target=target)
        params = KeyAnchorParams()
        # default run: retrieve unrestricted once, to determine the entity space
        rows = {
            "source": list(retrieve_key_rows(source, params.source)),
            "target": list(retrieve_key_rows(target, params.target)),
        }
        configured = self.config.task_kwargs.get("om", {}).get("scope", {}).get("namespaces", {})
        for side in SIDES:
            # helper nodes the partition query does not cover would distort the counts
            state.namespaces[side] = namespace_counts(
                r for r in rows[side] if r.partition != UNCATEGORIZED
            )
            state.entity_space[side] = determine_entity_space(
                state.namespaces[side], configured.get(side)
            )
            rows[side] = [r for r in rows[side] if r.entity.startswith(state.entity_space[side])]
        result = key_anchors(rows["source"], rows["target"], params.extended, params.pairing)
        run_id = state.add_run(params, result)
        return (
            f"Source ontology: {source.kg}"
            + (f" ({source.description})" if source.description else "")
            + f"\nTarget ontology: {target.kg}"
            + (f" ({target.description})" if target.description else "")
            + "\n\n" + format_entity_space(state)
            + "\n\nThe default strategy (name keys, partitions by kind) has already been run:\n\n"
            + format_key_anchor_result(run_id, result, state)
        )

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
            "You can execute multiple tool calls in one step. Prefer multiple tool calls "
            "and sparql queries in one round over single tool calls per step.",
            "Call finish exactly once, at the end.",
            f"You have at most {self.config.max_steps} steps; one step is one response of yours, "
            "with all tool calls in it. Use about a third of them to explore the ontologies, "
            "then test your ideas with try_key_strategy and call finish well before the "
            "limit. The default result run-0 is always used with all its groups; if you run "
            "out of steps without calling finish, it is the only one.",
        ]

    def function_definitions(self) -> list[dict]:
        return anchor_functions()

    def output(self, messages: list[Message]) -> dict:
        state: AnchorTaskState = self.state
        finished = state.selection is not None
        if not finished and "run-0" in state.runs:
            # step limit reached without finish: fall back to the default result
            groups = [g.label for g in state.runs["run-0"][1].group_statistics]
            state.selection = [("run-0", groups)]
            state.merged = merge_selection(state, state.selection)
        selected = []
        for run_id, groups in state.selection or []:
            params, result = state.runs[run_id]
            selected.append(
                SelectedRun(
                    run_id=run_id,
                    groups=groups,
                    params=params,
                    statistics={
                        g.label: {k: v for k, v in vars(s).items() if k != "levels"}
                        for g, s in result.group_statistics.items() if g.label in groups
                        }
                )
                )
        merged = state.merged or MergedSelection([], [])
        anchors = [
            AnchorRecord(**a.model_dump(), stratum=cell.label, run_id=run_id)
            for cell in merged.cells for run_id, a in cell.anchors
        ]
        return AnchorStageResult(
            source_kg=state.source.kg,
            target_kg=state.target.kg,
            entity_space={side: list(state.entity_space.get(side, ())) for side in SIDES},
            finished=finished,
            fallback=None if finished else "default",
            rationale=state.rationale,
            runs_tried=len(state.runs) - 1,
            selected=selected,
            strata={cell.label: len(cell.anchors) for cell in merged.cells},
            anchors=anchors,
            conflicts=[
                AnchorConflict(source_iri=IRI(s), target_iri=IRI(t), runs=runs) for (s, t), runs in merged.conflicts
            ]
        ).model_dump(mode="json")

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
