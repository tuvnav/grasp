"""
Profile stage of the orchestrator (om-profile): the LLM checks the default
profile of one KG (kg_profile.default_profile) against the KG, and corrects and
completes it. Output: the profile, the default profile and the statistics.
"""
from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Any, Callable

from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.kg_profile import (
    ROLE_PREDICATES, PartitionStats, default_profile, partition_query, space_filter,
)
from grasp.tasks.om.model import (
    DEFAULT_STRUCTURE_RELATIONS, LABEL_PREDICATES, FunctionRegistry, KgPartitionProfile, KgProfile,
    PredicatePattern, sparql_path, get_om_task_config, load_prompt, load_task_instructions,
)
from grasp.tasks.om.render import clip
from grasp.tasks.om.retrieve import select_rows
from grasp.utils import FunctionCallException, format_section
from pydantic import BaseModel, ValidationError

NUM_EXAMPLES = 8
MAX_VALUE_CHARS = 80


class ProfileFunctionName(StrEnum):
    TRY_PATTERN = "try_pattern"
    FINISH = "finish"


@dataclass
class ProfileTaskState:
    manager: KgManager
    default: KgProfile
    stats: dict[str, PartitionStats]
    profile: KgProfile | None = None
    rationale: str | None = None


func_reg = FunctionRegistry()


def inline_schema(model: type[BaseModel], drop: set[str] = frozenset()) -> dict:
    """JSON schema of a model with its $defs inlined (not every model API
    resolves $ref), without titles and without the given properties."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    schema = resolve(schema)

    def without(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: without(v) for k, v in node.items()}
            if "properties" in node:
                node["properties"] = {k: v for k, v in node["properties"].items() if k not in drop}
                if "required" in node:
                    node["required"] = [r for r in node["required"] if r not in drop]
            return node
        if isinstance(node, list):
            return [without(v) for v in node]
        return node

    return without(schema)


def partition_patch_schema() -> dict:
    """KgPartitionProfile with all fields optional: given fields replace the default."""
    schema = inline_schema(KgPartitionProfile, drop={"source"})
    schema.pop("required", None)
    return schema


def profile_functions() -> list[dict]:
    return [
        {
            "name": ProfileFunctionName.TRY_PATTERN,
            "description": (
                "Tests a predicate pattern on one partition of the knowledge graph: how many of "
                "its entities are ?s of the pattern, how many distinct values ?o it has, how many "
                "of them are literals, and examples. Use it for attributes (names, definitions, "
                "codes: ?o is a literal or a code) as well as for structure relations (?o is "
                "another entity)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "partition": {"type": "string", "description": "Partition name, e.g. class or instance."},
                    "predicate_pattern": inline_schema(PredicatePattern),
                },
                "required": ["partition", "predicate_pattern"],
                "additionalProperties": False,
            },
        },
        {
            "name": ProfileFunctionName.FINISH,
            "description": (
                "Submits the profile. Only the partitions and fields given replace the default "
                "profile; a list given replaces the whole default list (so repeat the default "
                "entries you keep). Omitted partitions and fields keep the default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "partitions": {
                        "type": "object",
                        "description": "Partition name -> changed fields of its profile.",
                        "additionalProperties": partition_patch_schema(),
                    },
                    "entity_space": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IRI prefixes of the entities to align, if the default is wrong.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "What you changed and why, and what remains uncertain.",
                    },
                },
                "required": ["rationale"],
                "additionalProperties": False,
            },
        },
    ]


def run_rows(manager: KgManager, sparql: str) -> list[dict]:
    try:
        return select_rows(manager, sparql)
    except Exception as e:
        raise FunctionCallException(f"The SPARQL query for the pattern failed: {e}") from e


@func_reg.register(ProfileFunctionName.TRY_PATTERN)
def try_pattern(state: ProfileTaskState, fn_args: dict) -> str:
    partition = fn_args.get("partition")
    if partition not in state.default.partitions:
        raise FunctionCallException(
            f"Unknown partition {partition}; partitions: {', '.join(state.default.partitions)}"
        )
    try:
        pattern = PredicatePattern.model_validate(fn_args.get("predicate_pattern") or {})
    except ValidationError as e:
        raise FunctionCallException(f"Invalid predicate_pattern: {e}") from e
    manager, space = state.manager, tuple(state.default.alignment_namespaces)
    where = (f'FILTER(?partition = "{partition}")\n{space_filter(space)}\n'
             f"BIND(?entity AS ?s)\n{pattern.predicates_to_sparql()}")
    counts = run_rows(manager, partition_query(
        manager,
        "(COUNT(DISTINCT ?s) AS ?subjects) (COUNT(DISTINCT ?o) AS ?objects) "
        "(SUM(IF(isLiteral(?o), 1, 0)) AS ?literals) (COUNT(*) AS ?n)",
        where,
    ))
    row = counts[0] if counts else {}
    value = lambda key: int(row[key].value) if key in row else 0
    size = state.stats[partition].size
    subjects = value("subjects")
    lines = [
        f"Pattern: {pattern.predicates_to_sparql()}",
        f"Partition {partition}: {subjects} of {size} entities are ?s"
        + (f" ({subjects / size:.0%})" if size else "")
        + f"; {value('n')} matches, {value('objects')} distinct ?o, {value('literals')} literal matches.",
    ]
    if subjects:
        lines.append("Examples (?s -> ?o, with names in quotes):")
        lines += format_examples(state, partition, where)
    return "\n".join(lines)


def format_examples(state: ProfileTaskState, partition: str, where: str) -> list[str]:
    """Example matches with a name of ?s and of ?o (codes alone say little)."""
    manager = state.manager
    names = sparql_path(list(dict.fromkeys(state.default.partitions[partition].name_predicates + LABEL_PREDICATES)))
    rows = run_rows(manager, partition_query(
        manager, "?s ?o ?s_name ?o_name",
        f"{where}\nOPTIONAL {{ ?s {names} ?s_name }}\nOPTIONAL {{ ?o {names} ?o_name }}",
        f"LIMIT {NUM_EXAMPLES * 4}",
    ))
    seen, lines = set(), []
    for row in rows:
        if "s" not in row or "o" not in row or (row["s"].value, row["o"].value) in seen:
            continue
        seen.add((row["s"].value, row["o"].value))
        named = lambda var: row[var].value + (f' "{clip(row[var + "_name"].value, 40)}"' if var + "_name" in row else "")
        lines.append(f"  {named('s')} -> {clip(named('o'), MAX_VALUE_CHARS)}")
        if len(lines) == NUM_EXAMPLES:
            break
    return lines


def check_structure_names(partition: str, profile: KgPartitionProfile) -> None:
    names = [r.name for r in profile.structure_predicates]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise FunctionCallException(f"Partition {partition}: duplicate structure relation names {sorted(duplicates)}")
    unknown = {r.inverse_of for r in profile.structure_predicates if r.inverse_of and r.inverse_of not in names}
    if unknown:
        raise FunctionCallException(
            f"Partition {partition}: inverse_of must name a structure relation of the partition; unknown: {sorted(unknown)}"
        )


def mark_llm_relations(profile: KgPartitionProfile) -> None:
    """Structure relations that differ from the default relation of the same name are the LLM's."""
    defaults = {r.name: r.model_dump() for r in DEFAULT_STRUCTURE_RELATIONS}
    for relation in profile.structure_predicates:
        if relation.model_dump() != defaults.get(relation.name):
            relation.source = "llm"


@func_reg.register(ProfileFunctionName.FINISH)
def finish(state: ProfileTaskState, fn_args: dict) -> str:
    patches = fn_args.get("partitions") or {}
    unknown = set(patches) - set(state.default.partitions)
    if unknown:
        raise FunctionCallException(
            f"Unknown partitions {sorted(unknown)}; partitions: {', '.join(state.default.partitions)}"
        )
    partitions = {}
    for name, default in state.default.partitions.items():
        patch = patches.get(name) or {}
        try:
            profile = KgPartitionProfile.model_validate(default.model_dump() | patch)
        except ValidationError as e:
            raise FunctionCallException(f"Invalid profile for partition {name}: {e}") from e
        check_structure_names(name, profile)
        if "structure_predicates" in patch:
            mark_llm_relations(profile)
        partitions[name] = profile
    state.profile = KgProfile(
        kg_name=state.default.kg_name,
        alignment_namespaces=fn_args.get("entity_space") or state.default.alignment_namespaces,
        partitions=partitions,
    )
    state.rationale = fn_args.get("rationale")
    changed = {name: sorted(patch) for name, patch in patches.items() if patch}
    return "Profile submitted. Changed fields: " + (json.dumps(changed) if changed else "none (default profile)")


def format_stats(name: str, stats: PartitionStats, profile: KgPartitionProfile) -> str:
    share = lambda n: f" ({n / stats.size:.0%})" if stats.size else ""
    lines = [f"Partition {name}: {stats.size} entities"]
    if stats.sample_size:
        agreement = "–" if stats.local_name_agreement is None else f"{stats.local_name_agreement:.0%}"
        speaking = "–" if stats.local_name_speaking is None else f"{stats.local_name_speaking:.0%}"
        lines.append(
            f"  local names (sample of {stats.sample_size}, {stats.labelled} with a name): "
            f"{agreement} agree with a name, {speaking} look like words -> {profile.local_name_type.value}"
        )
    for title, counts in (("canon predicates used", stats.predicate_counts),
                          ("structure relations used", stats.structure_counts)):
        if counts:
            lines.append(f"  {title} (entities): "
                         + ", ".join(f"{k} {n}{share(n)}" for k, n in sorted(counts.items(), key=lambda x: -x[1])))
    return "\n".join(lines)


def format_profile_input(state: ProfileTaskState) -> str:
    manager = state.manager
    canon = ", ".join(f"{role}: {' '.join(ps)}" for role, ps in ROLE_PREDICATES.items())
    structure = ", ".join(r.name for r in DEFAULT_STRUCTURE_RELATIONS)
    default = {name: p.model_dump(mode="json", exclude_defaults=True) for name, p in state.default.partitions.items()}
    return "\n\n".join([
        f"Knowledge graph: {manager.kg}" + (f" ({manager.description})" if manager.description else ""),
        "Entity space (IRI prefixes of the entities to align): " + ", ".join(state.default.alignment_namespaces),
        f"Default canon checked in every partition ({canon}); structure relations: {structure}. "
        "Canon predicates not listed for a partition are not used there.",
        "Statistics:\n" + "\n".join(format_stats(n, state.stats[n], p) for n, p in state.default.partitions.items()),
        "Default profile (fields with their default value omitted):\n" + json.dumps(default, indent=1),
    ])


class OMProfileTask(GraspTask):
    """Profile stage: checks and completes the default profile of one KG.
    Input: a dict with kg (name of a configured knowledge graph) or the name
    itself; may be omitted if exactly one KG is configured."""
    name = "om-profile"
    state: ProfileTaskState

    def setup(self, input: Any) -> str:
        kg = input.get("kg") if isinstance(input, dict) else input
        if kg:
            manager, _ = find_manager(self.managers, kg)
        elif len(self.managers) == 1:
            manager = self.managers[0]
        else:
            raise ValueError("Input must name the kg unless exactly one KG is configured")
        profile, stats = default_profile(manager, self.config)
        self.state = ProfileTaskState(manager=manager, default=profile, stats=stats)
        return format_profile_input(self.state)

    def system_information(self) -> str:
        info = load_prompt("profile_task.txt")
        task_instructions = load_task_instructions(get_om_task_config(self.config))
        if task_instructions:
            info += "\n\n" + format_section("Task-specific instructions", task_instructions)
        return info

    def rules(self) -> list[str]:
        return [
            "Read the default profile and the statistics first, then inspect typical entities "
            "of each partition with the SPARQL functions before you change anything.",
            "Test every predicate or pattern you add or change with try_pattern.",
            "You can execute multiple tool calls in one step. Prefer several tool calls and "
            "SPARQL queries per step over single calls.",
            "Call finish exactly once, at the end.",
            f"You have at most {self.config.max_steps} steps; one step is one response of yours, "
            "with all tool calls in it. If you run out of steps without calling finish, the "
            "default profile is used.",
        ]

    def function_definitions(self) -> list[dict]:
        return profile_functions()

    def final_step_functions(self) -> list[dict] | None:
        return [f for f in profile_functions() if f["name"] == ProfileFunctionName.FINISH]

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        fn: Callable | None = func_reg.Registry.get(fn_name)
        if fn is None:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return fn(self.state, fn_args)

    def done(self, fn_name: str) -> bool:
        return fn_name == ProfileFunctionName.FINISH and self.state.profile is not None

    def output(self, messages: list[Message]) -> dict:
        state = self.state
        finished = state.profile is not None
        return {
            "kg": state.default.kg_name,
            "finished": finished,
            "fallback": None if finished else "default",
            "rationale": state.rationale,
            "profile": (state.profile or state.default).model_dump(mode="json"),
            "default_profile": state.default.model_dump(mode="json"),
            "stats": {name: s.model_dump(mode="json") for name, s in state.stats.items()},
        }
