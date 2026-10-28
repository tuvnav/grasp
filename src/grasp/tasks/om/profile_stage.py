"""
Base of the profile stage tasks (profile_tasks.py). Each stage reads the
profile of the stages before it (input: kg, profile), computes its default and
statistics, lets the LLM check and change its part, and outputs the profile
with its part; without a finish call, its default.

Shared here: the input layout (earlier results with the findings as unverified
observations), finish with findings and flags, and for the role stages
try_pattern and the role editing functions.
"""
from abc import abstractmethod
from typing import Any, Callable

from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.kg_profile import Partitions, Sides, element_usage, partition_query
from grasp.tasks.om.model import (
    LABEL_PREDICATES, PredicatePattern, StructureRelation, get_om_task_config, load_prompt,
    load_task_instructions, sparql_path,
)
from grasp.tasks.om.profile_model import (
    ROLE_FIELDS, STAGES, ElementKey, Finding, Flag, KgProfile, Roles, Stage, Usage, check_structure,
    describe_pattern, describe_profile, describe_relation, role_elements, roles_from_elements,
)
from grasp.tasks.om.render import clip
from grasp.tasks.om.retrieve import select_rows
from grasp.utils import FunctionCallException, format_enumerate, format_section
from pydantic import BaseModel, ValidationError

FINISH = "finish"
# findings per finish: only the most useful ones end up in prompts of later stages
MAX_FINDINGS = 5
NUM_EXAMPLES = 8
MAX_VALUE_CHARS = 80


def inline_schema(model: type[BaseModel], drop: set[str] = frozenset()) -> dict:
    """JSON schema of a model with its $defs inlined (not every model API
    resolves $ref), without titles and without the given properties."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            node = {k: resolve(v) for k, v in node.items() if k != "title"}
            if "properties" in node:
                node["properties"] = {k: v for k, v in node["properties"].items() if k not in drop}
                if "required" in node:
                    node["required"] = [r for r in node["required"] if r not in drop]
            return node
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def share(n: int, size: int) -> str:
    return f"{n}/{size} ({n / size:.0%})" if size else str(n)


def run_rows(manager: KgManager, sparql: str) -> list[dict]:
    try:
        return select_rows(manager, sparql)
    except Exception as e:
        raise FunctionCallException(f"The SPARQL query failed: {e}") from e


class ProfileStageTask(GraspTask):
    """One stage of the KG profile. Input: a dict with kg (name of a configured
    KG; may be omitted if exactly one KG is configured), profile (output
    profile of the stage before; omitted for the first stage) and, for the
    partition stage, partition."""
    stage: Stage
    prompt_file: str

    manager: KgManager
    input_profile: KgProfile   # as received
    default: KgProfile         # with the stage's default; output without finish
    profile: KgProfile         # with the stage's result after finish
    finished: bool
    rationale: str | None
    stats: dict[str, Any]

    # --- stage interface ---

    @abstractmethod
    def prepare(self, input: dict) -> str:
        """Computes default and statistics, sets self.default, returns the stage's input text."""

    def stage_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return []

    def finish_properties(self) -> dict[str, dict]:
        return {}

    def finish_required(self) -> list[str]:
        return []

    @abstractmethod
    def apply_finish(self, fn_args: dict) -> tuple[KgProfile, str]:
        """(profile with the stage's result, summary); raises FunctionCallException on invalid arguments."""

    def stage_rules(self) -> list[str]:
        return []

    # --- GraspTask ---

    def setup(self, input: Any) -> str:
        if not isinstance(input, dict):
            input = {"kg": input} if input else {}
        kg = input.get("kg")
        if kg:
            self.manager, _ = find_manager(self.managers, kg)
        elif len(self.managers) == 1:
            self.manager = self.managers[0]
        else:
            raise ValueError("Input must name the kg unless exactly one KG is configured")
        profile = input.get("profile")
        self.input_profile = (KgProfile.model_validate(profile) if profile
                              else KgProfile(kg_name=self.manager.kg))
        if self.input_profile.kg_name != self.manager.kg:
            raise ValueError(f"Profile of {self.input_profile.kg_name} given for {self.manager.kg}")
        self.finished, self.rationale, self.stats = False, None, {}
        body = self.prepare(input)
        self.profile = self.default.model_copy(deep=True)
        head = f"Knowledge graph: {self.manager.kg}" + (
            f" ({self.manager.description})" if self.manager.description else "")
        return "\n\n".join(part for part in (head, self.format_earlier_results(), body) if part)

    def format_earlier_results(self) -> str:
        profile = self.input_profile
        parts = []
        notes = describe_profile(profile, findings=False)
        if notes:
            parts.append(format_section("Results of earlier stages", format_enumerate(notes)))
        if profile.findings:
            findings = [f"[{f.stage.value}{', ' + f.partition if f.partition else ''}] {f.text}"
                        for f in profile.findings]
            parts.append(format_section(
                "Observations of earlier stages",
                "Unverified notes of earlier stages. Rely on them only as far as you can confirm "
                "them; if one contradicts what you see, report it with a flag in finish.\n"
                + format_enumerate(findings),
            ))
        if profile.flags:
            flags = [f"[{f.stage.value} on {f.concerns.value}{', ' + f.partition if f.partition else ''}] {f.text}"
                     for f in profile.flags]
            parts.append(format_section("Flags raised so far", format_enumerate(flags)))
        return "\n\n".join(parts)

    def system_information(self) -> str:
        info = load_prompt("profile_common.txt") + "\n\n" + load_prompt(self.prompt_file)
        task_instructions = load_task_instructions(get_om_task_config(self.config))
        if task_instructions:
            info += "\n\n" + format_section("Task-specific instructions", task_instructions)
        return info

    def rules(self) -> list[str]:
        return self.stage_rules() + [
            "Inspect the knowledge graph with the SPARQL functions before you change anything.",
            "You can execute multiple tool calls in one step. Prefer several tool calls and "
            "SPARQL queries per step over single calls.",
            "Call finish exactly once, at the end.",
            f"You have at most {self.config.max_steps} steps; one step is one response of yours, "
            "with all tool calls in it. If you run out of steps without calling finish, the "
            "default of this stage is used.",
        ]

    def earlier_stages(self) -> list[Stage]:
        return STAGES[:STAGES.index(self.stage)]

    def finish_definition(self) -> dict:
        properties = {
            "rationale": {"type": "string",
                          "description": "What you changed and why, and what remains uncertain."},
            "findings": {
                "type": "array",
                "description": (
                    f"At most {MAX_FINDINGS} short observations for later stages and for matching, the "
                    "most useful ones (one fact each, e.g. where synonyms are stored or which entities "
                    "only annotate others); omit what the profile itself already says."
                ),
                "maxItems": MAX_FINDINGS,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "partition": {"type": "string", "description": "Partition it is about, if one."},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
        if self.earlier_stages():
            properties["flags"] = {
                "type": "array",
                "description": "Objections to results or observations of earlier stages that you found wrong.",
                "items": {
                    "type": "object",
                    "properties": {
                        "concerns": {"type": "string", "enum": [s.value for s in self.earlier_stages()]},
                        "text": {"type": "string"},
                        "partition": {"type": "string"},
                    },
                    "required": ["concerns", "text"],
                    "additionalProperties": False,
                },
            }
        properties |= self.finish_properties()
        return {
            "name": FINISH,
            "description": "Submits the result of this stage.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["rationale"] + self.finish_required(),
                "additionalProperties": False,
            },
        }

    def functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        return self.stage_functions() + [(self.finish_definition(), self.finish)]

    def function_definitions(self) -> list[dict]:
        return [definition for definition, _ in self.functions()]

    def final_step_functions(self) -> list[dict] | None:
        return [self.finish_definition()]

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        handlers = {definition["name"]: handler for definition, handler in self.functions()}
        if fn_name not in handlers:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return handlers[fn_name](fn_args)

    def finish(self, fn_args: dict) -> str:
        if len(fn_args.get("findings") or []) > MAX_FINDINGS:
            raise FunctionCallException(f"At most {MAX_FINDINGS} findings; keep the most useful ones for matching")
        profile, summary = self.apply_finish(fn_args)
        try:
            findings = [Finding(stage=self.stage, **f) for f in fn_args.get("findings") or []]
            flags = [Flag(stage=self.stage, **f) for f in fn_args.get("flags") or []]
        except (ValidationError, TypeError) as e:
            raise FunctionCallException(f"Invalid findings or flags: {e}") from e
        unknown = {x.partition for x in findings + flags if x.partition and x.partition not in profile.partitions}
        if unknown:
            raise FunctionCallException(
                f"Unknown partitions {sorted(unknown)}; partitions: {', '.join(profile.partitions) or 'none'}")
        if any(flag.concerns not in self.earlier_stages() for flag in flags):
            raise FunctionCallException("A flag must concern an earlier stage")
        profile.findings += findings
        profile.flags += flags
        self.profile, self.finished, self.rationale = profile, True, fn_args.get("rationale")
        return "Submitted. " + summary

    def done(self, fn_name: str) -> bool:
        return fn_name == FINISH and self.finished

    def output(self, messages: list[Message]) -> dict:
        return {
            "kg": self.manager.kg,
            "stage": self.stage.value,
            "finished": self.finished,
            "fallback": None if self.finished else "default",
            "rationale": self.rationale,
            "profile": (self.profile if self.finished else self.default).model_dump(mode="json"),
            "default": self.default.model_dump(mode="json"),
            "stats": self.stats,
        }


ROLE_DESCRIPTIONS = {
    "names": "preferred names",
    "synonyms": "alternative names",
    "definitions": "textual definitions or descriptions",
    "codes": "codes of the entity in its own system (notations, IDs)",
    "cross_references": "codes or IRIs of the entity in other systems",
    "noise": "values that say nothing about the meaning of an entity (e.g. formatting, provenance, media files)",
}


class RoleEditingTask(ProfileStageTask):
    """Role stages: try_pattern and editing of the working roles (self.roles),
    with coverage over the partitions in scope."""
    roles: Roles
    space: list[str]
    partitions: Partitions
    sizes: dict[str, int]
    scope: list[str]            # partitions whose coverage is shown
    usage: dict[str, Usage]     # partition -> element -> entities, cached
    sides: dict[str, Sides]     # partition -> relation -> (as ?s, as ?o), cached
    counted: set[ElementKey]    # elements counted, also those used nowhere

    def role_functions(self) -> list[tuple[dict, Callable[[dict], str]]]:
        pattern_schema = inline_schema(PredicatePattern)
        partition = {"type": "string", "enum": self.scope}
        return [
            ({
                "name": "try_pattern",
                "description": (
                    "Tests a predicate pattern: per partition, how many entities are ?s, how many "
                    "distinct ?o and literal values it has, and examples with names. Use it for "
                    "attributes (names, codes: ?o is a literal) and structure relations (?o is an entity)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"predicate_pattern": pattern_schema,
                                   "partition": partition | {"description": "Partition for the examples; default: the best covered one."}},
                    "required": ["predicate_pattern"],
                    "additionalProperties": False,
                },
            }, self.try_pattern),
            ({
                "name": "edit_role",
                "description": (
                    "Adds predicates or patterns to a role, or removes them. Shows the role and the "
                    "coverage of each element afterwards. Roles: "
                    + "; ".join(f"{k}: {v}" for k, v in ROLE_DESCRIPTIONS.items())
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": list(ROLE_FIELDS)},
                        "add": pattern_schema | {"description": "Predicates and patterns to add."},
                        "remove_predicates": {"type": "array", "items": {"type": "string"}},
                        "remove_patterns": {"type": "array", "items": {"type": "string"},
                                            "description": "Patterns to remove, exactly as listed."},
                    },
                    "required": ["role"],
                    "additionalProperties": False,
                },
            }, self.edit_role),
            ({
                "name": "set_structure_relation",
                "description": (
                    "Adds a structure relation or replaces the one of the same name. Shows its coverage."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"relation": inline_schema(StructureRelation, drop={"source"})},
                    "required": ["relation"],
                    "additionalProperties": False,
                },
            }, self.set_structure_relation),
            ({
                "name": "remove_structure_relation",
                "description": "Removes a structure relation by its name.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }, self.remove_structure_relation),
        ]

    # --- coverage ---

    def update_usage(self, roles: Roles) -> None:
        """Usage of the elements of roles not counted yet."""
        known = {key for counts in self.usage.values() for key in counts} | self.counted
        new = {k: v for k, v in role_elements(roles).items() if k not in known}
        if not new:
            return
        sides: dict[str, Sides] = {}
        try:
            usage = element_usage(self.manager, self.partitions, self.space, roles_from_elements(new), sides)
        except Exception as e:
            raise FunctionCallException(f"The SPARQL query failed: {e}") from e
        for partition, counts in usage.items():
            self.usage.setdefault(partition, {}).update(counts)
        for partition, counts in sides.items():
            self.sides.setdefault(partition, {}).update(counts)
        self.counted |= set(new)

    def format_coverage(self, key: ElementKey) -> str:
        counts = [f"{p} {share(self.usage.get(p, {}).get(key, 0), self.sizes.get(p, 0))}"
                  for p in self.scope]
        return "; ".join(counts)

    def format_field(self, field: str, pattern: PredicatePattern | None,
                     origin: Callable[[ElementKey], str] | None = None) -> list[str]:
        if pattern is None:
            return [f"{field}: none"]
        mark = lambda key: f" [{origin(key)}]" if origin else ""
        return [f"{field}:"] + [f"  {key[1]} {key[2]}{mark(key)}: {self.format_coverage(key)}"
                                for key in role_elements(Roles(**{field: pattern}))]

    def format_relation(self, relation: StructureRelation, origin: Callable[[ElementKey], str] | None = None) -> str:
        key = ("structure", "relation", relation.name)
        mark = f" [{origin(key)}]" if origin else ""
        side = lambda i: "; ".join(f"{p} {share(self.sides.get(p, {}).get(key, (0, 0))[i], self.sizes.get(p, 0))}"
                                   for p in self.scope)
        return f"  {describe_relation(relation)}{mark}\n    as ?s: {side(0)}\n    as ?o: {side(1)}"

    def format_roles(self, roles: Roles, origin: Callable[[ElementKey], str] | None = None) -> str:
        """Each element with its coverage per partition in scope."""
        self.update_usage(roles)
        lines = [line for field in ROLE_FIELDS for line in self.format_field(field, getattr(roles, field), origin)]
        lines.append("structure relations:" if roles.structure else "structure relations: none")
        lines += [self.format_relation(relation, origin) for relation in roles.structure]
        return "\n".join(lines)

    # --- functions ---

    def try_pattern(self, fn_args: dict) -> str:
        try:
            pattern = PredicatePattern.model_validate(fn_args.get("predicate_pattern") or {})
        except ValidationError as e:
            raise FunctionCallException(f"Invalid predicate_pattern: {e}") from e
        partition = fn_args.get("partition")
        if partition is not None and partition not in self.scope:
            raise FunctionCallException(f"Unknown partition {partition}; partitions: {', '.join(self.scope)}")
        sparql = pattern.predicates_to_sparql()
        where = f"BIND(?entity AS ?s)\n{sparql}"
        rows = run_rows(self.manager, partition_query(
            self.manager, self.partitions, self.space,
            "?partition (COUNT(DISTINCT ?s) AS ?subjects) (COUNT(DISTINCT ?o) AS ?objects) "
            "(SUM(IF(isLiteral(?o), 1, 0)) AS ?literals) (COUNT(*) AS ?n)",
            where, "GROUP BY ?partition",
        ))
        counts = {row["partition"].value: row for row in rows if "partition" in row}
        value = lambda row, key: int(row[key].value) if key in row else 0
        lines = [f"Pattern: {sparql}"]
        for p in self.scope:
            row = counts.get(p, {})
            lines.append(f"Partition {p}: {share(value(row, 'subjects'), self.sizes.get(p, 0))} entities are ?s; "
                         f"{value(row, 'n')} matches, {value(row, 'objects')} distinct ?o, "
                         f"{value(row, 'literals')} literal matches.")
        if partition is None:
            in_scope = [p for p in self.scope if value(counts.get(p, {}), "subjects")]
            partition = max(in_scope, key=lambda p: value(counts[p], "subjects"), default=None)
        if partition is not None and value(counts.get(partition, {}), "subjects"):
            lines.append(f"Examples in {partition} (?s -> ?o, with names in quotes):")
            lines += self.format_examples(partition, where)
        return "\n".join(lines)

    def format_examples(self, partition: str, where: str) -> list[str]:
        """Example matches with a name of ?s and of ?o (codes alone say little)."""
        names = self.roles.names.predicates if self.roles.names else []
        names_path = sparql_path(list(dict.fromkeys(names + LABEL_PREDICATES)))
        rows = run_rows(self.manager, partition_query(
            self.manager, self.partitions, self.space, "?s ?o ?s_name ?o_name",
            f'FILTER(?partition = "{partition}")\n{where}\n'
            f"OPTIONAL {{ ?s {names_path} ?s_name }}\nOPTIONAL {{ ?o {names_path} ?o_name }}",
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

    def edit_role(self, fn_args: dict) -> str:
        role = fn_args.get("role")
        if role not in ROLE_FIELDS:
            raise FunctionCallException(f"Unknown role {role}; roles: {', '.join(ROLE_FIELDS)}")
        current: PredicatePattern | None = getattr(self.roles, role)
        predicates = list(current.predicates) if current else []
        patterns = list(current.patterns) if current else []
        missing = [p for p in fn_args.get("remove_predicates") or [] if p not in predicates]
        missing += [p for p in fn_args.get("remove_patterns") or [] if p not in patterns]
        if missing:
            raise FunctionCallException(
                f"Not in role {role}: {missing}; it has: {describe_pattern(current) if current else 'nothing'}")
        predicates = [p for p in predicates if p not in (fn_args.get("remove_predicates") or [])]
        patterns = [p for p in patterns if p not in (fn_args.get("remove_patterns") or [])]
        if fn_args.get("add"):
            try:
                add = PredicatePattern.model_validate(fn_args["add"])
            except ValidationError as e:
                raise FunctionCallException(f"Invalid add: {e}") from e
            predicates += [p for p in add.predicates if p not in predicates]
            patterns += [p for p in add.patterns if p not in patterns]
        new = PredicatePattern(predicates=predicates, patterns=patterns) if predicates or patterns else None
        self.update_usage(Roles(**{role: new}))  # may raise on SPARQL errors; then nothing changes
        self.roles = self.roles.model_copy(update={role: new})
        return "Role now (coverage as ?s per partition):\n" + "\n".join(self.format_field(role, new))

    def set_structure_relation(self, fn_args: dict) -> str:
        try:
            relation = StructureRelation.model_validate(fn_args.get("relation") or {})
        except ValidationError as e:
            raise FunctionCallException(f"Invalid relation: {e}") from e
        relation.source = "llm"
        key = ("structure", "relation", relation.name)
        # a changed pattern must be counted again
        for counts in self.usage.values():
            counts.pop(key, None)
        self.counted.discard(key)
        self.update_usage(Roles(structure=[relation]))
        text = self.format_relation(relation)
        structure = [r for r in self.roles.structure if r.name != relation.name]
        replaced = len(structure) < len(self.roles.structure)
        self.roles = self.roles.model_copy(update={"structure": structure + [relation]})
        problems = check_structure(self.roles.structure)
        result = f"Structure relation {relation.name} {'replaced' if replaced else 'added'}:\n{text}"
        if problems:
            result += "\nTo resolve before finish: " + "; ".join(problems)
        return result

    def remove_structure_relation(self, fn_args: dict) -> str:
        name = fn_args.get("name")
        names = [r.name for r in self.roles.structure]
        if name not in names:
            raise FunctionCallException(f"No structure relation {name}; relations: {', '.join(names) or 'none'}")
        self.roles = self.roles.model_copy(update={"structure": [r for r in self.roles.structure if r.name != name]})
        return f"Structure relation {name} removed; relations: {', '.join(r.name for r in self.roles.structure) or 'none'}"

    def checked_roles(self) -> Roles:
        problems = check_structure(self.roles.structure)
        if problems:
            raise FunctionCallException("Structure relations: " + "; ".join(problems))
        return self.roles.model_copy(deep=True)

    def init_roles(self, roles: Roles, usage: dict[str, Usage], sides: dict[str, Sides], counted: Roles) -> None:
        """Working roles with the usage counted for the elements of counted."""
        self.roles = roles.model_copy(deep=True)
        self.usage = {p: dict(counts) for p, counts in usage.items()}
        self.sides = {p: dict(counts) for p, counts in sides.items()}
        self.counted = set(role_elements(counted))

