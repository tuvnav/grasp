"""
Building SPARQL text: prologues, composed queries, property paths, UNION
alternatives, and joining a group pattern with the alternatives of a pattern
(union_with, bind_entity).
"""
import re
from dataclasses import dataclass
from typing import Iterable

from grasp.manager import KgManager
from grasp.manager.utils import merge_prefixes
from grasp.tasks.om.foundation.types import IRI


def sparql_path(predicates: list[IRI]) -> str:
    """Alternative property path over the given predicates, e.g. <a>|<b>."""
    return "|".join(f"<{p}>" for p in predicates)


PREFIX_RE = re.compile(r"^\s*PREFIX\s+([^:\s]*):\s*<([^>]*)>\s*$", re.IGNORECASE)


def split_prologue(manager: KgManager, sparql: str) -> tuple[dict[str, str], str]:
    """Normalizes the prefixes of a query and splits it into (prefixes, body)."""
    lines = manager.fix_prefixes(sparql).splitlines()
    prefixes = {}
    while lines and (m := PREFIX_RE.match(lines[0])):
        prefixes[m.group(1)] = m.group(2)
        lines.pop(0)
    return prefixes, "\n".join(lines)


@dataclass(frozen=True)
class QueryPart:
    """A self-contained SELECT query as part of a composed query. Optional parts
    are left-joined: rows without a match keep their other variables."""
    sparql: str
    optional: bool = False


def compose(manager: KgManager, parts: list[str | QueryPart], projection: list[str]) -> str:
    """Joins self-contained SELECT queries on their shared projected variables. Also moves all PREFIX-declarations of the individual
    queries to a single PREFIX-block at the top of the aggregate query."""
    prefixes, bodies = {}, []
    for part in parts:
        if isinstance(part, str):
            part = QueryPart(part)
        part_prefixes, body = split_prologue(manager, part.sparql)
        prefixes, _, _ = merge_prefixes(prefixes, part_prefixes, do_raise=True)
        bodies.append(("OPTIONAL {\n" if part.optional else "{\n") + body + "\n}")
    prologue = "\n".join(f"PREFIX {s}: <{l}>" for s, l in prefixes.items())
    return (f"{prologue}\nSELECT DISTINCT {' '.join(projection)} WHERE {{\n"
            + "\n".join(bodies) + "\n}")


def union_of(alternatives: list[str] | tuple[str, ...]) -> str:
    """A group pattern of alternatives: the pattern itself if only one."""
    return alternatives[0] if len(alternatives) == 1 else " UNION ".join(f"{{ {a} }}" for a in alternatives)


IRI_REFERENCE = re.compile(r"<[^\s<>\"{}|^`\\]*>")


def closing_brace(text: str, start: int) -> int | None:
    """Index of the brace closing the one at start; braces in strings and
    IRIs do not count."""
    depth, quote, i = 0, None, start
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "<" and (m := IRI_REFERENCE.match(text, i)):
            i = m.end() - 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def union_alternatives(pattern: str) -> list[str]:
    """The alternatives of a pattern of the form { A } UNION { B } ...; the
    pattern itself if it is not of this form."""
    text, parts, i = pattern.strip(), [], 0
    while True:
        if not text.startswith("{", i) or (end := closing_brace(text, i)) is None:
            return [pattern]
        parts.append(text[i + 1:end].strip())
        i = len(text) - len(text[end + 1:].lstrip())
        if i == len(text):
            return parts if len(parts) > 1 else [pattern]
        if not text.startswith("UNION", i):
            return [pattern]
        i = len(text) - len(text[i + len("UNION"):].lstrip())


def space_filter(space: Iterable[str]) -> str:
    space = tuple(space)
    if not space:
        return ""
    return "FILTER(" + " || ".join(f'STRSTARTS(STR(?entity), "{ns}")' for ns in space) + ")"


def prologue_of(prefixes: dict[str, str]) -> str:
    return "\n".join(f"PREFIX {short}: <{long}>" for short, long in prefixes.items())


def literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def down_pushed_union(
        search_space_restriction: str,
        branch_patterns: list[str],
        var_to_replace: str | None = "s"
        ) -> str:
    """Builds a SPARQL query with the space restriction down pushed
    into the branch patterns. ?var is bound to ?entity (bind_entity):
    '{<restriction_condiction> <branch_1>} UNION {<restriction_condiction> <branch_2>}'
    instead of '<restriction_condition> {branch_1} UNION {branch_2}'.
    A Qlever server evaluates an alternative in braces on its own,
    so it would scan the whole KG for it if the group were outside."""
    bound = [bind_entity(part, var_to_replace) if var_to_replace else part for part in branch_patterns]
    if len(bound) == 1:
        return f"{search_space_restriction}\n{bound[0]}"
    return " UNION ".join(f"{{ {search_space_restriction}\n{part} }}" for part in bound)


def bind_entity(pattern: str, var: str = "s") -> str:
    """The pattern with ?var replaced by ?entity, followed by BIND(?entity AS ?var).
    Equivalent to binding ?var to ?entity first, but the server can then look
    up the pattern from the entities (index scan) instead of evaluating it on the
    whole KG and joining afterwards, which does not finish on a large KG."""
    return re.sub(rf"\?{var}\b", "?entity", pattern) + f"\nBIND(?entity AS ?{var})"
