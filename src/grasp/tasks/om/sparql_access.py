"""
Layer 2: generic SPARQL access (execution, VALUES, query building, lookups),
without domain logic.

Variable convention of all queries and patterns: ?s is the entity a query is
about (also where the entity is the object of a triple, e.g. incoming edges
?o ?p ?s), ?o its value, ?p the predicate between them. Any further entity gets
a speaking name (e.g. ?linked, ?usage_subject).
"""

import json
import re
import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from grasp.manager import KgManager
from grasp.sparql.types import AskResult, SelectResult, Binding
from grasp.sparql.utils import prepare_identifier_for_sparql
from grasp.utils import FunctionCallException
from grasp.build.data import stream_json
from grasp.sparql.utils import READ_TIMEOUT, execute

from grasp.tasks.om.foundation.types import IRI, BindingType, BlankNodeTree


# IRIs per VALUES block in batch queries
BATCH_SIZE = 200
# VALUES blocks with more IRIs are put into
# a query after its prefixes are fixed
MAX_INLINE_VALUES = 100
VALUES_PLACEHOLDER_IRI = "urn:grasp-values:"
VALUES_PLACEHOLDER_INDEX: dict[str, str] = {}
PREFIX_RE = re.compile(
    r"^\s*PREFIX\s+([^:\s]*):\s*<([^>]*)>\s*$", re.IGNORECASE
    )
IRI_REFERENCE = re.compile(r"<[^\s<>\"{}|^`\\]*>")
# statistics queries of the profile may take longer than a query of the model
STATS_TIMEOUT = (6, 120)


def split_prologue(
        manager: KgManager, sparql: str
        ) -> tuple[dict[str, str], str]:
    """
    Normalizes the prefixes of a query and splits it into (prefixes, body).
    """
    lines = manager.fix_prefixes(sparql).splitlines()
    prefixes = {}
    while lines and (m := PREFIX_RE.match(lines[0])):
        prefixes[m.group(1)] = m.group(2)
        lines.pop(0)
    return prefixes, "\n".join(lines)


def closing_brace(text: str, start: int) -> int | None:
    """
    Index of the brace closing the one at start; braces in strings and
    IRIs do not count.
    """
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


def extract_union_alternatives(pattern: str) -> list[str]:
    """
    The alternatives of a pattern of the form { A } UNION { B } ...; the
    pattern itself if it is not of this form.
    """
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


def union_of(alternatives: Sequence[str]) -> str:
    """
    Returns a group pattern of alternatives: the pattern itself if only one.
    """
    return (
        alternatives[0] if len(alternatives) == 1
        else " UNION ".join(f"{{ {a} }}" for a in alternatives)
    )


def sparql_filter_by_space(space: Iterable[str]) -> str:
    space = tuple(space)
    if not space:
        return ""
    return "FILTER(" + " || ".join(
        f'STRSTARTS(STR(?s), "{name_space}")'
        for name_space in space
        ) + ")"


def make_sparql_prologue(prefixes: dict[str, str]) -> str:
    return "\n".join(
        f"PREFIX {short}: <{long}>" for short, long in prefixes.items()
        )


def sparql_literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def down_pushed_union(
        search_space_restriction: str,
        branch_patterns: list[str],
        ) -> str:
    """
    Builds a SPARQL query with the space restriction down pushed
    into the branch patterns:
    '{<restriction_condiction> <branch_1>} UNION {<restriction_condiction> <branch_2>}'
    instead of '<restriction_condition> {branch_1} UNION {branch_2}'.
    A Qlever server evaluates an alternative in braces on its own,
    so it would scan the whole KG for it if the group were outside.
    The restriction binds ?s (the entity) before each branch, so the server
    looks the branch up from the entities (index scan).
    """
    if len(branch_patterns) == 1:
        return f"{search_space_restriction}\n{branch_patterns[0]}"
    return " UNION ".join(
        f"{{ {search_space_restriction}\n{part} }}" for part in branch_patterns
        )


def values_clause(var: str, iris: Iterable[str]) -> str:
    """
    VALUES ?var { <iri> ... }; a large one as a placeholder IRI that
    prepare_query replaces by the IRIs.
    """
    terms = " ".join(f"<{iri}>" for iri in iris)
    if terms.count("<") <= MAX_INLINE_VALUES:
        return f"VALUES ?{var} {{ {terms} }}"
    return f"VALUES ?{var} {{ {replace_values_by_placeholder(terms)} }}"


def values_table(variables: list[str], rows: Iterable[tuple[str, ...]]) -> str:
    """
    VALUES (?a ?b) { (term term) ... } of SPARQL terms; a large one as a
    placeholder row that prepare_query replaces by the rows.
    """
    head = "(" + " ".join(f"?{v}" for v in variables) + ")"
    rows = list(rows)
    terms = " ".join("(" + " ".join(row) + ")" for row in rows)
    if len(rows) <= MAX_INLINE_VALUES:
        return f"VALUES {head} {{ {terms} }}"
    undef = " UNDEF" * (len(variables) - 1)
    return f"VALUES {head} {{ ({replace_values_by_placeholder(terms)}{undef}) }}"


def replace_values_by_placeholder(terms: str) -> str:
    key = f"{VALUES_PLACEHOLDER_IRI}{hashlib.sha1(terms.encode()).hexdigest()}"
    VALUES_PLACEHOLDER_INDEX[key] = terms
    return f"<{key}>"


def expand_values_blocks(sparql: str) -> str:
    if VALUES_PLACEHOLDER_IRI not in sparql:
        return sparql
    # a placeholder row of a table (<key> UNDEF ...), or a placeholder IRI
    return re.sub(
        rf"\(\s*<({re.escape(VALUES_PLACEHOLDER_IRI)}[0-9a-f]+)>(?:\s+UNDEF)+\s*\)|<({re.escape(VALUES_PLACEHOLDER_IRI)}[0-9a-f]+)>",
        lambda m: VALUES_PLACEHOLDER_INDEX[m.group(1) or m.group(2)], sparql)


def as_batched_values_blocks(iris: Iterable[IRI]) -> Iterable[str]:
    """
    The IRIs as VALUES contents (<a> <b> ...) in blocks of BATCH_SIZE.
    """
    iris = sorted(set(iris))
    for start in range(0, len(iris), BATCH_SIZE):
        yield " ".join(f"<{iri}>" for iri in iris[start:start + BATCH_SIZE])


def prepare_query(manager: KgManager, sparql: str) -> str:
    """
    The query with fixed prefixes. Large VALUES blocks 
    (sparql_access/values.py) are put in only afterwards: 
    the prefix fixing parses the query, and a parse of
    thousands of VALUES entries exceeds Python's recursion limit.
    """
    return expand_values_blocks(manager.fix_prefixes(sparql))


def stream_rows(manager: KgManager, sparql: str) -> Iterator[dict]:
    """
    Layed out to handle big KGs later without the need to fundamentally
    refactor the whole code.
    """
    try:
        yield from stream_json(
            manager.endpoint, prepare_query(manager, sparql),
            manager.params, manager.headers, manager.auth
        )
    except ValueError as error:
        raise ValueError(query_failure_message(error)) from error


def select_rows(manager: KgManager, sparql: str) -> list[dict[str, Binding]]:
    """
    Rows of a SELECT query as variable -> binding (unbound variables are missing).
    """
    result = execute(
        prepare_query(manager, sparql),
        manager.endpoint,
        STATS_TIMEOUT,
        READ_TIMEOUT,
        0,
        manager.headers, manager.params, None, manager.auth)
    if not isinstance(result, SelectResult):
        return []
    return [dict(row) for row in result.rows()]


def entity_exists_in_kg(manager: KgManager, iri: str) -> bool:
    # subject and object position cover classes/instances; predicate position
    # is needed too since a property may appear only there (e.g. used in
    # triples without a separate rdf:type owl:ObjectProperty/DatatypeProperty
    # declaration triple of its own)
    wrapped = prepare_identifier_for_sparql(iri, manager.iri_literal_parser)
    sparql = (
        f"ASK {{ {{ {wrapped} ?p ?o }} UNION {{ ?s ?p {wrapped} }} "
        f"UNION {{ ?s {wrapped} ?o }} }}"
    )
    result = manager.execute_sparql(sparql)
    assert isinstance(result, AskResult)
    return result.boolean


def ensure_known(manager: KgManager, iri: str, known: set[str]) -> None:
    """
    Fast path: skip the SPARQL round-trip if the IRI was already surfaced by
    an earlier function call result during this conversation. Otherwise fall
    back to an actual live existence check against the KG, and only report
    the IRI as invalid to the LLM once that check has genuinely failed.
    """
    if iri in known:
        return

    if entity_exists_in_kg(manager, iri):
        known.add(iri)
        return

    raise FunctionCallException(
        f"The entity {iri} does not seem to exist in the knowledge graph. "
        "Double check the IRI."
    )


def retrieve_blank_nodes(
        manager: KgManager, entity: IRI, max_depth: int
        ) -> BlankNodeTree:
    """
    The blank nodes below the entity with their edges, level by level: the query
    for depth d is a chain from the entity whose every link is a blank node.
    A property path would also walk through named entities into their blank nodes.
    """
    tree: BlankNodeTree = {}
    start = prepare_identifier_for_sparql(entity, manager.iri_literal_parser)
    for depth in range(1, max_depth + 1):
        chain = f"{start} ?p0 ?b0 . FILTER(isBlank(?b0)) " + " ".join(
            f"?b{i - 1} ?p{i} ?b{i} . FILTER(isBlank(?b{i}))" for i in range(1, depth)
        )
        last = f"b{depth - 1}"
        rows = select_rows(manager, f"SELECT DISTINCT ?{last} ?p ?o WHERE {{ {chain} ?{last} ?p ?o }}")
        new_edges = False
        for row in rows:
            edge = (IRI(row["p"].value), row["o"])
            edges = tree.setdefault(row[last].value, [])
            if edge not in edges:
                edges.append(edge)
                new_edges = True
        if not new_edges or not any(row["o"].typ == BindingType.BNODE for row in rows):
            break
    return tree


@dataclass(frozen=True)
class ValueGroup:
    total_count: int  # all values in the KG
    # all values, or the first <sample_size> values for hubs (ORDER BY ?o)
    values: list[Binding]


def retrieve_value_groups(
        manager: KgManager,
        entities: Iterable[IRI],
        branches: list[str],
        sample_size: int,
        # entity -> role name or predicate IRI -> group;
        # entities without any value are missing
        ) -> dict[IRI, dict[str, ValueGroup]]:
    """
    The values of the entities, grouped by ?predicate_ref, with the number of all
    values and at most sample_size of them. SPARQL has no LIMIT per group:
    the groups are counted first, small groups are retrieved in batches and
    only the large ones (hubs) one by one.
    branches: SPARQL alternatives over ?s (the entity), ?predicate_ref and ?o (the
    value). A branch binds ?predicate_ref by VALUES, not by BIND: BIND to a variable
    that the restriction VALUES (?s ?predicate_ref) already binds is not allowed.
    """
    if not branches:
        return {}
    # 1. count per (entity, predicate_ref)
    counts: dict[tuple[IRI, Binding], int] = {}
    for values_block in as_batched_values_blocks(entities):
        query = (
            "SELECT ?s ?predicate_ref (COUNT(DISTINCT ?o) AS ?count) WHERE {\n"
            + down_pushed_union(f"VALUES ?s {{ {values_block} }}", branches)
            + "\n} GROUP BY ?s ?predicate_ref"
        )
        for row in select_rows(manager, query):
            counts[(IRI(row["s"].value), row["predicate_ref"])] = int(row["count"].value)
    # 2. small groups in batches, 3. large groups one by one, with a sample
    small_value_groups = [
        entity_predicate_pair for entity_predicate_pair, count in counts.items()
        if count <= sample_size
        ]
    queries = [
        # ((entity, predicate_binding), <modifier>): no modifiers here
        (small_value_groups[start:start + BATCH_SIZE], ""
         ) for start in range(0, len(small_value_groups), BATCH_SIZE)
        ]
    queries += [
        # ((entity, predicate_binding), <modifier>)
        ([entity_predicate_pair], f"ORDER BY ?o LIMIT {sample_size}")
        for entity_predicate_pair, count in counts.items() if count > sample_size
    ]
    values_by_entity_predicate_pair: dict[tuple[IRI, Binding], list[Binding]] = {
        pair: [] for pair in counts
        }
    for pairs, modifiers in queries:
        restriction = values_table(
            ["s", "predicate_ref"],
            [
                (
                    f"<{entity}>", f"<{predicate_ref.value}>"
                    if predicate_ref.typ == BindingType.URI
                    else sparql_literal(predicate_ref.value)
                    )  # profile role name
                for entity, predicate_ref in pairs
            ],
        )
        query = (
            "SELECT DISTINCT ?s ?predicate_ref ?o WHERE {\n"
            + down_pushed_union(restriction, branches)
            + "\n} " + modifiers
        )
        for row in select_rows(manager, query):
            values_by_entity_predicate_pair[
                (IRI(row["s"].value), row["predicate_ref"])
                ].append(row["o"])
    groups: dict[IRI, dict[str, ValueGroup]] = {}
    for (entity, predicate_ref), values in values_by_entity_predicate_pair.items():
        groups.setdefault(entity, {})[predicate_ref.value] = ValueGroup(
            counts[(entity, predicate_ref)], values
            )
    return groups


def query_failure_message(
        error: Exception, llm_written_patterns: Sequence[str] = ()
        ) -> str:
    """
    The reason of a failed query for the LLM: QLever's "exception" text if the
    error carries its JSON body, else the first line; without positions, which
    refer to the whole built query.
    """
    error_text = str(error)
    exception = re.search(r'"exception":\s*"((?:[^"\\]|\\.)*)"', error_text)
    reason = json.loads(f'"{exception.group(1)}"') if exception else error_text.splitlines()[0]
    reason = re.sub(r"\s*at position \d+", "", reason)
    message = f"The SPARQL query failed: {reason}"
    if llm_written_patterns:
        message += (
            " Your pattern (built into a larger query): "
            + " | ".join(llm_written_patterns)
            )
    return message
