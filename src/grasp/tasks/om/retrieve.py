from collections import Counter
from dataclasses import replace
import hashlib
import re
from typing import Iterable, Iterator
from urllib.parse import urlsplit

from grasp.build.data import stream_json
from grasp.manager import KgManager
from grasp.sparql.types import Binding, SelectResult
from grasp.sparql.utils import READ_TIMEOUT, execute, prepare_identifier_for_sparql
from grasp.tasks.om.model import (
    IRI, ALL, BindingType, BlankNodeTree, KeyEntityRow, QueryPart, SideQueries, SparqlTerm,
    compose, namespace_of,
)

# IRIs per VALUES block in batch queries
BATCH_SIZE = 200

KEY_ROW_PROJECTION = ["?entity", "?key", "?role", "?partition"]


# statistics queries of the profile may take longer than a query of the model
STATS_TIMEOUT = (6, 120)
# VALUES blocks with more IRIs are put into a query after its prefixes are fixed
MAX_INLINE_VALUES = 100
_PLACEHOLDER = "urn:grasp-values:"
_VALUES_BLOCKS: dict[str, str] = {}


def values_clause(var: str, iris: Iterable[str]) -> str:
    """VALUES ?var { <iri> ... }; a large one as a placeholder IRI that
    prepare_query replaces by the IRIs."""
    terms = " ".join(f"<{iri}>" for iri in iris)
    if terms.count("<") <= MAX_INLINE_VALUES:
        return f"VALUES ?{var} {{ {terms} }}"
    return f"VALUES ?{var} {{ {placeholder(terms)} }}"


def values_table(variables: list[str], rows: Iterable[tuple[str, ...]]) -> str:
    """VALUES (?a ?b) { (term term) ... } of SPARQL terms; a large one as a
    placeholder row that prepare_query replaces by the rows."""
    head = "(" + " ".join(f"?{v}" for v in variables) + ")"
    rows = list(rows)
    terms = " ".join("(" + " ".join(row) + ")" for row in rows)
    if len(rows) <= MAX_INLINE_VALUES:
        return f"VALUES {head} {{ {terms} }}"
    undef = " UNDEF" * (len(variables) - 1)
    return f"VALUES {head} {{ ({placeholder(terms)}{undef}) }}"


def placeholder(terms: str) -> str:
    key = f"{_PLACEHOLDER}{hashlib.sha1(terms.encode()).hexdigest()}"
    _VALUES_BLOCKS[key] = terms
    return f"<{key}>"


def expand_values_blocks(sparql: str) -> str:
    if _PLACEHOLDER not in sparql:
        return sparql
    # a placeholder row of a table (<key> UNDEF ...), or a placeholder IRI
    return re.sub(rf"\(\s*<({re.escape(_PLACEHOLDER)}[0-9a-f]+)>(?:\s+UNDEF)+\s*\)|<({re.escape(_PLACEHOLDER)}[0-9a-f]+)>",
                  lambda m: _VALUES_BLOCKS[m.group(1) or m.group(2)], sparql)


def prepare_query(manager: KgManager, sparql: str) -> str:
    """The query with fixed prefixes. Large VALUES blocks (kg_sample) are put in
    only afterwards: the prefix fixing parses the query, and a parse of
    thousands of VALUES entries exceeds Python's recursion limit."""
    return expand_values_blocks(manager.fix_prefixes(sparql))


def stream_rows(manager: KgManager, sparql: str) -> Iterator[dict]:
    """Layed out to handle big KGs later without the need to fundamentally
    refactor the whole code."""
    return stream_json(
        manager.endpoint, prepare_query(manager, sparql),
        manager.params, manager.headers, manager.auth
        )


def retrieve_entities_by_key(
        manager: KgManager, sparql: str
        ) -> Iterator[KeyEntityRow]:
    rows = stream_rows(manager, sparql)
    for row in rows:
        yield KeyEntityRow.from_raw_row(row, manager.prefixes)


def compose_query(manager: KgManager, queries: SideQueries) -> str:
    """Key query joined with the optional scope query and left-joined with the
    optional partition query."""
    parts: list[str | QueryPart] = [queries.key]
    if queries.scope is not None:
        parts.append(queries.scope)
    if queries.partition is not None:
        parts.append(QueryPart(queries.partition, optional=True))
    return compose(manager, parts, KEY_ROW_PROJECTION)


def retrieve_key_rows(
        manager: KgManager,
        queries: SideQueries,
        entity_space: tuple[str, ...] | None = None,
        ) -> Iterator[KeyEntityRow]:
    """Key rows of one side; with an entity space (IRI prefixes) only rows of
    entities within it, whatever the queries select."""
    rows = retrieve_entities_by_key(manager, compose_query(manager, queries))
    if entity_space is not None:
        rows = (row for row in rows if row.entity.startswith(entity_space))
    if queries.partition is None:
        # no partitioning: one partition with all entities instead of UNCATEGORIZED
        rows = (replace(row, partition=ALL) for row in rows)
    return rows


def namespace_counts(rows: Iterable[KeyEntityRow]) -> Counter[str]:
    """Number of distinct entities per namespace."""
    return Counter(namespace_of(e) for e in {row.entity for row in rows})


def determine_entity_space(counts: Counter[str], configured: list[str] | None) -> tuple[str, ...]:
    """Configured IRI prefixes, otherwise the namespace with the most entities."""
    if configured:
        return tuple(configured)
    if not counts:
        return ()
    return (counts.most_common(1)[0][0],)


def to_sparql_term(manager: KgManager, iri: IRI) -> SparqlTerm:
    """IRI prepared for insertion into SPARQL text (e.g. <...>, escaped)."""
    return SparqlTerm(prepare_identifier_for_sparql(iri, manager.iri_literal_parser))


def retrieve_outgoing_edges(manager: KgManager, entity: IRI) -> list[tuple[Binding, Binding]]:
    """(predicate, object) of all triples with the entity as subject."""
    result = manager.execute_sparql(
        f"SELECT ?p ?o WHERE {{ {to_sparql_term(manager, entity)} ?p ?o }}",
        force_select_result=True,
    )
    if not isinstance(result, SelectResult):
        return []
    # bindings() drops unbound variables; for ?p ?o of a triple pattern both are always bound
    return [row for row in result.bindings() if len(row) == 2]  # type: ignore[misc]


def select_rows(manager: KgManager, sparql: str) -> list[dict[str, Binding]]:
    """Rows of a SELECT query as variable -> binding (unbound variables are missing)."""
    result = execute(prepare_query(manager, sparql), manager.endpoint, STATS_TIMEOUT, READ_TIMEOUT, 0,
                     manager.headers, manager.params, None, manager.auth)
    if not isinstance(result, SelectResult):
        return []
    return [dict(row) for row in result.rows()]


def select_object_iris(manager: KgManager, sparql: str) -> set[IRI]:
    """IRIs bound to ?o in the result of a SELECT query."""
    return {
        IRI(row["o"].value) for row in select_rows(manager, sparql)
        if "o" in row and row["o"].typ == BindingType.URI
    }


def values_blocks(iris: Iterable[IRI]) -> Iterable[str]:
    """The IRIs as VALUES contents (<a> <b> ...) in blocks of BATCH_SIZE."""
    iris = sorted(set(iris))
    for start in range(0, len(iris), BATCH_SIZE):
        yield " ".join(f"<{iri}>" for iri in iris[start:start + BATCH_SIZE])


def retrieve_blank_nodes(manager: KgManager, entity: IRI, max_depth: int) -> BlankNodeTree:
    """
    The blank nodes below the entity with their edges, level by level: the query
    for depth d is a chain from the entity whose every link is a blank node.
    A property path would also walk through named entities into their blank nodes.
    """
    tree: BlankNodeTree = {}
    start = to_sparql_term(manager, entity)
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
