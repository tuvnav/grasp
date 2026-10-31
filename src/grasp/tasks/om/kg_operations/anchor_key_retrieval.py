"""
Key rows of one side for the anchor stage: the composed key, scope and partition queries (SideQueries).
"""
from collections import Counter
from dataclasses import replace
from typing import Iterable, Iterator

from grasp.manager import KgManager
from grasp.tasks.om.data_models.anchors import KeyEntityRow, SideQueries
from grasp.tasks.om.foundation.vocabulary import ALL, namespace_of
from grasp.tasks.om.sparql_access.execution import stream_rows
from grasp.tasks.om.sparql_access.query_building import QueryPart, compose

KEY_ROW_PROJECTION = ["?entity", "?key", "?role", "?partition"]


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
