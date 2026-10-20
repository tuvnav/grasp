from dataclasses import replace
from typing import Iterator

from grasp.build.data import stream_json
from grasp.manager import KgManager
from grasp.sparql.types import Binding
from grasp.tasks.om.model import ALL, KeyEntityRow, QueryPart, SideQueries, compose

KEY_ROW_PROJECTION = ["?entity", "?key", "?role", "?partition"]


def stream_rows(manager: KgManager, sparql: str) -> Iterator[dict]:
    """Layed out to handle big KGs later without the need to fundamentally
    refactor the whole code."""
    return stream_json(
        manager.endpoint, manager.fix_prefixes(sparql),
        manager.params, manager.headers, manager.auth
        )


def retrieve_entities_by_key(
        manager: KgManager, sparql: str
        ) -> Iterator[KeyEntityRow]:
    rows = stream_rows(manager, sparql)
    for row in rows:
        yield KeyEntityRow.from_raw_row(row, manager.prefixes)


def compose_side(manager: KgManager, queries: SideQueries) -> str:
    """Key query joined with the optional scope query and left-joined with the
    optional partition query."""
    parts: list[str | QueryPart] = [queries.key]
    if queries.scope is not None:
        parts.append(queries.scope)
    if queries.partition is not None:
        parts.append(QueryPart(queries.partition, optional=True))
    return compose(manager, parts, KEY_ROW_PROJECTION)


def retrieve_key_rows(manager: KgManager, queries: SideQueries) -> Iterator[KeyEntityRow]:
    rows = retrieve_entities_by_key(manager, compose_side(manager, queries))
    if queries.partition is None:
        # no partitioning: one partition with all entities instead of UNCATEGORIZED
        return (replace(row, partition=ALL) for row in rows)
    return rows
