from collections import Counter
from dataclasses import replace
from typing import Iterable, Iterator
from urllib.parse import urlsplit

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


def retrieve_key_rows(
        manager: KgManager,
        queries: SideQueries,
        entity_space: tuple[str, ...] | None = None,
        ) -> Iterator[KeyEntityRow]:
    """Key rows of one side; with an entity space (IRI prefixes) only rows of
    entities within it, whatever the queries select."""
    rows = retrieve_entities_by_key(manager, compose_side(manager, queries))
    if entity_space is not None:
        rows = (row for row in rows if row.entity.startswith(entity_space))
    if queries.partition is None:
        # no partitioning: one partition with all entities instead of UNCATEGORIZED
        rows = (replace(row, partition=ALL) for row in rows)
    return rows


def namespace_of(iri: str) -> str:
    """Scheme and host of an IRI (e.g. http://mouse.owl), or the part up to the
    last # or / for IRIs without a host (e.g. URNs). Host rather than namespace,
    because some ontologies use a path per entity."""
    parts = urlsplit(iri)
    if parts.netloc:
        return f"{parts.scheme}://{parts.netloc}"
    cut = max(iri.rfind("#"), iri.rfind("/"))
    return iri[:cut + 1] if cut >= 0 else iri


def namespace_counts(rows: Iterable[KeyEntityRow]) -> Counter[str]:
    """Number of distinct entities per namespace."""
    return Counter(namespace_of(e) for e in {row.entity for row in rows})
