"""
Running SPARQL queries against a KG: rows of a SELECT, streamed rows, query preparation.
"""
from typing import Iterator

from grasp.build.data import stream_json
from grasp.manager import KgManager
from grasp.sparql.types import Binding, SelectResult
from grasp.sparql.utils import READ_TIMEOUT, execute
from grasp.tasks.om.foundation.types import IRI, BindingType
from grasp.tasks.om.sparql_access.values import expand_values_blocks

# statistics queries of the profile may take longer than a query of the model
STATS_TIMEOUT = (6, 120)


def prepare_query(manager: KgManager, sparql: str) -> str:
    """The query with fixed prefixes. Large VALUES blocks (sparql_access/values.py) are put in
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
