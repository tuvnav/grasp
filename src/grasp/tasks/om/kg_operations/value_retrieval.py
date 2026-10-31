"""
Values of the entities of a grouping (names, synonyms, ... or own patterns) for comparison.
"""
from typing import Iterable

from grasp.manager import KgManager
from grasp.tasks.om.data_models.patterns import PredicatePattern
from grasp.tasks.om.data_models.profile import Roles
from grasp.tasks.om.data_models.scope import ValuePatternSource
from grasp.tasks.om.kg_operations.grouping import Grouping, grouping_query
from grasp.tasks.om.sparql_access.execution import stream_rows


def retrieve_value_rows(
        manager: KgManager,
        grouping: Grouping,
        namespace: Iterable[str],
        value_source: ValuePatternSource,
        roles: Roles
        ):
    parts = value_parts(value_source, roles)

    query_sparql = grouping_query(
        manager, grouping, namespace,
        "?entity ?group ?o", parts=parts
        )
    rows = stream_rows(manager, query_sparql)


def value_parts(source: ValuePatternSource, roles: Roles) -> list[str]:
    """SPARQL alternatives over ?s (entity) and ?o (value), each binding ?role."""
    if isinstance(source, PredicatePattern):
        return [
            f'{part} BIND("own" AS ?role)'
            for part in source.alternatives()
            ]
    role_names = source
    parts = []
    for role_name in role_names:
        pattern = getattr(roles, role_name)
        if pattern is not None:
            parts += [
                f'{part} BIND("{role_name}" AS ?role)'
                for part in pattern.alternatives()
                ]
    return parts
