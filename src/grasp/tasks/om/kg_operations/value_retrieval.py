"""
Values of the entities of a grouping (names, synonyms, ... or own patterns) for comparison.
"""
from typing import Iterable, Iterator

from grasp.manager import KgManager
from grasp.sparql.types import Binding
from grasp.tasks.om.data_models.patterns import PredicatePattern
from grasp.tasks.om.data_models.profile import Roles
from grasp.tasks.om.data_models.scope import ValuePatternSource
from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.types import IRI, BindingType
from grasp.tasks.om.kg_operations.grouping import Grouping, grouping_to_sparql_query
from grasp.tasks.om.sparql_access.execution import stream_rows


def retrieve_value_rows(
        manager: KgManager,
        grouping: Grouping,
        entity_space: Iterable[str],
        value_source: ValuePatternSource,
        roles: Roles
        ) -> Iterator[ValueRow] | None:

    branches = value_branches(value_source, roles)

    if not branches:
        return None

    query_sparql = grouping_to_sparql_query(
        manager, grouping, entity_space,
        "?entity ?group ?o ?role",
        branches=branches
        )
    rows = stream_rows(manager, query_sparql)
    for row in rows:
        entity = IRI(Binding.from_dict(row["entity"]).identifier())
        group = Binding.from_dict(row["group"]).value
        role = Binding.from_dict(row["role"]).value
        value_binding = Binding.from_dict(row["o"])
        if value_binding.typ == BindingType.BNODE:
            continue
        value = value_binding.value
        language = value_binding.lang

        yield ValueRow(
            entity=entity,
            group_name=group,
            role_name=role,
            raw_projection_value=value,
            value_language=language
            )


def value_branches(
        source: ValuePatternSource,
        roles: Roles,
        new_role_name: str = "own"
        ) -> list[str]:
    """SPARQL alternatives over ?s (entity) and ?o (value), each binding ?role."""
    if isinstance(source, PredicatePattern):
        return [
            f'{branch} BIND("{new_role_name}" AS ?role)'
            for branch in source.alternatives()
            ]
    role_names = source
    branches = []
    for role_name in role_names:
        pattern: PredicatePattern = getattr(roles, role_name)
        if pattern is not None:
            branches += [
                f'{branch} BIND("{role_name}" AS ?role)'
                for branch in pattern.alternatives()
                ]
    return branches
