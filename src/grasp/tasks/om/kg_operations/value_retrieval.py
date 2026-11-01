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
from grasp.tasks.om.kg_operations.grouping import Grouping, grouping_query
from grasp.tasks.om.sparql_access.execution import stream_rows


def retrieve_value_rows(
        manager: KgManager,
        grouping: Grouping,
        entity_space: Iterable[str],
        value_source: ValuePatternSource,
        roles: Roles
        ) -> Iterator[ValueRow] | None:
    parts = value_parts(value_source, roles)

    if not parts:
        return None

    query_sparql = grouping_query(
        manager, grouping, entity_space,
        "?entity ?group ?o ?role", parts=parts
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
            group=group,
            role=role,
            value=value,
            language=language
            )


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
