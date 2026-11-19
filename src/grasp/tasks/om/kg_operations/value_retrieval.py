"""
Values of the entities of a grouping (names, synonyms, ... or own patterns) for comparison.
"""
from itertools import batched
from typing import Collection, Iterable, Iterator

from grasp.manager import KgManager
from grasp.sparql.types import Binding
from grasp.tasks.om.data_models.profile import KgProfile, Roles, PredicatePattern, StructureRelation
from grasp.tasks.om.data_models.scope import ValuePatternSource
from grasp.tasks.om.data_models.value_rows import ValueRow
from grasp.tasks.om.foundation.types import IRI, BindingType
from grasp.tasks.om.kg_operations.grouping import Grouping, grouping_to_sparql_query, profile_partitions_to_grouping
from grasp.tasks.om.sparql_access.lookups import (
    stream_rows,
    values_clause,
)


def retrieve_value_rows(
        manager: KgManager,
        grouping: Grouping,
        entity_space: Iterable[str],
        value_source: ValuePatternSource,
        roles: Roles,
        object_iris: Collection[IRI] | None = None
        ) -> Iterator[ValueRow]:

    if object_iris is not None and not object_iris:
        return  # restricted to no values (e.g. no witnesses yet): no query
    branches = prepare_value_branches_for_sparql(value_source, roles, object_iris)

    if not branches:
        return  # none of the roles is set in the profile: no values

    query_sparql = grouping_to_sparql_query(
        manager, grouping, entity_space,
        "?s ?group ?o ?role",
        branches=branches
        )
    rows = stream_rows(manager, query_sparql)
    for row in rows:
        entity = IRI(Binding.from_dict(row["s"]).identifier())
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


def prepare_value_branches_for_sparql(
        value_source: ValuePatternSource,
        roles: Roles,
        restrict_to_object_iris: Collection[IRI] | None = None,
        new_role_name: str = "own"
        ) -> list[str]:
    """
    SPARQL alternatives over ?s (entity) and ?o (value), each binding ?role.
    """
    restriction_to_values = (
        f"{values_clause('o', restrict_to_object_iris)}\n"
        if restrict_to_object_iris is not None else ""
        )
    return [
        f'{restriction_to_values}{branch} BIND("{role_name}" AS ?role)'
        for role_name, pattern in named_value_patterns(
            value_source, roles, new_role_name
            )
        for branch in pattern.branches_for_union()
    ]


def named_value_patterns(
        value_source: ValuePatternSource,
        roles: Roles,
        new_role_name: str = "own"
        ) -> list[tuple[str, PredicatePattern]]:
    """
    (role name of the rows, pattern) of every value source;
    roles that are not set are left out.
    """
    match value_source:
        case PredicatePattern():
            return [(new_role_name, value_source)]
        case [*predicate_pattern_refs]:
            named = []
            for pattern_ref in predicate_pattern_refs:
                if isinstance(pattern_ref, StructureRelation):
                    named.append((pattern_ref.name, pattern_ref.predicate_pattern))
                elif (
                        predicate_pattern := roles.predicate_pattern_by_name(pattern_ref)
                        ) is not None:
                    named.append((pattern_ref, predicate_pattern))
            return named


def retrieve_entities(
        manager: KgManager,
        grouping: Grouping,
        entity_space: Iterable[str],
        ) -> list[IRI]:
    """
    The entities of a grouping, each once, sorted (so that a sample drawn
    from them with a fixed seed is the same in every run).
    """
    query = grouping_to_sparql_query(manager, grouping, entity_space, "DISTINCT ?s")
    return sorted({IRI(Binding.from_dict(row["s"]).identifier()) for row in stream_rows(manager, query)})


def retrieve_linked_entities(
        manager: KgManager,
        grouping: Grouping,
        entity_space: Iterable[str],
        linked_to: Collection[IRI],
        batch_size: int = 1000
        ) -> set[IRI]:
    """
    The entities of a grouping linked by a triple, in either direction, to
    one of the given entities (e.g. the neighbours of matched entities).
    """
    entity_space = list(entity_space)
    linked_entities: set[IRI] = set()
    for batch in batched(sorted(linked_to), batch_size):
        query = grouping_to_sparql_query(
            manager, grouping, entity_space, "DISTINCT ?s",
            where=f"{values_clause('linked', batch)}\n"
                  "{ ?s ?p ?linked } UNION { ?linked ?p ?s }"
            )
        linked_entities.update(
            IRI(Binding.from_dict(row["s"]).identifier()) for row in stream_rows(manager, query)
            )
    return linked_entities


def retrieve_partitions_by_entity(
        manager: KgManager,
        kg_profile: KgProfile,
        entity_space: Iterable[str]
        ) -> dict[IRI, str]:
    partitions_grouping = profile_partitions_to_grouping(kg_profile)
    query = grouping_to_sparql_query(
        manager, partitions_grouping,
        entity_space, "?s ?group"
    )

    rows = stream_rows(manager, query)

    partition_names_by_entity: dict[IRI, str] = {}

    for row in rows:
        entity = IRI(Binding.from_dict(row["s"]).identifier())
        partition = Binding.from_dict(row["group"]).value
        partition_names_by_entity[entity] = partition

    return partition_names_by_entity
