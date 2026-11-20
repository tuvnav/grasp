"""
Groupings: named membership patterns over ?s, as exclusive partitions or
overlapping groups, and the queries over their entities (?s ?group) that
every statistic and value retrieval builds on. In a large KG the members are
the sampled entities (large_kg_sampling).
"""
from dataclasses import dataclass
from typing import Iterable, Sequence

from grasp.manager import KgManager
from grasp.sparql.types import Binding
from grasp.tasks.om.data_models.profile import (
    GroupDefinition,
    GroupOrigin,
    GroupProfile,
    KgProfile,
    Partition,
    WithinKgNonCompetingPartitionPair,
    short_iri,
)
from grasp.tasks.om.foundation.rdf_vocabulary import (
    KIND_ALTERNATIVES,
    UNCATEGORIZED,
    UNIVERSE,
    VOCABULARY_PROLOGUE,
)
from grasp.tasks.om.kg_operations.large_kg_sampling import class_counts, current_sample, default_class_groups, type_membership
from grasp.tasks.om.sparql_access import (
    select_rows,
    sparql_literal,
    make_sparql_prologue,
    sparql_filter_by_space,
    split_prologue,
    union_of,
    down_pushed_union,
    as_batched_values_blocks,
    values_table,
)
from grasp.tasks.om.foundation.types import IRI

# kinds of the default partitions (stage b1), in order: the first match wins
KINDS = tuple(KIND_ALTERNATIVES)


# kind alternatives left out in a large KG (their negations are evaluated on the whole KG)
LARGE_KG_SKIPPED_ALTERNATIVES = {"instance": 2}


@dataclass(frozen=True)
class Grouping:
    """Named membership patterns over ?s, in order. Exclusive (partitions):
    an entity belongs to the first member that matches. Overlapping (groups): to
    every one that matches. A name may occur more than once (one set of
    entities defined by several patterns). Entities matching a membership of
    minus belong to no member (e.g. the entities of excluded groups)."""
    members: tuple[tuple[str, str], ...]  # ((group_name, sparql_pattern), ...)
    exclusive: bool = True
    minus: tuple[str, ...] = ()  # sparql_patterns

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(name for name, _ in self.members))

    def merged(self, name: str) -> "Grouping":
        """All members as one: the entities of any member."""
        return Grouping(tuple((name, membership) for _, membership in self.members if membership != UNIVERSE),
                        exclusive=True, minus=self.minus)

    def extended_by(
            self,
            additional_members: Sequence[tuple[str, str]] = (),
            additional_minus: Sequence[str] = ()
            ) -> "Grouping":
        new_members = self.members + tuple(additional_members)
        new_minus = self.minus + tuple(additional_minus)
        return Grouping(new_members, self.exclusive, new_minus)


def default_partition_sparql(kind: str, large: bool = False) -> str:
    alternatives = list(KIND_ALTERNATIVES[kind])
    if large and kind in LARGE_KG_SKIPPED_ALTERNATIVES:
        alternatives.pop(LARGE_KG_SKIPPED_ALTERNATIVES[kind])
    return union_of(alternatives)


def get_default_partitions(
        kinds: Iterable[str] = KINDS, large: bool = False
        ) -> list[Partition]:
    return [
        Partition(
            name=kind,
            membership_sparql=default_partition_sparql(kind, large)
            )
        for kind in kinds
        ]


def get_assigned_default_partitions(
        kg_manager: KgManager,
        kg_profile: KgProfile,
        entity_space: list[str]
        ) -> dict[str, Partition]:
    """
    The kind partitions with entities (in a large KG without the alternatives
    that cannot be evaluated there).
    """
    large = kg_profile.large
    partitions = get_default_partitions(large=large)
    sizes = estimated_group_sizes(
        kg_manager,
        partitions_to_grouping(partitions, mark_uncategorized_as_rest=False),
        entity_space
        )
    return {p.name: p for p in partitions if sizes.get(p.name)}


def partitions_to_grouping(
        partitions: list[Partition], mark_uncategorized_as_rest: bool = True
        ) -> Grouping:
    """
    The partitions, exclusive; with rest, the subjects outside all of them as
    UNCATEGORIZED (not in a large KG, see grouping_subquery).
    """
    members = tuple((p.name, p.membership_sparql) for p in partitions)
    return Grouping(
        members + (((
            UNCATEGORIZED, UNIVERSE),) if mark_uncategorized_as_rest else ()
        ),
        exclusive=True
        )


def default_partitions_to_grouping(large: bool = False) -> Grouping:
    return partitions_to_grouping(get_default_partitions(large=large))


def groups_to_grouping(
        groups: Sequence[GroupDefinition | GroupProfile]
        ) -> Grouping:
    return Grouping(
        tuple(
            (g.name, g.membership_sparql)
            if isinstance(g, GroupDefinition)
            else (g.name, g.definition.membership_sparql)
            for g in groups
            ),
        exclusive=False
        )


def single_group_grouping(profile: KgProfile, group: str) -> Grouping:
    """
    The grouping of one group only, without the entities of excluded groups
    (they are not aligned): stage e looks at no other group.
    """
    excluded = tuple(
        profile.get_group(name).definition.membership_sparql
        for name in profile.excluded_group_names() if name != group
        )
    return Grouping(
        ((group, profile.get_group(group).definition.membership_sparql), ),
        exclusive=False, minus=excluded
        )


def grouping_to_sparql_subquery(
        manager: KgManager,
        grouping: Grouping | None,
        space: Iterable[str] = ()
        ) -> tuple[dict[str, str], str]:
    """
    Generates a SPARQL query with prefixes as dict. Returns:
    ({shortened_prefix: <http://full_prefix_iri>}, sparql_subquery_for_grouping).
    binding ?s ?group within the entity space.
    Without a grouping: the kind partitions. In a large KG, the sampled members
    (large_kg_sampling), assigned in Python, as a VALUES table.
    """
    grouping = grouping or default_partitions_to_grouping()
    prefixes, _ = split_prologue(
        manager, VOCABULARY_PROLOGUE + "\nSELECT * WHERE { ?s ?p ?o }"
        )
    sample = current_sample(manager)
    if sample is not None:
        rows = [(f"<{e}>", sparql_literal(g)) for e, g in sample.assignments(grouping)]
        return prefixes, f"{{ {values_table(['s', 'group'], rows)} }}"
    branches = " UNION ".join(
        f"{{ {{ SELECT DISTINCT ?s WHERE {{ {membership} }} }} BIND({rank} AS ?rank) }}"
        for rank, (_, membership) in enumerate(grouping.members)
    )
    minus_part = "".join(f"\nMINUS {{ SELECT DISTINCT ?s WHERE {{ {membership} }} }}" for membership in grouping.minus)
    internal_wheres = f"{branches}\nFILTER(isIRI(?s))\n{sparql_filter_by_space(space)}{minus_part}"
    names_rank_mapping = " ".join(f"({rank} {sparql_literal(name)})" for rank, (name, _) in enumerate(grouping.members))
    if grouping.exclusive:
        return prefixes, (f"{{ SELECT ?s ?group WHERE {{\n"
                          f"{{ SELECT ?s (MIN(?rank) AS ?first) WHERE {{\n{internal_wheres}\n}} GROUP BY ?s }}\n"
                          f"VALUES (?first ?group) {{ {names_rank_mapping} }}\n}} }}")
    return prefixes, (
        f"{{ SELECT DISTINCT ?s ?group WHERE {{\n{internal_wheres}\n"
        f"VALUES (?rank ?group) {{ {names_rank_mapping} }}\n}} }}"
        )


def grouping_to_sparql_query(
    manager: KgManager,
    grouping: Grouping | None,
    space: Iterable[str],
    select: str,
    where: str = "",
    tail: str = "",
    branches: list[str] | None = None,
        ) -> str:
    """
    Query over the entities of a grouping (?s ?group), joined with the
    given patterns; with branches, joined with the alternatives of a pattern
    over ?s (the entity).
    """
    # TODO: down push only cheap parts, not whole subquery for non large KGs.
    # E. g. proceed as in large KGs (reach down only value-tables).
    prefixes, subquery = grouping_to_sparql_subquery(manager, grouping, space)
    body = (
        down_pushed_union(subquery, branches)
        if branches is not None
        else subquery
        )
    return (
        f"{make_sparql_prologue(prefixes)}\nSELECT {select}"
        f" WHERE {{\n{body}\n{where}\n}} {tail}"
    )


def profile_partitions_to_grouping(
        kg_profile: KgProfile
        ) -> Grouping:
    return partitions_to_grouping(
        kg_profile.partitions, mark_uncategorized_as_rest=False
        )


def groups_by_entity(
        manager: KgManager, kg_profile: KgProfile, entities: Iterable[IRI]
        ) -> dict[IRI, frozenset[str]]:
    """
    The names of the groups of each entity (empty: in no group).
    """
    group_names: dict[IRI, set[str]] = {entity: set() for entity in entities}
    branches = [
        f"{group.definition.membership_sparql} BIND({sparql_literal(group.name)} AS ?group)"
        for group in kg_profile.groups
    ]
    if branches:
        for values_block in as_batched_values_blocks(group_names):
            query = (
                "SELECT DISTINCT ?s ?group WHERE {\n"
                + down_pushed_union(f"VALUES ?s {{ {values_block} }}", branches)
                + "\n}"
            )
            for row in select_rows(manager, query):
                group_names[IRI(row["s"].value)].add(row["group"].value)
    return {entity: frozenset(names) for entity, names in group_names.items()}



def ensure_groups(
        kg_manager: KgManager, kg_profile: KgProfile, entity_space: list[str]
        ) -> dict[str, GroupDefinition]:
    """
    The groups of the input profile; otherwise one per partition and, in a
    large KG, one per large class.
    """
    profile = kg_profile
    if profile.groups:
        return {g.name: g.definition for g in profile.groups}
    groups = {name: GroupDefinition(name=p.name, membership_sparql=p.membership_sparql, origin=GroupOrigin.PARTITION,
                                    features=p.features)
              for name, p in ensure_partitions(
                  kg_manager, kg_profile, entity_space
                  ).items()}
    if profile.large:
        for name, type_iri, _ in default_class_groups(
                kg_manager, tuple(entity_space), class_counts(kg_manager)
                ):
            if name not in groups:
                groups[name] = GroupDefinition(
                    name=name,
                    membership_sparql=type_membership(type_iri),
                    origin=GroupOrigin.CLASS,
                    features=f"typed {short_iri(type_iri)}"
                    )
    return groups


def ensure_partitions(
        kg_manager: KgManager, kg_profile: KgProfile, entity_space: list[str]
        ) -> dict[str, Partition]:
    if kg_profile.partitions:
        return {p.name: p for p in kg_profile.partitions}
    return get_assigned_default_partitions(kg_manager, kg_profile, entity_space)


def default_non_competing_pairs(
        partitions: dict[str, Partition]
        ) -> list[WithinKgNonCompetingPartitionPair]:
    """
    Properties do not compete with the other kinds (the default before stage b decides).
    """
    if "property" not in partitions:
        return []
    return [
        WithinKgNonCompetingPartitionPair(
            partition_a="property",
            partition_b=name,
            reason="properties and entities are not equivalent")
        for name in partitions if name != "property"
        ]


def count_rows(
        rows: list[dict[str, Binding]], *keys: str
        ) -> dict[tuple[str, ...], int]:
    return {
        tuple(row[k].value for k in keys): int(row["n"].value)
        for row in rows if all(k in row for k in keys)
        }


def retrieve_counts_by_group(
        manager: KgManager,
        grouping: Grouping | None,
        space: Iterable[str],
        where: str,
        branches: list[str] | None = None
        ) -> dict[str, int]:
    sparql = grouping_to_sparql_query(
        manager,
        grouping,
        space,
        "?group (COUNT(DISTINCT ?s) AS ?n)",
        where,
        "GROUP BY ?group",
        branches
        )
    return {
        group: n for (group,), n in
        count_rows(select_rows(manager, sparql), "group").items()
        }


def retrieve_group_sizes(
        manager: KgManager, grouping: Grouping | None, space: Iterable[str]
        ) -> dict[str, int]:
    return retrieve_counts_by_group(manager, grouping, space, "")


def estimated_group_sizes(
        manager: KgManager, grouping: Grouping, space: Iterable[str]
        ) -> dict[str, int]:
    """
    Entities per member of a grouping: counted in a small KG, estimated from
    the samples of the memberships in a large one (large_kg_sampling).
    """
    sample = current_sample(manager)
    if sample is None:
        return retrieve_group_sizes(manager, grouping, space)
    return sample.estimated_sizes(grouping)


def profile_groups_to_grouping(
        kg_profile: KgProfile, additional_groups: Sequence[GroupDefinition] = ()
        ) -> Grouping:
    return groups_to_grouping(kg_profile.groups).extended_by(
        additional_minus=kg_profile.excluded_group_membership_sparqls()
    ).extended_by(
        additional_members=tuple((g.name, g.membership_sparql) for g in additional_groups)
        )
