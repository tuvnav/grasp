"""
Groupings: named membership patterns over ?entity, as exclusive partitions or
overlapping groups, and the queries over their entities (?entity ?group) that
every statistic and value retrieval builds on. In a large KG the members are
the sampled entities (large_kg_sampling).
"""
from dataclasses import dataclass
from typing import Iterable, Sequence

from grasp.manager import KgManager
from grasp.tasks.om.data_models.profile import (
    GroupDefinition,
    GroupProfile,
    KgProfile,
    Partition,
)
from grasp.tasks.om.foundation.vocabulary import (
    KIND_ALTERNATIVES,
    UNCATEGORIZED,
    UNIVERSE,
    VOCABULARY_PROLOGUE,
)
from grasp.tasks.om.kg_operations.large_kg_sampling import current_sample
from grasp.tasks.om.sparql_access.execution import select_rows
from grasp.tasks.om.sparql_access.query_building import (
    literal,
    prologue_of,
    space_filter,
    split_prologue,
    union_of,
    down_pushed_union,
)
from grasp.tasks.om.sparql_access.values import values_table

# kinds of the default partitions (stage b1), in order: the first match wins
KINDS = tuple(KIND_ALTERNATIVES)


# kind alternatives left out in a large KG (their negations are evaluated on the whole KG)
LARGE_KG_SKIPPED_ALTERNATIVES = {"instance": 2}


@dataclass(frozen=True)
class Grouping:
    """Named membership patterns over ?entity, in order. Exclusive (partitions):
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


def kind_membership(kind: str, large: bool = False) -> str:
    alternatives = list(KIND_ALTERNATIVES[kind])
    if large and kind in LARGE_KG_SKIPPED_ALTERNATIVES:
        alternatives.pop(LARGE_KG_SKIPPED_ALTERNATIVES[kind])
    return union_of(alternatives)


def kind_partitions(kinds: Iterable[str] = KINDS, large: bool = False) -> list[Partition]:
    return [Partition(name=kind, membership_sparql=kind_membership(kind, large)) for kind in kinds]


def partitions_to_grouping(partitions: list[Partition], mark_uncategorized_as_rest: bool = True) -> Grouping:
    """The partitions, exclusive; with rest, the subjects outside all of them as
    UNCATEGORIZED (not in a large KG, see grouping_subquery)."""
    members = tuple((p.name, p.membership_sparql) for p in partitions)
    return Grouping(members + (((UNCATEGORIZED, UNIVERSE),) if mark_uncategorized_as_rest else ()), exclusive=True)


def default_kinds_to_grouping(large: bool = False) -> Grouping:
    return partitions_to_grouping(kind_partitions(large=large))


def groups_to_grouping(groups: Sequence[GroupDefinition | GroupProfile]) -> Grouping:
    """The groups, overlapping."""
    return Grouping(
        tuple(
            (g.name, g.membership_sparql)
            if isinstance(g, GroupDefinition)
            else (g.name, g.definition.membership_sparql)
            for g in groups
            ),
        exclusive=False
        )


def grouping_to_sparql_subquery(
        manager: KgManager,
        grouping: Grouping | None,
        space: Iterable[str] = ()
        ) -> tuple[dict[str, str], str]:
    """
    Generates a SPARQL query with prefixes as dict. Returns:
    ({shortened_prefix: <http://full_prefix_iri>}, sparql_subquery_for_grouping).
    binding ?entity ?group within the entity space.
    Without a grouping: the kind partitions. In a large KG, the sampled members
    (large_kg_sampling), assigned in Python, as a VALUES table.
    """
    grouping = grouping or default_kinds_to_grouping()
    prefixes, _ = split_prologue(manager, VOCABULARY_PROLOGUE + "\nSELECT * WHERE { ?s ?p ?o }")
    sample = current_sample(manager)
    if sample is not None:
        rows = [(f"<{e}>", literal(g)) for e, g in sample.assignments(grouping)]
        return prefixes, f"{{ {values_table(['entity', 'group'], rows)} }}"
    branches = " UNION ".join(
        f"{{ {{ SELECT DISTINCT ?entity WHERE {{ {membership} }} }} BIND({rank} AS ?rank) }}"
        for rank, (_, membership) in enumerate(grouping.members)
    )
    minus_part = "".join(f"\nMINUS {{ SELECT DISTINCT ?entity WHERE {{ {membership} }} }}" for membership in grouping.minus)
    internal_wheres = f"{branches}\nFILTER(isIRI(?entity))\n{space_filter(space)}{minus_part}"
    names_rank_mapping = " ".join(f"({rank} {literal(name)})" for rank, (name, _) in enumerate(grouping.members))
    if grouping.exclusive:
        return prefixes, (f"{{ SELECT ?entity ?group WHERE {{\n"
                          f"{{ SELECT ?entity (MIN(?rank) AS ?first) WHERE {{\n{internal_wheres}\n}} GROUP BY ?entity }}\n"
                          f"VALUES (?first ?group) {{ {names_rank_mapping} }}\n}} }}")
    return prefixes, (
        f"{{ SELECT DISTINCT ?entity ?group WHERE {{\n{internal_wheres}\n"
        f"VALUES (?rank ?group) {{ {names_rank_mapping} }}\n}} }}"
        )


def estimated_group_sizes(manager: KgManager, grouping: Grouping, space: Iterable[str]) -> dict[str, int]:
    """Entities per member of a grouping: counted in a small KG, estimated from
    the samples of the memberships in a large one (large_kg_sampling)."""
    sample = current_sample(manager)
    if sample is None:
        return group_sizes(manager, grouping, space)
    return sample.estimated_sizes(grouping)


def grouping_to_sparql_query(
    manager: KgManager,
    grouping: Grouping | None,
    space: Iterable[str],
    select: str,
    where: str = "",
    tail: str = "",
    branches: list[str] | None = None,
    var: str = "s"
        ) -> str:
    """
    Query over the entities of a grouping (?entity ?group), joined with the
    given patterns; with parts, joined with the alternatives of a pattern whose
    ?var is the entity (union_with).
    """
    # TODO: down push only cheap parts, not whole subquery for non large KGs.
    # E. g. proceed as in large KGs (reach down only value-tables).
    prefixes, subquery = grouping_to_sparql_subquery(manager, grouping, space)
    body = (
        down_pushed_union(subquery, branches, var)
        if branches is not None
        else subquery
        )
    return (
        f"{prologue_of(prefixes)}\nSELECT {select}"
        f" WHERE {{\n{body}\n{where}\n}} {tail}"
    )


def count_rows(rows: list[dict], *keys: str) -> dict[tuple[str, ...], int]:
    return {tuple(row[k].value for k in keys): int(row["n"].value) for row in rows if all(k in row for k in keys)}


def counts_by_group(manager: KgManager, grouping: Grouping | None, space: Iterable[str],
                    where: str, parts: list[str] | None = None, var: str = "s") -> dict[str, int]:
    sparql = grouping_to_sparql_query(manager, grouping, space, "?group (COUNT(DISTINCT ?entity) AS ?n)",
                                      where, "GROUP BY ?group", parts, var)
    return {group: n for (group,), n in count_rows(select_rows(manager, sparql), "group").items()}


def group_sizes(
        manager: KgManager, grouping: Grouping | None, space: Iterable[str]
        ) -> dict[str, int]:
    return counts_by_group(manager, grouping, space, "")


def profile_groups_to_grouping(
        kg_profile: KgProfile, additional_groups: Sequence[GroupDefinition] = ()
        ) -> Grouping:
    return groups_to_grouping(kg_profile.groups).extended_by(
        additional_minus=kg_profile.excluded_group_membership_sparqls()
    ).extended_by(
        additional_members=tuple((g.name, g.membership_sparql) for g in additional_groups)
        )


def profile_partitions_to_grouping(
        kg_profile: KgProfile
        ) -> Grouping:
    return partitions_to_grouping(
        kg_profile.partitions, mark_uncategorized_as_rest=False
        )
