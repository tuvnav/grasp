"""
Groupings: named membership patterns over ?entity, as exclusive partitions or
overlapping groups, and the queries over their entities (?entity ?group) that
every statistic and value retrieval builds on. In a large KG the members are
the sampled entities (large_kg_sampling).
"""
from dataclasses import dataclass
from typing import Iterable

from grasp.manager import KgManager
from grasp.tasks.om.data_models.profile import (
    GroupDefinition,
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
    union_with,
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
    minus: tuple[str, ...] = ()

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(name for name, _ in self.members))

    def merged(self, name: str) -> "Grouping":
        """All members as one: the entities of any member."""
        return Grouping(tuple((name, membership) for _, membership in self.members if membership != UNIVERSE),
                        exclusive=True, minus=self.minus)


def kind_membership(kind: str, large: bool = False) -> str:
    alternatives = list(KIND_ALTERNATIVES[kind])
    if large and kind in LARGE_KG_SKIPPED_ALTERNATIVES:
        alternatives.pop(LARGE_KG_SKIPPED_ALTERNATIVES[kind])
    return union_of(alternatives)


def kind_partitions(kinds: Iterable[str] = KINDS, large: bool = False) -> dict[str, Partition]:
    return {kind: Partition(membership_sparql=kind_membership(kind, large)) for kind in kinds}


def partition_grouping(partitions: dict[str, Partition], rest: bool = True) -> Grouping:
    """The partitions, exclusive; with rest, the subjects outside all of them as
    UNCATEGORIZED (not in a large KG, see grouping_subquery)."""
    members = tuple((name, p.membership_sparql) for name, p in partitions.items())
    return Grouping(members + (((UNCATEGORIZED, UNIVERSE),) if rest else ()), exclusive=True)


def kind_grouping(large: bool = False) -> Grouping:
    return partition_grouping(kind_partitions(large=large))


def group_grouping(groups: dict[str, GroupDefinition]) -> Grouping:
    """The groups, overlapping."""
    return Grouping(tuple((name, g.membership_sparql) for name, g in groups.items()), exclusive=False)


def grouping_subquery(manager: KgManager, grouping: Grouping | None,
                      space: Iterable[str] = ()) -> tuple[dict[str, str], str]:
    """(prefixes, group pattern) binding ?entity ?group within the entity space.
    Without a grouping: the kind partitions. In a large KG, the sampled members
    (large_kg_sampling), assigned in Python, as a VALUES table."""
    grouping = grouping or kind_grouping()
    prefixes, _ = split_prologue(manager, VOCABULARY_PROLOGUE + "\nSELECT * WHERE { ?s ?p ?o }")
    sample = current_sample(manager)
    if sample is not None:
        rows = [(f"<{e}>", literal(g)) for e, g in sample.assignments(grouping)]
        return prefixes, f"{{ {values_table(['entity', 'group'], rows)} }}"
    branches = " UNION ".join(
        f"{{ {{ SELECT DISTINCT ?entity WHERE {{ {membership} }} }} BIND({rank} AS ?rank) }}"
        for rank, (_, membership) in enumerate(grouping.members)
    )
    minus = "".join(f"\nMINUS {{ SELECT DISTINCT ?entity WHERE {{ {membership} }} }}" for membership in grouping.minus)
    where = f"{branches}\nFILTER(isIRI(?entity))\n{space_filter(space)}{minus}"
    names = " ".join(f"({rank} {literal(name)})" for rank, (name, _) in enumerate(grouping.members))
    if grouping.exclusive:
        return prefixes, (f"{{ SELECT ?entity ?group WHERE {{\n"
                          f"{{ SELECT ?entity (MIN(?rank) AS ?first) WHERE {{\n{where}\n}} GROUP BY ?entity }}\n"
                          f"VALUES (?first ?group) {{ {names} }}\n}} }}")
    return prefixes, (f"{{ SELECT DISTINCT ?entity ?group WHERE {{\n{where}\n"
                      f"VALUES (?rank ?group) {{ {names} }}\n}} }}")


def estimated_sizes(manager: KgManager, grouping: Grouping, space: Iterable[str]) -> dict[str, int]:
    """Entities per member of a grouping: counted in a small KG, estimated from
    the samples of the memberships in a large one (large_kg_sampling)."""
    sample = current_sample(manager)
    if sample is None:
        return group_sizes(manager, grouping, space)
    return sample.estimated_sizes(grouping)


def grouping_query(manager: KgManager, grouping: Grouping | None, space: Iterable[str],
                   select: str, where: str = "", tail: str = "",
                   parts: list[str] | None = None, var: str = "s") -> str:
    """Query over the entities of a grouping (?entity ?group), joined with the
    given patterns; with parts, joined with the alternatives of a pattern whose
    ?var is the entity (union_with)."""
    prefixes, subquery = grouping_subquery(manager, grouping, space)
    body = union_with(subquery, parts, var) if parts is not None else subquery
    return f"{prologue_of(prefixes)}\nSELECT {select} WHERE {{\n{body}\n{where}\n}} {tail}"


def count_rows(rows: list[dict], *keys: str) -> dict[tuple[str, ...], int]:
    return {tuple(row[k].value for k in keys): int(row["n"].value) for row in rows if all(k in row for k in keys)}


def counts_by_group(manager: KgManager, grouping: Grouping | None, space: Iterable[str],
                    where: str, parts: list[str] | None = None, var: str = "s") -> dict[str, int]:
    sparql = grouping_query(manager, grouping, space, "?group (COUNT(DISTINCT ?entity) AS ?n)",
                            where, "GROUP BY ?group", parts, var)
    return {group: n for (group,), n in count_rows(select_rows(manager, sparql), "group").items()}


def group_sizes(manager: KgManager, grouping: Grouping | None, space: Iterable[str]) -> dict[str, int]:
    return counts_by_group(manager, grouping, space, "")
