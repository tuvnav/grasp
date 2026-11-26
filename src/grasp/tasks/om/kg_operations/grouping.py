"""
Groupings: named membership patterns over ?s, as exclusive partitions or
overlapping groups, and their entities with the names of their groups
(retrieve_groups_by_entity), which every statistic and value retrieval
builds on. One query per membership: the groups are combined (priority of
partitions, excluded groups, entity space) in Python, not in SPARQL.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Collection, Iterable, Sequence

from grasp.manager import KgManager
from grasp.tasks.om.data_models.candidates import POPULATION_REST, GroupRef, Population
from grasp.tasks.om.data_models.profile import (
    GroupDefinition,
    GroupOrigin,
    GroupProfile,
    KgProfile,
    Partition,
    WithinKgNonCompetingPartitionPair,
)
from grasp.tasks.om.foundation.rdf_vocabulary import (
    KIND_ALTERNATIVES,
    UNCATEGORIZED,
    UNIVERSE,
)
from grasp.tasks.om.sparql_access import (
    retrieve_values,
    select_rows,
    union_of,
    as_batched_values_blocks,
)
from grasp.tasks.om.foundation.types import IRI, Side
from grasp.utils import FunctionCallException

# kinds of the default partitions (stage b1), in order: the first match wins
KINDS = tuple(KIND_ALTERNATIVES)


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
        return Grouping(
            tuple(
                (name, membership) for _, membership in self.members
                if membership != UNIVERSE
                ),
            exclusive=True,
            minus=self.minus
            )

    def extended_by(
            self,
            additional_members: Sequence[tuple[str, str]] = (),
            additional_minus: Sequence[str] = ()
            ) -> "Grouping":
        new_members = self.members + tuple(additional_members)
        new_minus = self.minus + tuple(additional_minus)
        return Grouping(new_members, self.exclusive, new_minus)


def get_default_partitions(
        kinds: Iterable[str] = KINDS
        ) -> list[Partition]:
    return [
        Partition(
            name=kind,
            membership_sparql=union_of(KIND_ALTERNATIVES[kind])
            )
        for kind in kinds
        ]


def get_assigned_default_partitions(
        kg_manager: KgManager,
        entity_space: list[str]
        ) -> dict[str, Partition]:
    """
    The kind partitions with entities
    """
    partitions = get_default_partitions()
    sizes = retrieve_group_sizes(
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
    UNCATEGORIZED.
    """
    members = tuple((p.name, p.membership_sparql) for p in partitions)
    return Grouping(
        members + (((
            UNCATEGORIZED, UNIVERSE),) if mark_uncategorized_as_rest else ()
        ),
        exclusive=True
        )


def default_partitions_to_grouping() -> Grouping:
    return partitions_to_grouping(get_default_partitions())


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


def profile_partitions_to_grouping(
        kg_profile: KgProfile
        ) -> Grouping:
    return partitions_to_grouping(
        kg_profile.partitions, mark_uncategorized_as_rest=False
        )


def get_group_names_by_entity(
        manager: KgManager, kg_profile: KgProfile, entities: Iterable[IRI]
        ) -> dict[IRI, frozenset[str]]:
    """
    The names of the groups of each entity (empty: in no group).
    """
    group_names: dict[IRI, set[str]] = {entity: set() for entity in entities}
    for group in kg_profile.groups:
        for values_block in as_batched_values_blocks(group_names):
            query = (
                f"SELECT DISTINCT ?s WHERE {{ VALUES ?s {{ {values_block} }} "
                f"{group.definition.membership_sparql} }}")
            for row in select_rows(manager, query):
                group_names[IRI(row["s"].value)].add(group.name)
    return {entity: frozenset(names) for entity, names in group_names.items()}


def ensure_groups(
        kg_manager: KgManager, kg_profile: KgProfile, entity_space: list[str]
        ) -> dict[str, GroupDefinition]:
    """
    The groups of the input profile; otherwise one per partition
    """
    profile = kg_profile
    if profile.groups:
        return {g.name: g.definition for g in profile.groups}
    groups = {
        name: GroupDefinition(
            name=p.name,
            membership_sparql=p.membership_sparql,
            origin=GroupOrigin.PARTITION,
            features=p.features
            )
        for name, p in ensure_partitions(
            kg_manager, kg_profile, entity_space
            ).items()}
    return groups


def ensure_partitions(
        kg_manager: KgManager, kg_profile: KgProfile, entity_space: list[str]
        ) -> dict[str, Partition]:
    if kg_profile.partitions:
        return {p.name: p for p in kg_profile.partitions}
    return get_assigned_default_partitions(kg_manager, entity_space)


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


def retrieve_members(
        manager: KgManager, membership_sparql: str, entity_space: Iterable[str] = ()
        ) -> set[IRI]:
    """The IRIs that match a membership pattern (?s), in the entity space if one is given."""
    query = f"SELECT DISTINCT ?s WHERE {{ {membership_sparql} FILTER(isIRI(?s)) }}"
    space = tuple(entity_space)
    return {
        IRI(row["s"].value) for row in select_rows(manager, query)
        if not space or row["s"].value.startswith(space)
        }


def retrieve_groups_by_entity(
        manager: KgManager, grouping: Grouping | None, entity_space: Iterable[str]
        ) -> dict[IRI, set[str]]:
    """
    The entities of a grouping in the entity space with the names of their
    groups, one query per membership; the entities of grouping.minus are left
    out. Exclusive (partitions): an entity belongs to the first group that
    has it. Without a grouping: the default partitions.
    """
    grouping = grouping or default_partitions_to_grouping()
    excluded: set[IRI] = set()
    for membership_sparql in grouping.minus:
        excluded |= retrieve_members(manager, membership_sparql)
    groups_by_entity: dict[IRI, set[str]] = defaultdict(set)
    # exclusive: the members in order, the first group of an entity wins
    for group_name, membership_sparql in grouping.members:
        for entity in retrieve_members(manager, membership_sparql, entity_space) - excluded:
            # - {group_name}: a group may be defined by several memberships
            if grouping.exclusive and groups_by_entity[entity] - {group_name}:
                continue
            groups_by_entity[entity].add(group_name)
    return dict(groups_by_entity)


def members_by_group(groups_by_entity: dict[IRI, set[str]]) -> dict[str, set[IRI]]:
    """The entities of each group, from the groups of each entity."""
    members: dict[str, set[IRI]] = defaultdict(set)
    for entity, group_names in groups_by_entity.items():
        for group_name in group_names:
            members[group_name].add(entity)
    return dict(members)


def retrieve_counts_by_group(
        manager: KgManager,
        grouping: Grouping | None,
        space: Iterable[str],
        sparql_pattern: str
        ) -> dict[str, int]:
    """
    Per group: its entities with at least one value
    of the pattern (over ?s and ?o).
    """
    groups_by_entity = retrieve_groups_by_entity(manager, grouping, space)
    entities_with_value = retrieve_values(
        manager,
        groups_by_entity,
        [sparql_pattern],
        literals_only=False
        )

    return dict(Counter(
        group_name for entity in entities_with_value
        for group_name in groups_by_entity[entity]
        ))


def retrieve_group_sizes(
        manager: KgManager, grouping: Grouping | None, space: Iterable[str]
        ) -> dict[str, int]:
    """
    Per group: its entities (an entity in several
    groups counts in each).
    """
    return dict(Counter(
        group_name
        for group_names in retrieve_groups_by_entity(manager, grouping, space).values()
        for group_name in group_names
        ))


def profile_groups_to_grouping(
        kg_profile: KgProfile, additional_groups: Sequence[GroupDefinition] = ()
        ) -> Grouping:
    return groups_to_grouping(kg_profile.groups).extended_by(
        additional_minus=kg_profile.excluded_group_membership_sparqls()
    ).extended_by(
        additional_members=tuple((g.name, g.membership_sparql) for g in additional_groups)
        )


# --- populations of a strategy: their groups, grouping and entities ---

def group_refs_of_side(population: Population, side: Side) -> list[GroupRef]:
    return population.source_group_refs if side == Side.SOURCE else population.target_group_refs


def inline_groups(population: Population, side: Side) -> list[GroupDefinition]:
    """The groups a strategy defines inline for one side (not in the KgProfile)."""
    return [group_ref for group_ref in group_refs_of_side(population, side) if isinstance(group_ref, GroupDefinition)]


def population_group_names(kg_profile: KgProfile, population: Population, side: Side) -> set[str]:
    """
    The names of the groups of one side's population. REST stands for all groups
    of the profile that are not excluded (for now; later: only the entities that
    have no equivalence yet). Raises FunctionCallException for a name that is
    neither a group of the profile nor defined inline, and for an inline group
    named like a group of the profile.
    """
    clashing = sorted(group.name for group in inline_groups(population, side)
                      if kg_profile.has_group(group.name))
    if clashing:
        raise FunctionCallException(
            f"Groups defined inline must have new names; already groups of {kg_profile.kg_name}: {clashing}")

    names: set[str] = set()
    for group_ref in group_refs_of_side(population, side):
        if isinstance(group_ref, GroupDefinition):
            names.add(group_ref.name)
        elif group_ref == POPULATION_REST:
            names |= set(kg_profile.group_names()) - set(kg_profile.excluded_group_names())
        elif kg_profile.has_group(group_ref):
            names.add(group_ref)
        else:
            raise FunctionCallException(
                f"{group_ref} is not a group of {kg_profile.kg_name}; groups: {kg_profile.group_names()}, "
                f"or {POPULATION_REST!r} for all of them, or define a group inline")
    return names


def generate_group_grouping(kg_profile: KgProfile, population: Population, side: Side) -> Grouping:
    """
    The groups whose values are retrieved for one side: in a small KG all groups of
    the profile and the inline groups (all are needed to count rivals outside the
    population by group), without the entities of excluded groups.
    """
    return profile_groups_to_grouping(kg_profile, inline_groups(population, side))


def generate_population_groups_grouping(kg_profile: KgProfile, population: Population, side: Side) -> Grouping:
    """
    The groups of one side's population only 
    (no rivals), without the entities
    of excluded groups.
    """
    all_groups = profile_groups_to_grouping(kg_profile, inline_groups(population, side))
    group_names = population_group_names(kg_profile, population, side)
    return Grouping(
        tuple(member for member in all_groups.members if member[0] in group_names),
        exclusive=False, minus=all_groups.minus,
    )


def in_population(pair: tuple[IRI, IRI], population_by_side: dict[Side, frozenset[IRI]]) -> bool:
    """Both entities of a (source, target) pair are in their side's population."""
    return pair[0] in population_by_side[Side.SOURCE] and pair[1] in population_by_side[Side.TARGET]


def retrieve_side_population(
        manager: KgManager,
        kg_profile: KgProfile,
        population: Population,
        side: Side,
        leave_out_entities: Collection[IRI] = ()
) -> frozenset[IRI]:
    """
    The entities of one side's population, each once (also those without any
    value of the strategy: the denominator of the signals), without the
    leave_out_entities (those with an equivalence: they get no candidates).
    """
    groups_by_entity = retrieve_groups_by_entity(
        manager,
        generate_population_groups_grouping(kg_profile, population, side),
        kg_profile.entity_space
        )
    return frozenset(groups_by_entity) - frozenset(leave_out_entities)
