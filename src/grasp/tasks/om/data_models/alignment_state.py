"""
State of the alignment of one KG pair, passed through the stages after the KG
profiles (the anchor stage later, the candidate stage now):
    source, target                the profiles of the two KGs: the working profiles
                                  of the pair (copies of the profile pipeline's
                                  output, changed only here by the stages of the pair)
    equivalences                  the equivalences accepted so far
    rejected_equivalences         the equivalences rejected so far (not proposed again)
    cross_kg_non_competing_pairs  partitions of the two KGs that do not compete
    skipped                       populations not worth searching

The two equivalence lists are a snapshot of the current decisions: their
history (who proposed, judged, accepted or rejected what, and why) is kept in
the orchestration's ledger of events (data_models/equivalence.py), which
builds a new state before every stage (with_decisions). The lists are never
changed in place; the index over them is built when the state is constructed.

Competition: uniqueness, ambiguity and rank of a candidate are measured against
all entities of both KGs except those in partitions that do not compete with its
partition. Partitions (profile stage b1) are exclusive, so the partition of an
entity and thus its competitors are unique. Each fact has one place: within a KG,
the non-competing pairs of its profile; across the KGs, the non-competing pairs
here (none at the start: every partition competes until a stage of the pair
forbids it).
"""
from collections import Counter
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from grasp.tasks.om.data_models.equivalence import (
    Equivalence,
    ExistingEquivalences,
    build_existing_equivalences_index,
)
from grasp.tasks.om.data_models.profile import (
    KgProfile,
    WithinKgNonCompetingPartitionPair,
)
from grasp.tasks.om.data_models.candidates import (
    Population,
)
from grasp.tasks.om.foundation.types import IRI, EntityKey, Side


class CrossKgNonCompetingPartitionPair(BaseModel):
    """A partition of the source KG and one of the target KG whose entities do
    not compete with each other."""
    model_config = ConfigDict(extra="forbid")
    source_partition: str
    target_partition: str
    reason: str
    set_by: str  # the stage that set it


class CrossKgCompetitionOutput(BaseModel):
    """Output of om-cross-kg-competition: the partitions of the two KGs that do not compete."""
    model_config = ConfigDict(extra="forbid")
    finished: bool
    rationale: str | None = None
    non_competing_pairs: list[CrossKgNonCompetingPartitionPair]


class NonCompetingPartition(BaseModel):
    """A partition that does not compete with a given one (non_competing_partitions)."""
    model_config = ConfigDict(extra="forbid")
    side: Side
    partition: str
    reason: str
    set_by: str


class SkippedScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population: Population
    reason: str
    set_by: str


class AlignmentState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_kg_profile: KgProfile
    target_kg_profile: KgProfile
    equivalences: list[Equivalence] = []
    rejected_equivalences: list[Equivalence] = []
    cross_kg_non_competing_pairs: list[CrossKgNonCompetingPartitionPair] = []
    skipped: list[SkippedScope] = []
    # per entity its accepted and its rejected equivalences (built from the lists)
    _accepted_by_entity: dict[EntityKey, list[Equivalence]] = PrivateAttr(default_factory=dict)
    _rejected_by_entity: dict[EntityKey, list[Equivalence]] = PrivateAttr(default_factory=dict)
    _existing_equivalences: ExistingEquivalences = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def cross_kg_pairs_name_partitions(self) -> Self:
        for pair in self.cross_kg_non_competing_pairs:
            self.check_known_partitions(Side.SOURCE, pair.source_partition, Side.TARGET, pair.target_partition)
        return self

    @model_validator(mode="after")
    def each_equivalence_has_one_status(self) -> Self:
        """No equivalence twice in a list, none both accepted and rejected."""
        for name, equivalences in (
                ("equivalences", self.equivalences),
                ("rejected_equivalences", self.rejected_equivalences)):
            key_counts = Counter(equivalence.key for equivalence in equivalences)
            duplicates = sorted(key for key, count in key_counts.items() if count > 1)
            if duplicates:
                raise ValueError(f"equivalences more than once in {name}: {duplicates}")
        both = (
            {equivalence.key for equivalence in self.equivalences}
            & {equivalence.key for equivalence in self.rejected_equivalences}
            )
        if both:
            raise ValueError(f"equivalences both accepted and rejected: {sorted(both)}")
        return self

    def model_post_init(self, context: Any) -> None:
        self._accepted_by_entity = index_by_entity(self.equivalences)
        self._rejected_by_entity = index_by_entity(self.rejected_equivalences)
        self._existing_equivalences = build_existing_equivalences_index(self.equivalences)

    @classmethod
    def create(cls, source: KgProfile, target: KgProfile) -> "AlignmentState":
        """A new state with copies of the profiles; across the KGs, all partitions compete."""
        return cls(source_kg_profile=source.model_copy(deep=True), target_kg_profile=target.model_copy(deep=True))

    def with_decisions(
            self,
            equivalences: list[Equivalence],
            rejected_equivalences: list[Equivalence]
            ) -> "AlignmentState":
        """
        A new state with these current decisions (from the ledger) and the
        rest of this one. Constructed anew, so that the checks and the index
        run (model_copy runs neither).
        """
        return type(self)(
            **self.model_dump(exclude={"equivalences", "rejected_equivalences"}),
            equivalences=equivalences,
            rejected_equivalences=rejected_equivalences,
            )

    def with_cross_kg_non_competing_pairs(
            self,
            non_competing_pairs: list[CrossKgNonCompetingPartitionPair]
            ) -> "AlignmentState":
        """A new state with these pairs across the KGs (checked against the partitions)."""
        return type(self)(
            **self.model_dump(exclude={"cross_kg_non_competing_pairs"}),
            cross_kg_non_competing_pairs=non_competing_pairs,
            )

    @property
    def source_kg_name(self) -> str:
        return self.source_kg_profile.kg_name

    @property
    def target_kg_name(self) -> str:
        return self.target_kg_profile.kg_name

    @property
    def aligned_entities(self) -> dict[Side, set[IRI]]:
        """
        Per side the entities with an accepted equivalence.
        """
        aligned: dict[Side, set[IRI]] = {Side.SOURCE: set(), Side.TARGET: set()}
        for side, entity in self._accepted_by_entity:
            aligned[side].add(entity)
        return aligned

    def existing_equivalences(self) -> ExistingEquivalences:
        """
        The accepted equivalences per side by the entity of that side (the
        witnesses of a neighbour strategy); built once with the state.
        """
        return self._existing_equivalences

    def equivalences_of(self, entity: EntityKey) -> list[Equivalence]:
        """The accepted equivalences of the entity (side, IRI)."""
        return list(self._accepted_by_entity.get(entity_key(entity), []))

    def rejected_partners(self, entity: EntityKey) -> set[IRI]:
        """The entities of the other side rejected as partners of the entity."""
        side, iri = entity_key(entity)
        return {
            partner for equivalence in self._rejected_by_entity.get((side, iri), [])
            if (partner := equivalence.partner_of(side, iri)) is not None
            }

    def get_kg_profile(self, side: Side) -> KgProfile:
        return self.source_kg_profile if Side(side) == Side.SOURCE else self.target_kg_profile

    def check_known_partitions(self, side_a: Side, partition_a: str, side_b: Side, partition_b: str) -> None:
        for side, partition in ((side_a, partition_a), (side_b, partition_b)):
            if partition not in self.get_kg_profile(side).partition_names():
                raise ValueError(f"unknown partition {partition} of the {Side(side).value} KG; partitions: "
                                 f"{', '.join(self.get_kg_profile(side).partition_names()) or 'none'}")
        if Side(side_a) == Side(side_b) and partition_a == partition_b:
            raise ValueError("entities of one partition always compete with each other")

    def forbid_competition(self, side_a: Side, partition_a: str, side_b: Side, partition_b: str,
                           reason: str, set_by: str) -> None:
        """The two partitions no longer compete: within a KG as a non-competing
        pair of its profile, across the KGs as one of the state."""
        self.allow_competition(side_a, partition_a, side_b, partition_b)  # replaces an earlier reason
        if Side(side_a) == Side(side_b):
            self.get_kg_profile(side_a).non_competing_pairs.append(WithinKgNonCompetingPartitionPair(
                partition_a=partition_a, partition_b=partition_b, reason=reason, set_by=set_by))
            return
        source_partition, target_partition = self.by_side(side_a, partition_a, partition_b)
        self.cross_kg_non_competing_pairs.append(CrossKgNonCompetingPartitionPair(
            source_partition=source_partition, target_partition=target_partition, reason=reason, set_by=set_by))

    def allow_competition(self, side_a: Side, partition_a: str, side_b: Side, partition_b: str) -> bool:
        """Lets the two partitions compete again; whether they did not before."""
        self.check_known_partitions(side_a, partition_a, side_b, partition_b)
        if Side(side_a) == Side(side_b):
            profile = self.get_kg_profile(side_a)
            kept = [pair for pair in profile.non_competing_pairs if pair.partitions() != {partition_a, partition_b}]
            found = len(kept) < len(profile.non_competing_pairs)
            profile.non_competing_pairs = kept
            return found
        source_partition, target_partition = self.by_side(side_a, partition_a, partition_b)
        kept = [pair for pair in self.cross_kg_non_competing_pairs
                if (pair.source_partition, pair.target_partition) != (source_partition, target_partition)]
        found = len(kept) < len(self.cross_kg_non_competing_pairs)
        self.cross_kg_non_competing_pairs = kept
        return found

    @staticmethod
    def by_side(side_a: Side, partition_a: str, partition_b: str) -> tuple[str, str]:
        """(source partition, target partition) of two partitions of different KGs."""
        return (partition_a, partition_b) if Side(side_a) == Side.SOURCE else (partition_b, partition_a)

    def competes(self, side_a: Side, partition_a: str, side_b: Side, partition_b: str) -> bool:
        """Whether an entity of partition_b (side_b) may be a candidate or a
        rival for an entity of partition_a (side_a). Partitions, not entities:
        the caller looks up the partition of an entity first."""
        if Side(side_a) == Side(side_b):
            return self.get_kg_profile(side_a).competes(partition_a, partition_b)
        source_partition, target_partition = self.by_side(side_a, partition_a, partition_b)
        return all((pair.source_partition, pair.target_partition) != (source_partition, target_partition)
                   for pair in self.cross_kg_non_competing_pairs)

    def non_competing_partitions(self, side: Side, partition: str) -> list[NonCompetingPartition]:
        """The partitions of both KGs that do not compete with this one."""
        side = Side(side)
        other_side = Side.TARGET if side == Side.SOURCE else Side.SOURCE
        result = [
            NonCompetingPartition(side=side, partition=pair.partition_b if pair.partition_a == partition else pair.partition_a,
                                  reason=pair.reason, set_by=pair.set_by)
            for pair in self.get_kg_profile(side).non_competing_pairs if partition in pair.partitions()
        ]
        for pair in self.cross_kg_non_competing_pairs:
            own, other = ((pair.source_partition, pair.target_partition) if side == Side.SOURCE
                          else (pair.target_partition, pair.source_partition))
            if own == partition:
                result.append(NonCompetingPartition(side=other_side, partition=other, reason=pair.reason,
                                                    set_by=pair.set_by))
        return result


def entity_key(entity: EntityKey) -> EntityKey:
    """The key with the side as Side (it may come as a string from JSON)."""
    return (Side(entity[0]), entity[1])


def index_by_entity(equivalences: list[Equivalence]) -> dict[EntityKey, list[Equivalence]]:
    """Per entity (side, IRI) the equivalences it is part of."""
    index: dict[EntityKey, list[Equivalence]] = {}
    for equivalence in equivalences:
        for entity in equivalence.mentioned_entities():
            index.setdefault(entity, []).append(equivalence)
    return index
