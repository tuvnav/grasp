"""
State of the alignment of one KG pair, passed through the stages after the KG
profiles (the anchor stage later, the candidate stage now):
    source, target                the profiles of the two KGs: the working profiles
                                  of the pair (copies of the profile pipeline's
                                  output, changed only here by the stages of the pair)
    correspondences               accepted correspondences, with their provenance
    rejected                      pairs judged wrong (counter-evidence; not proposed again)
    reviewed                      source entities the judge has looked at, and the outcome
    cross_kg_non_competing_pairs  partitions of the two KGs that do not compete
    skipped                       populations not worth searching

Competition: uniqueness, ambiguity and rank of a candidate are measured against
all entities of both KGs except those in partitions that do not compete with its
partition. Partitions (profile stage b1) are exclusive, so the partition of an
entity and thus its competitors are unique. Each fact has one place: within a KG,
the non-competing pairs of its profile; across the KGs, the non-competing pairs
here (none at the start: every partition competes until a stage of the pair
forbids it).
"""
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from grasp.tasks.om.data_models.correspondence import Correspondence
from grasp.tasks.om.data_models.profile import KgProfile, WithinKgNonCompetingPartitionPair
from grasp.tasks.om.data_models.scope import Population
from grasp.tasks.om.foundation.types import IRI, Side


class CrossKgNonCompetingPartitionPair(BaseModel):
    """A partition of the source KG and one of the target KG whose entities do
    not compete with each other."""
    model_config = ConfigDict(extra="forbid")
    source_partition: str
    target_partition: str
    reason: str
    set_by: str  # the stage that set it


class NonCompetingPartition(BaseModel):
    """A partition that does not compete with a given one (non_competing_partitions)."""
    model_config = ConfigDict(extra="forbid")
    side: Side
    partition: str
    reason: str
    set_by: str


class ReviewedSource(BaseModel):
    """A source entity the judge has looked at. With an outcome other than
    match, it is reviewed again only with new candidates."""
    model_config = ConfigDict(extra="forbid")
    source: IRI
    round: int = 0
    outcome: Literal["match", "none_absent", "none_present", "escalated"]
    candidates_seen: list[IRI] = []


class SkippedScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population: Population
    reason: str
    set_by: str


class AlignmentState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_kg_profile: KgProfile
    target_kg_profile: KgProfile
    correspondences: list[Correspondence] = []
    rejected: list[Correspondence] = []
    reviewed: list[ReviewedSource] = []
    cross_kg_non_competing_pairs: list[CrossKgNonCompetingPartitionPair] = []
    skipped: list[SkippedScope] = []

    @model_validator(mode="after")
    def cross_kg_pairs_name_partitions(self) -> Self:
        for pair in self.cross_kg_non_competing_pairs:
            self.check_known_partitions(Side.SOURCE, pair.source_partition, Side.TARGET, pair.target_partition)
        return self

    @classmethod
    def create(cls, source: KgProfile, target: KgProfile) -> "AlignmentState":
        """A new state with copies of the profiles; across the KGs, all partitions compete."""
        return cls(source_kg_profile=source.model_copy(deep=True), target_kg_profile=target.model_copy(deep=True))

    @property
    def source_kg_name(self) -> str:
        return self.source_kg_profile.kg_name

    @property
    def target_kg_name(self) -> str:
        return self.target_kg_profile.kg_name

    def get_profile(self, side: Side) -> KgProfile:
        return self.source_kg_profile if Side(side) == Side.SOURCE else self.target_kg_profile

    def check_known_partitions(self, side_a: Side, partition_a: str, side_b: Side, partition_b: str) -> None:
        for side, partition in ((side_a, partition_a), (side_b, partition_b)):
            if partition not in self.get_profile(side).partition_names():
                raise ValueError(f"unknown partition {partition} of the {Side(side).value} KG; partitions: "
                                 f"{', '.join(self.get_profile(side).partition_names()) or 'none'}")
        if Side(side_a) == Side(side_b) and partition_a == partition_b:
            raise ValueError("entities of one partition always compete with each other")

    def forbid_competition(self, side_a: Side, partition_a: str, side_b: Side, partition_b: str,
                           reason: str, set_by: str) -> None:
        """The two partitions no longer compete: within a KG as a non-competing
        pair of its profile, across the KGs as one of the state."""
        self.allow_competition(side_a, partition_a, side_b, partition_b)  # replaces an earlier reason
        if Side(side_a) == Side(side_b):
            self.get_profile(side_a).non_competing_pairs.append(WithinKgNonCompetingPartitionPair(
                partition_a=partition_a, partition_b=partition_b, reason=reason, set_by=set_by))
            return
        source_partition, target_partition = self.by_side(side_a, partition_a, partition_b)
        self.cross_kg_non_competing_pairs.append(CrossKgNonCompetingPartitionPair(
            source_partition=source_partition, target_partition=target_partition, reason=reason, set_by=set_by))

    def allow_competition(self, side_a: Side, partition_a: str, side_b: Side, partition_b: str) -> bool:
        """Lets the two partitions compete again; whether they did not before."""
        self.check_known_partitions(side_a, partition_a, side_b, partition_b)
        if Side(side_a) == Side(side_b):
            profile = self.get_profile(side_a)
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
            return self.get_profile(side_a).competes(partition_a, partition_b)
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
            for pair in self.get_profile(side).non_competing_pairs if partition in pair.partitions()
        ]
        for pair in self.cross_kg_non_competing_pairs:
            own, other = ((pair.source_partition, pair.target_partition) if side == Side.SOURCE
                          else (pair.target_partition, pair.source_partition))
            if own == partition:
                result.append(NonCompetingPartition(side=other_side, partition=other, reason=pair.reason,
                                                    set_by=pair.set_by))
        return result
