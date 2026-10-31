"""
State of the alignment of one KG pair, passed through the stages after the KG
profiles (the anchor stage later, the candidate stage now):
    source, target   the profiles of the two KGs: the working profiles of the
                     pair (copies of the profile pipeline's output, changed
                     only here by the stages of the pair)
    correspondences  accepted correspondences, with their provenance
    rejected         pairs judged wrong (counter-evidence; not proposed again)
    reviewed         source entities the judge has looked at, and the outcome
    exclusions       which partitions of different KGs do not compete
    skipped          populations not worth searching

Competition: uniqueness, ambiguity and rank of a candidate are measured against
all entities of both KGs except those in partitions that do not compete with its
partition. Partitions (profile stage b1) are exclusive, so the partition of an
entity and thus its competitors are unique. Each fact has one place: within a KG,
the separations of its profile; across the KGs, the exclusions here (none at the
start: every partition competes until a pair stage excludes pairs).
"""
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from grasp.tasks.om.model import IRI, SIDES, Correspondence, Side
from grasp.tasks.om.profile_model import KgProfile, Separation
from grasp.tasks.om.scope_model import Population


def partition_key(side: Side, partition: str) -> str:
    """A partition of one side, e.g. source:class."""
    return f"{Side(side).value}:{partition}"


class Exclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str
    set_by: str  # the stage that set it


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
    source: KgProfile
    target: KgProfile
    correspondences: list[Correspondence] = []
    rejected: list[Correspondence] = []
    reviewed: list[ReviewedSource] = []
    # partition_key -> partition_key of a partition of the other KG it does not
    # compete with -> why; symmetric (each exclusion is stored for both partitions)
    exclusions: dict[str, dict[str, Exclusion]] = {}
    skipped: list[SkippedScope] = []

    @model_validator(mode="after")
    def exclusions_valid(self) -> Self:
        known = {partition_key(side, name) for side in SIDES for name in self.profile(side).partitions}
        for a, excluded in self.exclusions.items():
            for b, exclusion in excluded.items():
                if a.split(":", 1)[0] == b.split(":", 1)[0]:
                    raise ValueError(f"exclusion {a} - {b} within one KG: a separation of its profile")
                if {a, b} - known:
                    raise ValueError(f"exclusion {a} - {b} names unknown partitions {sorted({a, b} - known)}")
                if self.exclusions.get(b, {}).get(a) != exclusion:
                    raise ValueError(f"exclusion {a} - {b} is not stored for both partitions alike")
        return self

    @classmethod
    def create(cls, source: KgProfile, target: KgProfile) -> "AlignmentState":
        """A new state with copies of the profiles; across the KGs, all partitions compete."""
        return cls(source=source.model_copy(deep=True), target=target.model_copy(deep=True))

    @property
    def source_kg(self) -> str:
        return self.source.kg_name

    @property
    def target_kg(self) -> str:
        return self.target.kg_name

    def profile(self, side: Side) -> KgProfile:
        return self.source if Side(side) == Side.SOURCE else self.target

    def check_partitions(self, side_a: Side, a: str, side_b: Side, b: str) -> None:
        for side, name in ((side_a, a), (side_b, b)):
            if name not in self.profile(side).partitions:
                raise ValueError(f"unknown partition {partition_key(side, name)}; partitions: "
                                 f"{', '.join(self.profile(side).partitions) or 'none'}")
        if Side(side_a) == Side(side_b) and a == b:
            raise ValueError("entities of one partition always compete with each other")

    def exclude(self, side_a: Side, a: str, side_b: Side, b: str, reason: str, set_by: str) -> None:
        """The two partitions no longer compete: within a KG as a separation of
        its profile, across the KGs as an exclusion of the state."""
        self.check_partitions(side_a, a, side_b, b)
        if Side(side_a) == Side(side_b):
            profile = self.profile(side_a)
            profile.separations = [s for s in profile.separations if s.pair() != {a, b}]
            profile.separations.append(Separation(a=a, b=b, reason=reason, set_by=set_by))
            return
        exclusion = Exclusion(reason=reason, set_by=set_by)
        key_a, key_b = partition_key(side_a, a), partition_key(side_b, b)
        self.exclusions.setdefault(key_a, {})[key_b] = exclusion
        self.exclusions.setdefault(key_b, {})[key_a] = exclusion

    def include(self, side_a: Side, a: str, side_b: Side, b: str) -> bool:
        """Lets the two partitions compete again; whether they did not before."""
        self.check_partitions(side_a, a, side_b, b)
        if Side(side_a) == Side(side_b):
            profile = self.profile(side_a)
            kept = [s for s in profile.separations if s.pair() != {a, b}]
            found = len(kept) < len(profile.separations)
            profile.separations = kept
            return found
        key_a, key_b = partition_key(side_a, a), partition_key(side_b, b)
        found = self.exclusions.get(key_a, {}).pop(key_b, None) is not None
        self.exclusions.get(key_b, {}).pop(key_a, None)
        for key in (key_a, key_b):
            if key in self.exclusions and not self.exclusions[key]:
                del self.exclusions[key]
        return found

    def excluded(self, side: Side, partition: str) -> dict[str, Exclusion]:
        """The partitions (partition_key) the partition does not compete with,
        within its KG (separations of its profile) and across."""
        within = {partition_key(side, s.b if s.a == partition else s.a): Exclusion(reason=s.reason, set_by=s.set_by)
                  for s in self.profile(side).separations if partition in s.pair()}
        return within | self.exclusions.get(partition_key(side, partition), {})

    def competes(self, side_a: Side, a: str, side_b: Side, b: str) -> bool:
        """Whether an entity of partition b (side_b) may be a candidate or a
        rival for an entity of partition a (side_a)."""
        if Side(side_a) == Side(side_b):
            return not self.profile(side_a).separated(a, b)
        return partition_key(side_b, b) not in self.exclusions.get(partition_key(side_a, a), {})
