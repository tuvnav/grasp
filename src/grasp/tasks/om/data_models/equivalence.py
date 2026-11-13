"""
Equivalences between entities of two KGs, and their history.

The history is a list of events per equivalence, in the order of the
ledger (an append-only ledger of the orchestration): proposed (a hint, e.g. a
pair set by a judge for an entity outside its focus), accepted or rejected
(by a judge, with a stratum, or later by a conflict resolution or a repair).
The last accepted or rejected event determines the status; who decided and
how is in the provenance. Apart from that, the outcome of a judge's search for
the partner of a focus entity without setting one (no partner, unsure,
unresolved) is kept per entity, and so is each candidate list a stage showed
for an entity (its candidates in rank order: shown, not rejected when not
chosen). The record of an entity is a view computed from all three.
"""
from enum import StrEnum
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict

from grasp.tasks.entities import Entity
from grasp.tasks.om.foundation.types import IRI, EntityKey, Side


class Provenance(BaseModel):
    """Where an event of the pipeline comes from."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: str                     # e.g. anchors, candidates, recall
    round: int = 0                 # round of the meta-loop
    stratum: str | None = None     # stratum of the pair in its stage
    judge_variant: str | None = None
    # the pair itself was judged (not only taken over with its stratum)
    checked_individually: bool = False
    rationale: str | None = None   # the judge's reason


class Equivalence(BaseModel):
    """
    A single equivalence between two entities from different KGs; its
    provenance is in its events.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    entity_source: IRI
    entity_target: IRI

    # OAEI standard
    measure: float = 1.0

    @property
    def key(self) -> tuple[IRI, IRI]:
        return (self.entity_source, self.entity_target)

    def mentioned_entities(self) -> frozenset[EntityKey]:
        """
        The entities the equivalence is about, per side (with complex
        equivalences: all entities in its expressions).
        """
        return frozenset({(Side.SOURCE, self.entity_source), (Side.TARGET, self.entity_target)})

    def partner_of(self, side: Side, entity: IRI) -> IRI | None:
        """
        The entity on the other side, if the entity is on this side.
        """
        if Side(side) == Side.SOURCE and self.entity_source == entity:
            return self.entity_target
        if Side(side) == Side.TARGET and self.entity_target == entity:
            return self.entity_source
        return None


# per side: entity -> the equivalences (witnesses) it takes part in
type ExistingEquivalences = dict[Side, dict[IRI, list[tuple[IRI, IRI]]]]


def build_existing_equivalences_index(equivalences: Iterable[Equivalence]) -> ExistingEquivalences:
    """
    The witnesses of a neighbour strategy: the equivalences among the
    equivalences, per side by the entity of that side (an entity may take
    part in several).
    """
    witnesses_by_entity: ExistingEquivalences = {Side.SOURCE: {}, Side.TARGET: {}}
    for equivalence in equivalences:
        witness = (equivalence.entity_source, equivalence.entity_target)
        witnesses_by_entity[Side.SOURCE].setdefault(witness[0], []).append(witness)
        witnesses_by_entity[Side.TARGET].setdefault(witness[1], []).append(witness)
    return witnesses_by_entity


class EquivalenceEventKind(StrEnum):
    # a hint, changes no status: set by a judge for an entity outside its
    # focus, a competitor's mapping, an unsure verdict on a pair
    PROPOSED = "proposed"
    # set the status; the last one counts
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNSURE = "unsure"

    @property
    def sets_status(self) -> bool:
        return self != EquivalenceEventKind.PROPOSED


class EquivalenceEvent(BaseModel):
    """
    One event in the history of an equivalence. sequence: given by the
    ledger, the order of all events of an alignment.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    sequence: int
    kind: EquivalenceEventKind
    equivalence: Equivalence
    provenance: Provenance


class PartnerSearchOutcomeKind(StrEnum):
    NO_PARTNER = "no_partner"    # there is no partner in the other KG
    UNSURE = "unsure"            # the judge could not decide
    UNRESOLVED = "unresolved"    # the run ended without an outcome (e.g. step budget)


class PartnerSearchOutcome(BaseModel):
    """
    The outcome of a judge's search for the partner of a focus entity, when
    it set none. A partner found later (an accepted equivalence) takes
    precedence; the outcome stays as history.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    sequence: int
    side: Side
    entity: IRI
    outcome: PartnerSearchOutcomeKind
    provenance: Provenance


class CandidateListEvent(BaseModel):
    """
    The candidate list a stage showed for an entity (to a judge), its
    candidates in rank order (best first). Being listed changes no status.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    sequence: int
    side: Side
    entity: IRI
    candidates: list[IRI]
    provenance: Provenance


type CandidateStatus = Literal["open", "accepted", "rejected"]


class EntityRecord(BaseModel):
    """
    The record of an entity: all events of equivalences it is part of, the
    outcomes of searches for its partner and the candidate lists shown for
    it, in the order of the ledger. A view, computed from the ledger
    (EntityRecord.of); never stored as a source.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    side: Side
    entity: IRI
    equivalence_events: list[EquivalenceEvent]
    partner_search_outcomes: list[PartnerSearchOutcome]
    candidate_lists: list[CandidateListEvent] = []

    @classmethod
    def of(
            cls,
            side: Side,
            entity: IRI,
            equivalence_events: list[EquivalenceEvent],
            partner_search_outcomes: list[PartnerSearchOutcome],
            candidate_lists: Iterable[CandidateListEvent] = ()
            ) -> "EntityRecord":
        """The record of the entity from all events (of any entities)."""
        entity_key = (Side(side), entity)

        def own[E: PartnerSearchOutcome | CandidateListEvent](events: Iterable[E]) -> list[E]:
            return sorted(
                (event for event in events if (Side(event.side), event.entity) == entity_key),
                key=lambda event: event.sequence
                )

        return cls(
            side=side,
            entity=entity,
            equivalence_events=sorted(
                (event for event in equivalence_events
                 if entity_key in event.equivalence.mentioned_entities()),
                key=lambda event: event.sequence
                ),
            partner_search_outcomes=own(partner_search_outcomes),
            candidate_lists=own(candidate_lists)
            )

    def candidate_histories(self) -> dict[IRI, list[EquivalenceEvent]]:
        """Per partner its events, oldest first."""
        histories: dict[IRI, list[EquivalenceEvent]] = {}
        for event in self.equivalence_events:
            partner = event.equivalence.partner_of(self.side, self.entity)
            if partner is not None:
                histories.setdefault(partner, []).append(event)
        return histories

    @staticmethod
    def status(history: list[EquivalenceEvent]) -> CandidateStatus:
        """The status of the last accepted or rejected event; open without one."""
        return status_of_history(history)

    def candidates_with_status(
            self,
            status: CandidateStatus
            ) -> dict[IRI, list[EquivalenceEvent]]:
        """The partners with this status, each with its whole history."""
        return {
            partner: history
            for partner, history in self.candidate_histories().items()
            if self.status(history) == status
            }

    @property
    def open_candidates(self) -> dict[IRI, list[EquivalenceEvent]]:
        return self.candidates_with_status("open")

    @property
    def accepted_candidates(self) -> dict[IRI, list[EquivalenceEvent]]:
        return self.candidates_with_status("accepted")

    @property
    def rejected_candidates(self) -> dict[IRI, list[EquivalenceEvent]]:
        return self.candidates_with_status("rejected")

    def listings(self) -> dict[IRI, list[tuple[CandidateListEvent, int]]]:
        """
        Per candidate ever listed for the entity the lists it was in, each
        with its rank there (1 = first), oldest first.
        """
        listings: dict[IRI, list[tuple[CandidateListEvent, int]]] = {}
        for candidate_list in self.candidate_lists:
            for rank, candidate in enumerate(candidate_list.candidates, start=1):
                listings.setdefault(candidate, []).append((candidate_list, rank))
        return listings

    @property
    def latest_partner_search_outcome(self) -> PartnerSearchOutcome | None:
        return self.partner_search_outcomes[-1] if self.partner_search_outcomes else None


def status_after(current: CandidateStatus, kind: EquivalenceEventKind) -> CandidateStatus:
    """The status of an equivalence after an event of this kind (proposed keeps it)."""
    if kind == EquivalenceEventKind.ACCEPTED:
        return "accepted"
    if kind == EquivalenceEventKind.REJECTED:
        return "rejected"
    return current


def status_of_history(history: list[EquivalenceEvent]) -> CandidateStatus:
    """The status after the events of an equivalence, oldest first."""
    status: CandidateStatus = "open"
    for event in history:
        status = status_after(status, event.kind)
    return status


class PotentialEquivalences(BaseModel):
    source_entity: Entity
    candidates: list[Entity]
