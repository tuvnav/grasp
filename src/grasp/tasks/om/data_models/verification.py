"""
Data model of verification and certification: verdicts, judge variants, sampling plans and their results.
"""
from enum import StrEnum
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict

from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidates import RunEvidence
from grasp.tasks.om.data_models.profile import KgProfile
from grasp.tasks.om.foundation.types import IRI, EntityKey, Side


# all candidates the strategies found so far for an entity (side, IRI), best first
type CandidatesOf = Callable[[Side, IRI], list[IRI]]
# the same, in their global Pareto layers over the ranks in all runs, best first
type GetCandidateLayersOf = Callable[[Side, IRI], list[list[IRI]]]
# per candidate of an entity (side, IRI): what every run found for the pair
type GetCandidateSignalsOf = Callable[[Side, IRI], dict[IRI, list[RunEvidence]]]


class VerdictOption(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNSURE = "unsure"
    ERROR = "error"


class Verdict(BaseModel):
    source_iri: IRI
    target_iri: IRI
    verdict: VerdictOption
    rationale: str
    mappings: list[tuple[IRI, IRI]] | None = None


class VerifyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_kg: str
    target_kg: str
    source_profile: KgProfile
    target_profile: KgProfile
    source_iri: IRI
    target_iri: IRI
    agentic: bool = False
    competitors: dict[Side, list[IRI]] = {}
    # show the profiles of both knowledge graphs before the entities
    profiles: bool = False


class JudgeMode(StrEnum):
    ONE_SHOT = "one_shot"  # A1: no KG functions, decision from the given context
    AGENTIC = "agentic"  # A2: KG functions within a small step budget
    ESCALATING = "escalating"  # A3: one-shot, agentic only if not accepted


class JudgeVariant(BaseModel):
    """How an anchor is judged (A1-A3, each optionally with competitors = A4)."""
    model_config = ConfigDict(extra="forbid")
    mode: JudgeMode = JudgeMode.ONE_SHOT
    competitors: bool = False
    # the judge sees the profiles of both knowledge graphs
    profiles: bool = False
    # step budget of the agentic mode
    max_steps: int = 8

    @property
    def name(self) -> str:
        return (self.mode.value + ("+competitors" if self.competitors else "")
                + ("+profiles" if self.profiles else ""))


class Judgment(Verdict):
    """
    A verdict with how it came about, one per judged anchor.
    """
    variant: str
    steps: int  # LLM responses, summed over an escalation
    elapsed: float  # seconds, summed over an escalation
    error: str | None = None  # the run failed; the verdict is then unsure
    # escalating mode: verdict of the one-shot run if the agentic run followed
    first_verdict: VerdictOption | None = None


class CertificationParams(BaseModel):
    """
    Two-point sampling plan per stratum (hypergeometric): the smallest c, and
    for it the smallest n, with P(<= c rejects | p_max) <= alpha and
    P(<= c rejects | p_good) >= 1 - beta. Provisional values, to be revisited.
    """
    model_config = ConfigDict(extra="forbid")
    p_good: float = 0.01
    p_max: float = 0.05
    alpha: float = 0.05
    beta: float = 0.10
    p_fn: float = 0.05
    p_fp: float = 0.05
    # how unsure verdicts count: as rejects (conservative) or not at all (the
    # judged anchor is then replaced by the next one of the sample order)
    unsure_as_reject: bool = False
    seed: int = 0
    stop_rule: Literal["fixed", "sequential"] = "fixed"


class StopDecision(StrEnum):
    """What a stop rule says after each counted verdict of a stratum."""
    ACCEPT_STRATUM = "accept_stratum"
    REJECT_STRATUM = "reject_stratum"
    CONTINUE = "continue"


# stop rule of one stratum: (counted verdicts, rejects among them) -> decision
type StopRule = Callable[[int, int], StopDecision]


class StratumDecisionType(StrEnum):
    ACCEPTED = "accepted"  # sample passed: all pairs certified except those judged not accept
    REJECTED = "rejected"  # stop rule rejected: only the pairs judged accept are kept
    INSPECTED = "inspected"  # no decision before the end: every pair judged individually


class StratumDecision(BaseModel):
    """
    The decision on one stratum and the numbers the stop rule decided on; the
    rule and its parameters are the same for all strata of a certification.
    """
    stratum: str
    size: int
    decision: StratumDecisionType
    counted: int  # verdicts counted by the stop rule (skipped unsure and errors not)
    rejects: int  # including unsure verdicts if they count as rejects


class StratumCertification(BaseModel):
    """
    The result of certifying one stratum.
    """
    decision: StratumDecision
    judgments: list[Judgment]  # in the order of judging
    # accepted pairs of a stratum, either all inspected
    # or only a sample for big strata
    certified_pairs: list[tuple[IRI, IRI]]


class CertificationResult(BaseModel):
    source_kg: str
    target_kg: str
    variant: str
    params: CertificationParams
    strata: list[StratumCertification]


# --- judging of candidate lists ------------------------------------------------
# A judge gets a unit: one or two focus entities, each with its candidate list,
# and searches the KGs with all functions; it sets equivalences (also with
# entities outside the lists) and finishes. Every unit is judged (no
# acceptance sampling); its stratum is only kept for the evaluation.


class ListedCandidate(BaseModel):
    """
    A candidate of a focus entity's list, 
    with the evidence of every run that found it.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    entity: IRI
    pareto_layer: int  # global pareto rank over all runs of all rounds
    signals: list[RunEvidence] = []


class FocusEntity(BaseModel):
    """
    An entity a judge has to find the partner(s) of, with its candidates.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    side: Side
    entity: IRI
    # the cut Pareto list of the round in rank order, then every further
    # candidate found so far, best first; empty: a bare search
    candidates: list[ListedCandidate] = []

    @property
    def candidate_entities(self) -> list[IRI]:
        return [candidate.entity for candidate in self.candidates]


class JudgingUnit(BaseModel):
    """
    What one judge run is about: a single focus entity, or the two entities
    of a mutual pair (each the other's only best candidate).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    focus: list[FocusEntity]

    @property
    def key(self) -> str:
        """Unique per unit, e.g. "source:s1|target:t1" for a mutual pair."""
        return "|".join(sorted(f"{Side(focus.side).value}:{focus.entity}" for focus in self.focus))

    @property
    def is_mutual(self) -> bool:
        return len(self.focus) == 2

    def is_focus(self, side: Side, entity: IRI) -> bool:
        return any(Side(focus.side) == Side(side) and focus.entity == entity for focus in self.focus)


class CandidateJudgeVariant(BaseModel):
    """How a unit is shown to the judge; the ablation settings."""
    model_config = ConfigDict(extra="forbid")
    # focused: the focus entities are named; clusters: the judge only sees two
    # clusters of similar entities (the orchestration still knows the focus)
    presentation: Literal["focused", "clusters"] = "focused"
    show_signals: bool = True      # the signals that ranked the candidates
    # pareto: the candidates in rank order; random: shuffled per unit (seeded by the unit)
    candidate_order: Literal["pareto", "random"] = "pareto"
    max_steps: int = 20

    @property
    def name(self) -> str:
        return "+".join([
            self.presentation,
            "signals" if self.show_signals else "no_signals",
            "ranked" if self.candidate_order == "pareto" else "shuffled",
            ])


class UnitJudgment(BaseModel):
    """The result of judging one unit."""
    model_config = ConfigDict(extra="forbid")
    unit: JudgingUnit
    stratum: str                           # for the evaluation only; not shown to the judge
    equivalences: list[tuple[IRI, IRI]]    # all set, also outside the lists
    unsure_focus: list[EntityKey] = []     # focus entities marked unsure at finish
    finished: bool                         # the judge called finish
    rationale: str = ""
    variant: str
    steps: int
    elapsed: float
    error: str | None = None               # the run failed


class SearchVerifyInput(BaseModel):
    """The input of om-search-verify: the unit to judge, how it is shown, and
    the alignment state (the KGs' profiles, the decisions so far)."""
    model_config = ConfigDict(extra="forbid")
    alignment_state: AlignmentState
    unit: JudgingUnit
    variant: CandidateJudgeVariant = CandidateJudgeVariant()


class SearchVerifyResult(BaseModel):
    """The output of om-search-verify; the orchestration makes the UnitJudgment of it."""
    model_config = ConfigDict(extra="forbid")
    equivalences: list[tuple[IRI, IRI]]  # set by the judge, also outside the lists
    unsure: list[EntityKey] = []         # entities the judge marked unsure at finish
    finished: bool                       # the judge called finish
    rationale: str = ""


# --- recall estimation ------------------------------------------------------------
# A stratified random sample of the source entities without a partner is
# searched by the judge without candidate lists; the share with a partner found
# per stratum, weighted with the stratum sizes, estimates the missed
# equivalences.
# strata of the unmatched source entities, assigned in this order
RECALL_STRATA = ("neighbours", "excluded", "rest")


class RecallParams(BaseModel):
    """Recall estimation and stop criterion. Provisional values."""
    model_config = ConfigDict(extra="forbid")
    sample_size: int = 200
    # sample shares of the strata in the first round (later: Neyman allocation
    # with the shares found in the round before)
    first_round_shares: dict[str, float] = {"neighbours": 0.4, "excluded": 0.2, "rest": 0.4}
    min_per_stratum: int = 10
    confidence: float = 0.95
    # stop: lower confidence bound of the recall reached ...
    target_recall: float = 0.9
    # ... or the estimated missed equivalences fell by less than this share
    # since the round before ...
    min_relative_improvement: float = 0.1
    # ... or this many candidate rounds are done
    max_rounds: int = 5
    seed: int = 0


class RecallStratumEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stratum: str
    size: int                 # unmatched source entities in the stratum
    judged: int               # sampled and judged to the end
    found: int                # of them, with a partner found (or unsure)
    estimated_missed: float   # size * found / judged
    missed_upper: float       # with the upper confidence bound of the share


class RecallEstimate(BaseModel):
    """
    The recall of a round, relative to the judge: accepted / (accepted +
    missed), with a lower bound from the upper bounds of the missed ones.
    """
    model_config = ConfigDict(extra="forbid")
    round: int
    accepted: int
    strata: list[RecallStratumEstimate]
    estimated_missed: float
    missed_upper: float
    recall: float
    recall_lower: float
    confidence: float
