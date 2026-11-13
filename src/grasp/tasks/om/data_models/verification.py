"""
Data model of verification and certification: verdicts, judge variants, sampling plans and their results.
"""
from enum import StrEnum
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict

from grasp.tasks.om.data_models.anchors import AnchorRecord
from grasp.tasks.om.data_models.profile import KgProfile
from grasp.tasks.om.foundation.types import IRI, Side


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
    source_iri: IRI          # das zu prüfende Paar
    target_iri: IRI
    # vom Aufrufer gewählt; leer = ohne Konkurrenten
    competitors: dict[Side, list[IRI]] = {}


class JudgeMode(StrEnum):
    ONE_SHOT = "one_shot"  # A1: no KG functions, decision from the given context
    AGENTIC = "agentic"  # A2: KG functions within a small step budget
    ESCALATING = "escalating"  # A3: one-shot, agentic only if not accepted


class JudgeVariant(BaseModel):
    """How an anchor is judged (A1-A3, each optionally with competitors = A4)."""
    model_config = ConfigDict(extra="forbid")
    mode: JudgeMode = JudgeMode.ONE_SHOT
    competitors: bool = False
    # step budget of the agentic mode
    max_steps: int = 8

    @property
    def name(self) -> str:
        return self.mode.value + ("+competitors" if self.competitors else "")


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
