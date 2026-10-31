"""
Data model of verification and certification: verdicts, judge variants, sampling plans and their results.
"""
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from grasp.tasks.om.data_models.anchors import AnchorRecord


class VerdictOption(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNSURE = "unsure"


class Verdict(AnchorRecord):
    verdict: VerdictOption
    rationale: str


class VerifyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_kg: str
    target_kg: str
    anchor: AnchorRecord


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
    """A verdict with how it came about, one per judged anchor."""
    variant: str
    model: str
    steps: int  # LLM responses, summed over an escalation
    elapsed: float  # seconds, summed over an escalation
    error: str | None = None  # the run failed; the verdict is then unsure
    # escalating mode: verdict of the one-shot run if the agentic run followed
    first_verdict: VerdictOption | None = None


class CertificationParams(BaseModel):
    """Two-point sampling plan per stratum (hypergeometric): the smallest c, and
    for it the smallest n, with P(<= c rejects | p_max) <= alpha and
    P(<= c rejects | p_good) >= 1 - beta. Provisional values, to be revisited."""
    model_config = ConfigDict(extra="forbid")
    p_good: float = 0.01
    p_max: float = 0.05
    alpha: float = 0.05
    beta: float = 0.10
    # how unsure verdicts count: as rejects (conservative) or not at all (the
    # judged anchor is then replaced by the next one of the sample order)
    unsure_as_reject: bool = True
    seed: int = 0


class StratumDecisionType(StrEnum):
    ACCEPTED = "accepted"  # sample passed: all anchors of the stratum are certified
    REJECTED = "rejected"  # more than c rejects: only accepted judged anchors are kept
    INSPECTED = "inspected"  # no plan with n < N: every anchor judged individually


class StratumDecision(BaseModel):
    stratum: str
    size: int
    plan: tuple[int, int] | None  # (n, c); None for complete inspection
    decision: StratumDecisionType
    judged: int
    rejects: int  # including unsure verdicts if they count as rejects
    unsure: int


class CertificationResult(BaseModel):
    source_kg: str
    target_kg: str
    variant: str
    params: CertificationParams
    strata: list[StratumDecision]
    judgments: list[Judgment]
    # anchors certified: those of accepted strata and those accepted individually
    certified: list[AnchorRecord]
