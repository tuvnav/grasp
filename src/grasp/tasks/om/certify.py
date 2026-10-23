"""Certification of the anchor strata by sampling: the bookkeeping around the
om-verify task, which judges a single anchor per run.

- judge: one anchor, one judge variant (A1-A3, optionally with competitors)
- certify_stratum: sampling plan, sample, sequential judging, decision
- certify: all strata of an anchor stage result

GRASP is used as a library: the managers are loaded once by the caller, and
every judgment is a fresh conversation (independent verdicts).
"""
import math
import random
from typing import Callable, Iterator

from scipy.stats import hypergeom
from universal_ml_utils.ops import consume_generator

from grasp.configs import GraspConfig
from grasp.core import generate
from grasp.manager import KgManager
from grasp.tasks.om.model import (
    AnchorRecord, AnchorStageResult, CertificationParams, CertificationResult, JudgeMode,
    JudgeVariant, Judgment, StratumDecision, StratumDecisionType, Verdict, VerdictOption,
    VerifyInput,
)

VERIFY_TASK = "om-verify"
# config nesting: task_kwargs -> om (shared by all om stages) -> om_verify
OM_KWARGS = "om"
VERIFY_KWARGS = "om_verify"

# judges one anchor; certify_stratum and certify only need this signature, so
# that tests can pass a simulated judge (e.g. a gold oracle) instead of an LLM
type JudgeFn = Callable[[VerifyInput], Judgment]
# receives every om-verify run: the input, the mode of the run and its messages
type TraceFn = Callable[[VerifyInput, JudgeMode, list[dict]], None]


# ---------------------------------------------------------------- sampling plan

def sampling_plan(size: int, params: CertificationParams) -> tuple[int, int] | None:
    """(n, c): the smallest acceptance number c, and for it the smallest sample
    size n, with P(<= c rejects | p_max) <= alpha (a bad stratum is accepted
    rarely) and P(<= c rejects | p_good) >= 1 - beta (a good stratum is rejected
    rarely). None if only n = N would do: then every anchor is judged."""
    bad = max(1, math.ceil(params.p_max * size))
    good = math.floor(params.p_good * size)
    for c in range(size):
        # smallest n for the alpha condition; P(<= c | bad) falls with n
        n = next(
            (n for n in range(c + 1, size) if hypergeom.cdf(c, size, bad, n) <= params.alpha),
            None,
        )
        if n is None:
            return None
        # a larger n only lowers P(<= c | good), so try the next c if it fails
        if hypergeom.cdf(c, size, good, n) >= 1 - params.beta:
            return n, c
    return None


# ---------------------------------------------------------------- judging

def variant_config(base: GraspConfig, variant: JudgeVariant, mode: JudgeMode) -> GraspConfig:
    """Configuration of one om-verify run; mode is one-shot or agentic (the
    escalating variant runs both)."""
    om_kwargs = {**base.task_kwargs.get(OM_KWARGS, {})}
    om_kwargs[VERIFY_KWARGS] = {
        **om_kwargs.get(VERIFY_KWARGS, {}),
        "mode": mode.value,
        "competitors": variant.competitors,
    }
    update: dict = {"task_kwargs": {**base.task_kwargs, OM_KWARGS: om_kwargs}}
    if mode == JudgeMode.ONE_SHOT:
        # no KG functions, and the verdict function must be called
        update |= {"fn_set": "none", "tool_choice": "required", "max_steps": 2}
    else:
        update |= {"max_steps": variant.max_steps}
        if base.budget_warning_fraction is None:
            # warn during the last 30% of the steps; the last step must give a verdict
            update["budget_warning_fraction"] = 0.3
    return base.model_copy(update=update, deep=True)


def run_verify(
        verify_input: VerifyInput,
        config: GraspConfig,
        managers: list[KgManager],
        kg_notes: dict[str, list[str]] | None = None,
        notes: list[str] | None = None,
        on_trace: Callable[[list[dict]], None] | None = None,
        ) -> tuple[Verdict | None, int, float, str | None]:
    """One om-verify run: (verdict or None, LLM responses, seconds, error)."""
    result = consume_generator(generate(
        VERIFY_TASK, verify_input.model_dump(mode="json"), config, managers, kg_notes, notes,
    ))
    if on_trace is not None:
        on_trace(result["messages"])
    steps = sum(1 for m in result["messages"] if m.get("role") == "assistant")
    error = result.get("error")
    if result.get("output") is None:
        reason = error["content"] if error else "no verdict submitted"
        return None, steps, result["elapsed"], reason
    return Verdict.model_validate(result["output"]), steps, result["elapsed"], None


def judge(
        verify_input: VerifyInput,
        variant: JudgeVariant,
        base_config: GraspConfig,
        managers: list[KgManager],
        kg_notes: dict[str, list[str]] | None = None,
        notes: list[str] | None = None,
        on_trace: TraceFn | None = None,
        ) -> Judgment:
    """Judges one anchor with the given variant. Escalating: one-shot first, and
    an agentic run only if the one-shot verdict is not accept. on_trace receives
    the messages of every run (two for an escalation)."""
    def trace_of(mode: JudgeMode) -> Callable[[list[dict]], None] | None:
        if on_trace is None:
            return None
        return lambda messages: on_trace(verify_input, mode, messages)

    first_mode = JudgeMode.AGENTIC if variant.mode == JudgeMode.AGENTIC else JudgeMode.ONE_SHOT
    verdict, steps, elapsed, error = run_verify(
        verify_input, variant_config(base_config, variant, first_mode), managers, kg_notes, notes,
        trace_of(first_mode),
    )
    first_verdict = None
    if variant.mode == JudgeMode.ESCALATING and (verdict is None or verdict.verdict != VerdictOption.ACCEPT):
        first_verdict = verdict.verdict if verdict else VerdictOption.UNSURE
        verdict, more_steps, more_elapsed, error = run_verify(
            verify_input, variant_config(base_config, variant, JudgeMode.AGENTIC), managers, kg_notes, notes,
            trace_of(JudgeMode.AGENTIC),
        )
        steps += more_steps
        elapsed += more_elapsed

    if verdict is None:
        # failed run: unsure, with the error kept for the evaluation
        verdict = Verdict(
            **verify_input.anchor.model_dump(), verdict=VerdictOption.UNSURE, rationale="",
        )
    return Judgment(
        **verdict.model_dump(),
        variant=variant.name,
        model=base_config.model,
        steps=steps,
        elapsed=elapsed,
        error=error,
        first_verdict=first_verdict,
    )


# ---------------------------------------------------------------- certification

def certify_stratum(
        stratum: str,
        anchors: list[AnchorRecord],
        source_kg: str,
        target_kg: str,
        judge_fn: JudgeFn,
        params: CertificationParams,
        on_judgment: Callable[[Judgment], None] | None = None,
        ) -> tuple[StratumDecision, list[Judgment]]:
    """Judges a sample of the stratum in a fixed random order and stops at the
    (c+1)-th reject. Without a plan (small strata) every anchor is judged."""
    plan = sampling_plan(len(anchors), params)
    # the order depends on the seed and the stratum, not on other strata
    order = sorted(anchors, key=lambda a: (a.source_iri, a.target_iri))
    random.Random(f"{params.seed}:{stratum}").shuffle(order)

    judgments: list[Judgment] = []
    rejects = unsure = counted = 0
    n, c = plan if plan is not None else (len(order), None)
    candidates: Iterator[AnchorRecord] = iter(order)
    while counted < n:
        anchor = next(candidates, None)
        if anchor is None:
            break  # only possible if unsure verdicts are not counted
        judgment = judge_fn(VerifyInput(source_kg=source_kg, target_kg=target_kg, anchor=anchor))
        judgments.append(judgment)
        if on_judgment is not None:
            on_judgment(judgment)
        if judgment.verdict == VerdictOption.UNSURE:
            unsure += 1
            if not params.unsure_as_reject:
                continue  # not counted: the next anchor of the order replaces it
            rejects += 1
        elif judgment.verdict == VerdictOption.REJECT:
            rejects += 1
        counted += 1
        if c is not None and rejects > c:
            break

    if plan is None:
        decision = StratumDecisionType.INSPECTED
    elif rejects > plan[1] or counted < n:
        decision = StratumDecisionType.REJECTED
    else:
        decision = StratumDecisionType.ACCEPTED
    return StratumDecision(
        stratum=stratum,
        size=len(anchors),
        plan=plan,
        decision=decision,
        judged=len(judgments),
        rejects=rejects,
        unsure=unsure,
    ), judgments


def certify(
        result: AnchorStageResult,
        judge_fn: JudgeFn,
        variant: JudgeVariant,
        params: CertificationParams,
        on_judgment: Callable[[Judgment], None] | None = None,
        ) -> CertificationResult:
    """Certifies every stratum (Venn cell) of an anchor stage result."""
    by_stratum: dict[str, list[AnchorRecord]] = {}
    for anchor in result.anchors:
        by_stratum.setdefault(anchor.stratum, []).append(anchor)

    decisions: list[StratumDecision] = []
    judgments: list[Judgment] = []
    certified: list[AnchorRecord] = []
    for stratum, anchors in by_stratum.items():
        decision, stratum_judgments = certify_stratum(
            stratum, anchors, result.source_kg, result.target_kg, judge_fn, params, on_judgment,
        )
        decisions.append(decision)
        judgments.extend(stratum_judgments)
        if decision.decision == StratumDecisionType.ACCEPTED:
            certified.extend(anchors)
        else:
            # rejected or inspected stratum: only the individually accepted anchors
            accepted = {
                (j.source_iri, j.target_iri) for j in stratum_judgments
                if j.verdict == VerdictOption.ACCEPT
            }
            certified.extend(a for a in anchors if (a.source_iri, a.target_iri) in accepted)

    return CertificationResult(
        source_kg=result.source_kg,
        target_kg=result.target_kg,
        variant=variant.name,
        params=params,
        strata=decisions,
        judgments=judgments,
        certified=certified,
    )
