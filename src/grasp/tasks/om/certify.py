"""Certification of the anchor strata by sampling: the bookkeeping around the
om-verify task, which judges a single anchor per run.

- judge: one anchor, one judge variant (A1-A3, optionally with competitors)
- certify_stratum: sampling plan, sample, sequential judging, decision
- certify: all strata of an anchor stage result

GRASP is used as a library: the managers are loaded once by the caller, and
every judgment is a fresh conversation (independent verdicts). The agentic
mode runs through grasp.core.generate (the GRASP loop with KG functions); the
one-shot mode is a single model call outside of it, since it uses neither the
loop nor the KG functions: its prompt is what the om-verify task provides plus
a short description and the structural notes of the two KGs (no GRASP
instructions about indices, endpoints, prefixes or rules).
"""
import math
import random
import time
import traceback
from typing import Callable, Iterator

from scipy.stats import hypergeom
from universal_ml_utils.ops import consume_generator

from grasp.configs import GraspConfig
from grasp.core import generate
from grasp.manager import KgManager
from grasp.functions import find_manager
from grasp.model import Message, get_model
from grasp.tasks import get_task
from grasp.utils import format_enumerate, format_section
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
        # a single model call (run_one_shot) that must call a verdict function
        update |= {"tool_choice": "required"}
    else:
        update |= {"max_steps": variant.max_steps}
        if base.budget_warning_fraction is None:
            # warn during the last 30% of the steps; the last step must give a verdict
            update["budget_warning_fraction"] = 0.3
    return base.model_copy(update=update, deep=True)


# result of one om-verify run: (verdict or None, LLM responses, seconds, error)
type RunResult = tuple[Verdict | None, int, float, str | None]


def format_judged_kgs(
        verify_input: VerifyInput,
        managers: list[KgManager],
        kg_notes: dict[str, list[str]] | None,
        ) -> str:
    """Source and target KG for the one-shot prompt: name, description and the
    KG's notes (structural notes), without endpoint, indices or prefixes."""
    parts = []
    for role, kg in (("source", verify_input.source_kg), ("target", verify_input.target_kg)):
        manager, _ = find_manager(managers, kg)
        part = f"### {kg} ({role})"
        if manager.description:
            part += f"\n{manager.description}"
        notes = (kg_notes or {}).get(kg)
        if notes:
            part += "\n\n**Notes**\n" + format_enumerate(notes)
        parts.append(part)
    return format_section("Knowledge graphs", "\n\n".join(parts))


def run_one_shot(
        verify_input: VerifyInput,
        config: GraspConfig,
        managers: list[KgManager],
        kg_notes: dict[str, list[str]] | None = None,
        on_trace: Callable[[list[dict]], None] | None = None,
        ) -> RunResult:
    """One-shot om-verify run without the GRASP loop: the task's system
    information, the two KGs with their structural notes, the task's input, its
    verdict functions, one model call; the called verdict function is executed
    like in generate."""
    start = time.monotonic()
    task = get_task(VERIFY_TASK, managers, config)
    system = task.system_information() + "\n\n" + format_judged_kgs(verify_input, managers, kg_notes)
    messages = [
        Message.system(system),
        Message.user(task.setup(verify_input.model_dump(mode="json"))),
    ]
    error = None
    try:
        response = get_model(config)(messages, task.function_definitions())
    except Exception as e:
        response = None
        error = f"Failed to generate response:\n{e}\n\n{traceback.format_exc()}"
    if response is not None:
        messages.append(Message.assistant(response))
        if not response.tool_calls:
            error = "no verdict function called"
        for tool_call in response.tool_calls:
            try:
                tool_call.result = task.call_function(tool_call.name, tool_call.args, task.known, None)
            except Exception as e:
                tool_call.error = str(e)
                tool_call.result = f"Call to function {tool_call.name} returned an error:\n{e}"
                error = tool_call.result
    if on_trace is not None:
        on_trace([message.model_dump() for message in messages])

    elapsed = time.monotonic() - start
    output = task.output(messages)
    if output is None:
        return None, 1, elapsed, error or "no verdict submitted"
    return Verdict.model_validate(output), 1, elapsed, None


def run_agentic(
        verify_input: VerifyInput,
        config: GraspConfig,
        managers: list[KgManager],
        kg_notes: dict[str, list[str]] | None = None,
        notes: list[str] | None = None,
        on_trace: Callable[[list[dict]], None] | None = None,
        ) -> RunResult:
    """Agentic om-verify run in the GRASP loop (generate)."""
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

    def run(mode: JudgeMode) -> RunResult:
        config = variant_config(base_config, variant, mode)
        if mode == JudgeMode.ONE_SHOT:
            return run_one_shot(verify_input, config, managers, kg_notes, trace_of(mode))
        return run_agentic(verify_input, config, managers, kg_notes, notes, trace_of(mode))

    first_mode = JudgeMode.AGENTIC if variant.mode == JudgeMode.AGENTIC else JudgeMode.ONE_SHOT
    verdict, steps, elapsed, error = run(first_mode)
    first_verdict = None
    if variant.mode == JudgeMode.ESCALATING and (verdict is None or verdict.verdict != VerdictOption.ACCEPT):
        first_verdict = verdict.verdict if verdict else VerdictOption.UNSURE
        verdict, more_steps, more_elapsed, error = run(JudgeMode.AGENTIC)
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
