"""
Certification of strata of pairs by sampling: the method around the om-verify
task, which judges one pair per run.

- sampling_plan and the stop rules: when a stratum is accepted or rejected,
  planned on the reject rates an imperfect judge sees
- run_one_shot: the one-shot judge, a single model call outside the GRASP loop
- certify_stratum: fixed random order, counting, stop rule, certified pairs
- certify: all strata of an alignment state
- recall estimation: strata of the unmatched source entities, sample
  allocation and drawing, the estimate and the stop criterion

The judge itself (one-shot, agentic, escalating), its logs and the resumption
are up to the caller: certify only gets a judge function per stratum. Every
judgment is a fresh conversation (independent verdicts). The one-shot judge
uses neither the loop nor the KG functions: its prompt is what the om-verify
task provides plus a short description and the notes of the two KGs (no GRASP
instructions about indices, endpoints, prefixes or rules).
"""
import math
import random
import time
from typing import Any, Callable

from scipy.stats import hypergeom

from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.model import Message, get_model
from grasp.tasks import get_task
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.candidates import Stratum
from grasp.tasks.om.data_models.verification import (
    RECALL_STRATA,
    CandidatesOf,
    CertificationParams,
    CertificationResult,
    JudgeVariant,
    Judgment,
    RecallEstimate,
    RecallParams,
    RecallStratumEstimate,
    StopDecision,
    StopRule,
    StratumCertification,
    StratumDecision,
    StratumDecisionType,
    VerdictOption,
    VerifyInput,
)
from grasp.tasks.om.data_models.profile import EntityMemberships, KgProfile
from grasp.tasks.om.foundation.types import IRI, Side
from grasp.tasks.om.kg_operations.candidate_generation import sample_position
from grasp.tasks.om.kg_operations.grouping import (
    groups_to_grouping,
    profile_partitions_to_grouping,
    retrieve_grouping_by_entity,
    )
from grasp.tasks.om.kg_operations.candidate_generation import retrieve_linked_entities
from grasp.tasks.om.tasks.task_input import to_task_input
from grasp.utils import format_enumerate, format_section

VERIFY_TASK = "om-verify"
NUM_COMPETITORS_PER_SIDE = 4

# judges one pair; blind: it gets nothing but the pair and its context (no
# stratum, no evidence of the strategies)
type JudgeFunction = Callable[[VerifyInput], Judgment]
# the judge function to use for the stratum of the given name (e.g. one that
# logs its judgments with the stratum)
type JudgeForStratum = Callable[[str], JudgeFunction]
type BuildVerifyInput = Callable[[tuple[IRI, IRI]], VerifyInput]
# the stop rule of a stratum of the given size
type StopRuleForSize = Callable[[int], StopRule]


class JudgeUnavailableError(RuntimeError):
    """Too many judge errors in a row: most likely the model API is down."""


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


# ---------------------------------------------------------------- stop rules

def apparent_reject_rate(error_rate: float, params: CertificationParams) -> float:
    """
    The share of reject verdicts in a stratum with this share of wrong pairs,
    as an imperfect judge sees it: wrong pairs it does not accept (1 - p_fp)
    and correct pairs it rejects (p_fn).
    """
    return error_rate * (1 - params.p_fp) + (1 - error_rate) * params.p_fn


def plan_on_apparent_rates(params: CertificationParams) -> CertificationParams:
    """The parameters with the rates of a good and a bad stratum as the judge sees them."""
    return params.model_copy(update={
        "p_good": apparent_reject_rate(params.p_good, params),
        "p_max": apparent_reject_rate(params.p_max, params),
        })


def inspect_all(num_counted: int, num_rejects: int) -> StopDecision:
    """Stop rule of a stratum without a plan: every pair is judged."""
    return StopDecision.CONTINUE


def fixed_stop_rule(size: int, params: CertificationParams) -> StopRule:
    """
    The two-point plan (n, c) of the stratum on the apparent rates: reject at
    the (c+1)-th reject, accept after n counted verdicts; without a plan
    (small strata) every pair is judged.
    """
    plan = sampling_plan(size, plan_on_apparent_rates(params))
    if plan is None:
        return inspect_all
    sample_size, acceptance_number = plan

    def decide(num_counted: int, num_rejects: int) -> StopDecision:
        if num_rejects > acceptance_number:
            return StopDecision.REJECT_STRATUM
        if num_counted >= sample_size:
            return StopDecision.ACCEPT_STRATUM
        return StopDecision.CONTINUE

    return decide


def sequential_stop_rule(params: CertificationParams) -> StopRule:
    """
    Wald's sequential probability ratio test on the apparent rates: each
    reject adds log(q_max / q_good), each other counted verdict
    log((1 - q_max) / (1 - q_good)); the stratum is rejected at the upper
    bound log((1 - beta) / alpha), accepted at the lower bound
    log(beta / (1 - alpha)). Without a decision until the end of the stratum,
    every pair has been judged.
    """
    apparent = plan_on_apparent_rates(params)
    reject_step = math.log(apparent.p_max / apparent.p_good)
    accept_step = math.log((1 - apparent.p_max) / (1 - apparent.p_good))
    upper_bound = math.log((1 - params.beta) / params.alpha)
    lower_bound = math.log(params.beta / (1 - params.alpha))

    def decide(num_counted: int, num_rejects: int) -> StopDecision:
        log_likelihood_ratio = (
            num_rejects * reject_step + (num_counted - num_rejects) * accept_step
            )
        if log_likelihood_ratio >= upper_bound:
            return StopDecision.REJECT_STRATUM
        if log_likelihood_ratio <= lower_bound:
            return StopDecision.ACCEPT_STRATUM
        return StopDecision.CONTINUE

    return decide


def derive_stop_rule(params: CertificationParams) -> StopRuleForSize:
    """The stop rule of a stratum of the given size, by params.stop_rule."""
    match params.stop_rule:
        case "fixed":
            return lambda size: fixed_stop_rule(size, params)
        case "sequential":
            return lambda size: sequential_stop_rule(params)


def expected_judgments(size: int, params: CertificationParams) -> float:
    """
    The number of counted verdicts a stratum of this size is expected to need
    if it is good: n of the fixed plan, or Wald's average sample number of the
    sequential test at q_good; at most the size.
    """
    match params.stop_rule:
        case "fixed":
            plan = sampling_plan(size, plan_on_apparent_rates(params))
            return float(size if plan is None else plan[0])
        case "sequential":
            apparent = plan_on_apparent_rates(params)
            upper_bound = math.log((1 - params.beta) / params.alpha)
            lower_bound = math.log(params.beta / (1 - params.alpha))
            drift = (
                apparent.p_good * math.log(apparent.p_max / apparent.p_good)
                + (1 - apparent.p_good) * math.log((1 - apparent.p_max) / (1 - apparent.p_good))
                )
            average_sample_number = (
                (1 - params.alpha) * lower_bound + params.alpha * upper_bound
                ) / drift
            return min(float(size), average_sample_number)


# ---------------------------------------------------------------- judging

def format_judged_kgs(
        verify_input: VerifyInput,
        managers: list[KgManager],
        kg_notes: dict[str, list[str]] | None,
        ) -> str:
    """
    Source and target KG for the one-shot prompt: name, description and the
    KG's notes (structural notes), without endpoint, indices or prefixes.
    """
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
        kg_managers: list[KgManager],
        kg_notes: dict[str, list[str]] | None = None
        ) -> dict[str, Any]:
    """
    The one-shot judge: the om-verify task's system information with the two
    KGs, its input and its verdict functions, one model call; the called
    verdict function is executed like in generate. Returns a dict in the
    layout of the result of generate (errors in "error", not raised).
    """
    start = time.monotonic()
    # the model must call a verdict function, there is no second step
    config = config.model_copy(update={"tool_choice": "required"})
    task = get_task(VERIFY_TASK, kg_managers, config)
    # setup first: the system information depends on the input (competitors, mode)
    user_message = task.setup(to_task_input(verify_input))
    system = (
        task.system_information() + "\n\n"
        + format_judged_kgs(verify_input, kg_managers, kg_notes)
        )
    messages = [Message.system(system), Message.user(user_message)]

    error = None
    try:
        response = get_model(config)(messages, task.function_definitions())
    except Exception as e:
        response = None
        error = f"Failed to generate response: {e}"

    if response is not None:
        messages.append(Message.assistant(response))
        if not response.tool_calls:
            error = "no verdict function called"
        for tool_call in response.tool_calls:
            try:
                tool_call.result = task.call_function(tool_call.name, tool_call.args, task.known, None)
            except Exception as e:
                # e.g. a mapping with an entity that was not presented
                tool_call.error = str(e)
                tool_call.result = f"Call to function {tool_call.name} returned an error:\n{e}"
                error = tool_call.result

    return {
        "type": "output",
        "task": VERIFY_TASK,
        "output": task.output(messages),
        "elapsed": time.monotonic() - start,
        "error": error,
        "messages": [message.model_dump() for message in messages],
        "known": list(task.known),
        }


# ---------------------------------------------------------------- certification

def certify_stratum(
        stratum: Stratum,
        judge_function: JudgeFunction,
        build_verify_input: BuildVerifyInput,
        stop_rule_for_size: StopRuleForSize,
        unsure_as_reject: bool = False,
        num_max_errors_in_a_row: int = 3,
        seed: int = 0
        ) -> StratumCertification:
    """
    Judges the pairs of the stratum in a fixed random order (any prefix of it
    is a random sample) until the stop rule decides or all pairs are judged.
    Unsure verdicts count as rejects or are skipped (the next pair replaces
    them); errors are skipped, too many in a row stop the certification.
    """
    stop_rule = stop_rule_for_size(len(stratum.pairs))

    # the order depends on the seed and the stratum, not on other strata
    pairs = sorted(stratum.pairs)
    random.Random(f"{seed}:{stratum.name}").shuffle(pairs)

    num_counted = num_rejects = num_errors_in_a_row = 0
    accepted_pairs: set[tuple[IRI, IRI]] = set()
    judged_pairs: set[tuple[IRI, IRI]] = set()  # only with a verdict other than error
    judgments: list[Judgment] = []
    stop_decision = StopDecision.CONTINUE

    for pair in pairs:
        judgment = judge_function(build_verify_input(pair))
        judgments.append(judgment)

        match judgment.verdict:
            case VerdictOption.ACCEPT:
                accepted_pairs.add(pair)
                num_counted += 1
            case VerdictOption.REJECT:
                num_counted += 1
                num_rejects += 1
            case VerdictOption.UNSURE:
                if unsure_as_reject:
                    num_counted += 1
                    num_rejects += 1
            case VerdictOption.ERROR:
                num_errors_in_a_row += 1
                if num_errors_in_a_row >= num_max_errors_in_a_row:
                    # most likely the model API is down: stop everything (a
                    # judge function with a log allows to resume later)
                    raise JudgeUnavailableError(
                        f"{num_errors_in_a_row} judge errors in a row in stratum "
                        f"{stratum.name}, last: {judgment.error}"
                        )
                continue  # not counted: the stop rule gets nothing new
        num_errors_in_a_row = 0
        judged_pairs.add(pair)

        stop_decision = stop_rule(num_counted, num_rejects)
        if stop_decision != StopDecision.CONTINUE:
            break

    match stop_decision:
        case StopDecision.ACCEPT_STRATUM:
            decision_type = StratumDecisionType.ACCEPTED
            # all pairs except those judged reject or unsure
            not_accepted_pairs = judged_pairs - accepted_pairs
            certified_pairs = [pair for pair in pairs if pair not in not_accepted_pairs]
        case StopDecision.REJECT_STRATUM:
            decision_type = StratumDecisionType.REJECTED
            certified_pairs = [pair for pair in pairs if pair in accepted_pairs]
        case StopDecision.CONTINUE:
            # the order ended before a decision: every pair was judged
            decision_type = StratumDecisionType.INSPECTED
            certified_pairs = [pair for pair in pairs if pair in accepted_pairs]

    return StratumCertification(
        decision=StratumDecision(
            stratum=stratum.name,
            size=len(stratum.pairs),
            decision=decision_type,
            counted=num_counted,
            rejects=num_rejects
            ),
        judgments=judgments,
        certified_pairs=certified_pairs
        )


def certify_strata(
        strata: list[Stratum],
        certification_params: CertificationParams,
        judge_for_stratum: JudgeForStratum,
        build_verify_input: BuildVerifyInput,
        stop_rule_for_size: StopRuleForSize,
        num_max_errors_in_a_row: int = 3,
        ) -> list[StratumCertification]:
    """
    The strata certified one after the other, largest first (independent:
    each has its own order); the order only makes logs easier to read.
    """
    strata = sorted(strata, key=lambda stratum: -len(stratum.pairs))
    return [
        certify_stratum(
            stratum,
            judge_for_stratum(stratum.name),
            build_verify_input,
            stop_rule_for_size,
            certification_params.unsure_as_reject,
            num_max_errors_in_a_row,
            certification_params.seed
            )
        for stratum in strata
        ]


# ---------------------------------------------------------------- judge input

def choose_competitors(
        candidates_of: CandidatesOf,
        pair: tuple[IRI, IRI],
        max_per_side: int
        ) -> dict[Side, list[IRI]]:
    """
    The best-ranked rivals of the pair on each side, from all candidates the
    strategies found so far. The rivals of the source entity are source
    entities, so they are the candidates of the target entity, and vice
    versa; the partner of the pair is left out.
    """
    source_entity, target_entity = pair
    source_rivals = [
        candidate for candidate in candidates_of(Side.TARGET, target_entity)
        if candidate != source_entity
        ]
    target_rivals = [
        candidate for candidate in candidates_of(Side.SOURCE, source_entity)
        if candidate != target_entity
        ]
    return {
        Side.SOURCE: source_rivals[:max_per_side],
        Side.TARGET: target_rivals[:max_per_side]
        }


def verify_input_builder(
        alignment_state: AlignmentState,
        candidates_of: CandidatesOf | None,
        entity_memberships: EntityMemberships | None,
        num_competitors_per_side: int = NUM_COMPETITORS_PER_SIDE
        ) -> BuildVerifyInput:
    """
    Builds the judge's input of a pair: the two KGs with their profiles, the
    entity memberships (the groups of the shown entities) and, with
    candidates_of (all candidates found so far per entity, best first), the
    rivals of the pair. Without it, no competitors.
    """
    def build_verify_input(pair: tuple[IRI, IRI]) -> VerifyInput:
        return VerifyInput(
            source_kg=alignment_state.source_kg_name,
            target_kg=alignment_state.target_kg_name,
            source_profile=alignment_state.get_kg_profile(Side.SOURCE),
            target_profile=alignment_state.get_kg_profile(Side.TARGET),
            source_iri=pair[0],
            target_iri=pair[1],
            competitors=(
                choose_competitors(candidates_of, pair, num_competitors_per_side)
                if candidates_of is not None else {}
                ),
            entity_memberships=entity_memberships,
            )

    return build_verify_input


def certify(
        alignment_state: AlignmentState,
        strata: list[Stratum],
        candidates_of: CandidatesOf,
        judge_variant: JudgeVariant,
        judge_for_stratum: JudgeForStratum,
        certification_params: CertificationParams,
        entity_memberships: EntityMemberships | None,
        ) -> CertificationResult:
    """
    Certifies every stratum of pairs of the alignment state's two KGs.
    judge_for_stratum: the judge function per stratum, built by the caller
    with judge_variant (which here only selects the competitors and names the
    result). candidates_of: all candidates found so far per entity, best
    first; used only by variants with competitors. entity_memberships: the
    groups of the shown entities for the judges (None: they query them).
    """
    build_verify_input = verify_input_builder(
        alignment_state,
        candidates_of if judge_variant.competitors else None,
        entity_memberships,
        )
    certified_strata = certify_strata(
        strata,
        certification_params,
        judge_for_stratum,
        build_verify_input,
        derive_stop_rule(certification_params)
        )
    return CertificationResult(
        source_kg=alignment_state.source_kg_name,
        target_kg=alignment_state.target_kg_name,
        variant=judge_variant.name,
        params=certification_params,
        strata=certified_strata
        )


# ---------------------------------------------------------------- recall estimation

def retrieve_recall_strata(
        manager: KgManager,
        kg_profile: KgProfile,
        matched_entities: set[IRI]
        ) -> dict[str, frozenset[IRI]]:
    """
    The unmatched entities of the KG (all partitions, excluded groups
    included) in the recall strata, disjoint, in the order of RECALL_STRATA:
    neighbours of matched entities, members of excluded groups, the rest.
    """
    all_partitions = profile_partitions_to_grouping(kg_profile)
    unmatched = set(retrieve_grouping_by_entity(manager, all_partitions, kg_profile.entity_space)) - matched_entities
    neighbours = retrieve_linked_entities(
        manager, all_partitions, kg_profile.entity_space, matched_entities
        ) & unmatched
    excluded_groups = [group for group in kg_profile.groups if group.excluded]
    excluded = (
        set(retrieve_grouping_by_entity(manager, groups_to_grouping(excluded_groups), kg_profile.entity_space))
        if excluded_groups else set()
        ) & unmatched
    excluded -= neighbours
    rest = unmatched - neighbours - excluded
    return {
        stratum: frozenset(entities)
        for stratum, entities in zip(RECALL_STRATA, (neighbours, excluded, rest), strict=True)
        }


def allocate_recall_sample(
        stratum_sizes: dict[str, int],
        params: RecallParams,
        previous: RecallEstimate | None = None
        ) -> dict[str, int]:
    """
    The sample size per stratum: in the first round by the fixed shares,
    later by Neyman allocation (size * sqrt(p (1 - p)), p the share found in
    the round before, at least 1 / (judged + 1)); at least min_per_stratum,
    at most the stratum size.
    """
    if previous is None:
        weights = {stratum: params.first_round_shares.get(stratum, 0.0) for stratum in stratum_sizes}
    else:
        found_share = {
            estimate.stratum: max(estimate.found, 1) / (estimate.judged + 1) for estimate in previous.strata
            }
        weights = {
            stratum: size * math.sqrt(found_share.get(stratum, 0.5) * (1 - found_share.get(stratum, 0.5)))
            for stratum, size in stratum_sizes.items()
            }
    total_weight = sum(weight for stratum, weight in weights.items() if stratum_sizes[stratum]) or 1.0
    return {
        stratum: min(size, max(params.min_per_stratum, round(params.sample_size * weights[stratum] / total_weight)))
        if size else 0
        for stratum, size in stratum_sizes.items()
        }


def draw_consistent_sample(entities: frozenset[IRI], size: int, seed: int) -> list[IRI]:
    """
    The size entities with the smallest sample positions: the same in every
    run, and nested over the rounds (an entity drawn before and still in the
    stratum is drawn again).
    """
    return sorted(entities, key=lambda entity: (sample_position(entity, seed), entity))[:size]


def missed_upper_bound(found: int, judged: int, size: int, confidence: float) -> int:
    """
    The exact one-sided upper confidence bound of the number of entities with
    a missed partner in a stratum of this size, from a sample without
    replacement (hypergeometric): the largest number K for which finding at
    most `found` in `judged` drawn entities still has a probability above
    1 - confidence. A stratum judged completely gives `found` exactly; a
    stratum not judged gives its size.
    """
    if judged == 0:
        return size
    significance = 1 - confidence
    smallest, largest = found, found + (size - judged)  # K lies between them
    # P(at most `found` | K) falls with K: bisection for the last K above the significance
    while smallest < largest:
        middle = (smallest + largest + 1) // 2
        if hypergeom.cdf(found, size, middle, judged) > significance:
            smallest = middle
        else:
            largest = middle - 1
    return smallest


def estimate_recall(
        round_number: int,
        accepted: int,
        stratum_sizes: dict[str, int],
        judged_by_stratum: dict[str, int],
        found_by_stratum: dict[str, int],
        params: RecallParams
        ) -> RecallEstimate:
    """
    The stratified estimate: missed = sum of size * found / judged per
    stratum; its upper bound from the strata's exact (hypergeometric) upper
    bounds, each at the Bonferroni level (1 - confidence) / strata, so that
    all hold together with the confidence (conservative). A stratum not
    judged counts wholly, one judged completely exactly.
    """
    strata = [stratum for stratum, size in stratum_sizes.items() if size]
    stratum_confidence = 1 - (1 - params.confidence) / max(len(strata), 1)
    estimates = []
    for stratum in strata:
        size, judged, found = stratum_sizes[stratum], judged_by_stratum.get(stratum, 0), found_by_stratum.get(stratum, 0)
        estimates.append(RecallStratumEstimate(
            stratum=stratum, size=size, judged=judged, found=found,
            estimated_missed=size * found / judged if judged else float(size),
            missed_upper=missed_upper_bound(found, judged, size, stratum_confidence),
            ))
    estimated_missed = sum(estimate.estimated_missed for estimate in estimates)
    missed_upper = sum(estimate.missed_upper for estimate in estimates)

    def recall_of(missed: float) -> float:
        return accepted / (accepted + missed) if accepted + missed else 1.0

    return RecallEstimate(
        round=round_number, accepted=accepted, strata=estimates,
        estimated_missed=estimated_missed, missed_upper=missed_upper,
        recall=recall_of(estimated_missed), recall_lower=recall_of(missed_upper),
        confidence=params.confidence,
        )


def stop_reason(estimates: list[RecallEstimate], params: RecallParams) -> str | None:
    """
    Why to stop after the last estimate, or None to go on: the lower bound
    of the recall reached the target, the estimated missed equivalences fell
    by less than min_relative_improvement since the round before, or
    max_rounds rounds are done.
    """
    if not estimates:
        return None
    last = estimates[-1]
    if last.recall_lower >= params.target_recall:
        return f"recall >= {last.recall_lower:.3f} (at {params.confidence:.0%}), target {params.target_recall}"
    if len(estimates) >= 2:
        before = estimates[-2].estimated_missed
        if before > 0 and (before - last.estimated_missed) / before < params.min_relative_improvement:
            return (f"estimated missed equivalences fell from {before:.0f} to {last.estimated_missed:.0f}, "
                    f"less than {params.min_relative_improvement:.0%}")
    if last.round >= params.max_rounds:
        return f"{params.max_rounds} rounds done"
    return None
