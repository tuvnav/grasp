"""
The anchor stage (om-anchors): the LLM tries key strategies and selects those
whose anchors are run in full later and certified. An anchor is a pair that a
key strategy finds unique: the two entities share a key that no other
competing entity of either KG shares on that or a finer normalization level
(the anchor rule, Selection.unique). Only key strategies: names, codes and
cross-references in a shared code system give anchors; similarity measures
(vectors, embeddings) give candidates, not anchors (their mutual nearest
neighbours are often related, not equivalent).

Built on the candidate stage (candidate_task.py): the same generators (with a
fixed selection), runs, state and full runs; only the functions for the LLM,
their texts and the merge of the runs are the stage's own. The anchors of an
ensemble are merged (kg_operations/candidate_selection.py: merge_anchors):
pairs whose entity has another partner in another run are conflicts; the rest
form the certification strata, by the run families that found them, head or
specific (the words of a name extended by another name of a KG) and extended
(unique only on a coarse normalization level).
"""
import random
from enum import StrEnum, auto
from typing import Annotated, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, WithJsonSchema

from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.om.data_models.alignment_state import AlignmentState
from grasp.tasks.om.data_models.profile import EntityMemberships
from grasp.tasks.om.data_models.candidates import (
    Anchor,
    AnchorMerge,
    AnchorSpecificity,
    AnchorStageResult,
    CandidateRankIndex,
    CandidateStrategyKind,
    Generator,
    KeyStrategy,
    PairSignalIndex,
    Selection,
    RunEnsemble,
    StrategyResult,
    Stratum,
    TriedRun,
    StringNormalization,
    GroupRef,
    POPULATION_REST,
    Population,
    ValuePatternSource,
    ValuePatterns,
)
from grasp.tasks.om.foundation.configuration import load_task_instructions
from grasp.tasks.om.foundation.text_normalization import NORMALIZATION_LEVELS_DESCRIPTION, STANDARD_LEVELS, NormalizationLevel
from grasp.tasks.om.foundation.types import IRI, SIDES, Side
from grasp.tasks.om.kg_operations.candidate_selection import (
    build_candidate_rank_index,
    build_pair_signal_index,
    get_anchor_strata,
    merge_anchors,
)
from grasp.tasks.om.kg_operations.entity_rendering import EntityDisplay
from grasp.tasks.om.tasks.candidate_task import (
    GENERATOR_SCHEMA,
    MAX_ENTITY_PAIRS_PER_BUCKET,
    CandidateTaskState,
    format_found_again,
    format_normalization_examples,
    format_hubs,
    format_share,
    format_baseline_runs,
    format_start_message,
    parse_run_ensemble,
    prefetch_shown_entities,
    reselect_result,
    run_strategy,
)
from grasp.tasks.om.foundation.rendering import Item, Section, SectionContent, format_display_content
from grasp.tasks.om.tasks.prompting import (
    BLANK_NODE_RULE,
    add_equivalence_condition,
    fill_function_name_placeholders,
    format_validation_error,
    inline_schema,
    load_prompt,
)
from grasp.tasks.om.tasks.task_input import (
    managers_by_side, resolve_alignment_state, resolve_entity_memberships, resolve_strategy_history,
)
from grasp.utils import FunctionCallException, format_section

# the selection of every anchor run: the unique candidate only (the anchor rule)
ANCHOR_SELECTION = Selection(top_k=1, unique=True)
# the selection for the competitors that certification may show with an anchor
COMPETITOR_SELECTION = Selection(top_k=5)

NUM_EXAMPLE_ANCHORS = 5      # anchors shown per kind in a result
NUM_AMBIGUOUS_EXAMPLES = 3   # entities with a key match but no anchor shown in a result
NUM_SHOWN_COUNTERPARTS = 3   # tied counterparts shown per ambiguous entity
NUM_SHOWN_RIVALS = 3  # rivals (own side, same best counterpart) shown per ambiguous entity
MAX_SHOWN_ANCHORS = 20
NUM_CONFLICT_EXAMPLES = 5


class AnchorFunctionName(StrEnum):
    TRY_ANCHOR_STRATEGY = auto()
    SHOW_ANCHORS = auto()
    EVALUATE_ANCHOR_ENSEMBLE = auto()
    FINISH = auto()


class AnchorListKind(StrEnum):
    RANDOM = auto()
    HEAD = auto()        # head anchors, lowest information content first
    EXTENDED = auto()    # unique only on a coarse normalization level
    AMBIGUOUS = auto()   # entities with a key match but no anchor


# ---------------------------------------------------------------- runs

def anchor_generator(fn_args: dict) -> Generator:
    """
    A key generator with the anchor selection from the
    arguments of try_anchor_strategy.
    """
    try:
        arguments = AnchorStrategyArguments.model_validate(fn_args)
        values = ValuePatterns(
            source=arguments.source_values,
            target=arguments.target_values
        )
        return Generator(
            population=Population(
                source_group_refs=arguments.source_group_refs,
                target_group_refs=arguments.target_group_refs,
                ),
            strategy=KeyStrategy(
                values=values,
                normalization=StringNormalization(levels=arguments.normalization_levels),
                ),
            selection=ANCHOR_SELECTION,
            )
    except ValidationError as error:
        raise FunctionCallException(
            f"Invalid anchor strategy: {format_validation_error(error)}"
            ) from error


def run_anchors_of(state: CandidateTaskState, run_id: str) -> list[Anchor]:
    """
    The anchors of one tried run (a merge of the run alone: no conflicts).
    """
    return merge_anchors(RunEnsemble.model_validate(
        {"runs": [{"run_id": run_id}]}), state.runs
        ).anchors


def run_full_anchor_ensemble(
        managers: dict[Side, KgManager],
        alignment_state: AlignmentState,
        run_ensemble: RunEnsemble,
        generators_by_run: dict[str, Generator],
        entity_memberships: EntityMemberships,
        ) -> tuple[AnchorMerge, list[Stratum], CandidateRankIndex, PairSignalIndex]:
    """
    The full runs of the selected ensemble: the merged anchors, their
    certification strata, and the candidate rank index of the runs with a
    wider selection (COMPETITOR_SELECTION), for the candidate store.
    """
    runs = {
        run_id: (generators_by_run[run_id], run_strategy(
            managers, alignment_state, generators_by_run[run_id],
            entity_memberships,
            MAX_ENTITY_PAIRS_PER_BUCKET[CandidateStrategyKind.KEY],
            ))
        for run_id in run_ensemble.run_ids()
        }
    anchor_merge = merge_anchors(run_ensemble, runs)
    strategy_result_by_run = {
        run_id: reselect_result(
            result, COMPETITOR_SELECTION
            ) for run_id, (_, result) in runs.items()
        }
    competitor_rank_index = build_candidate_rank_index(
        run_ensemble,
        strategy_result_by_run
        )
    return (
        anchor_merge,
        get_anchor_strata(anchor_merge),
        competitor_rank_index,
        build_pair_signal_index(run_ensemble, strategy_result_by_run)
        )


# ---------------------------------------------------------------- formatting

def format_anchor(
        anchor: Anchor,
        entity_displays: dict[Side, EntityDisplay]
        ) -> str:
    text = (
        f"{entity_displays[Side.SOURCE].show_line(anchor.source_entity)} = "
        f"{entity_displays[Side.TARGET].show_line(anchor.target_entity)} "
        f"[on normalization level '{anchor.normalization_level}'"
        )
    if (anchor.specificity == AnchorSpecificity.HEAD
            and anchor.information_content is not None):
        text += f", specificity: 'head', information content {anchor.information_content:.2f}"
    return text + "]"


def prefetch_anchors(
        anchors: list[Anchor], entity_displays: dict[Side, EntityDisplay]
        ) -> None:
    entity_displays[Side.SOURCE].prefetch(anchor.source_entity for anchor in anchors)
    entity_displays[Side.TARGET].prefetch(anchor.target_entity for anchor in anchors)


def get_top_rank_rivals(result: StrategyResult, entity: IRI) -> dict[IRI, list[IRI]]:
    """
    Per shown best candidate of a source entity: the other source entities on
    the best rank of that candidate (a reason why the entity has no anchor).
    In a run on a sample, the lists of one side lack entities, so rivals may be missing.
    """
    rivals_by_candidate: dict[IRI, list[IRI]] = {}
    top_rank_candidates = result.collected[
        (Side.SOURCE, entity)
        ].get_top_rank_candidates()[:NUM_SHOWN_COUNTERPARTS]

    for candidate in top_rank_candidates:
        candidates_of_candidate = result.collected.get((Side.TARGET, candidate))
        rivals_by_candidate[candidate] = [
            rival for rival in (
                candidates_of_candidate.get_top_rank_candidates()
                if candidates_of_candidate else []
                )
            if rival != entity
        ]

    return rivals_by_candidate


def get_all_ambiguous_entities(result: StrategyResult) -> list[IRI]:
    """
    Source entities with a key match but no anchor, in a fixed order.
    """
    return sorted(
        entity for (side, entity), selected in result.selected.items()
        if side == Side.SOURCE
        and selected.collected.candidates
        and selected.unique_candidate is None
        )


def choose_ambiguous_entities(
        result: StrategyResult,
        num_entities: int,
        seed: int = 0
        ) -> list[IRI]:
    pool = get_all_ambiguous_entities(result)
    random.Random(seed).shuffle(pool)
    chosen: list[IRI] = []
    shown_rivals: set[IRI] = set()
    for entity in pool:
        if len(chosen) == num_entities:
            break
        if entity in shown_rivals:
            continue
        chosen.append(entity)
        shown_rivals.update(
            rival for rivals in get_top_rank_rivals(result, entity).values()
            for rival in rivals[:NUM_SHOWN_RIVALS]
            )
    return chosen


def format_ambiguous_entity(
        result: StrategyResult,
        entity: IRI,
        entity_displays: dict[Side, EntityDisplay]
        ) -> Item:
    """
    An entity without an anchor, with the counterparts on its best rank and
    its rivals (own side) on the best rank of these counterparts.
    """
    entity_candidates = result.collected[(Side.SOURCE, entity)]
    rivals_by_candidate = get_top_rank_rivals(result, entity)
    candidate_items = []

    for candidate, rivals in rivals_by_candidate.items():
        candidate_text = entity_displays[Side.TARGET].show_line(candidate)
        if rivals:
            num_cut_off_rivals = len(rivals) - NUM_SHOWN_RIVALS
            candidate_text += (
                ", rivals on its best rank: "
                + ", ".join(entity_displays[Side.SOURCE].show_line(rival)
                            for rival in rivals[:NUM_SHOWN_RIVALS])
                + (f", +{num_cut_off_rivals} more" if num_cut_off_rivals > 0 else "")
                )
        candidate_items.append(Item(candidate_text))

    num_cut_off_candidates = len(entity_candidates.get_top_rank_candidates()) - NUM_SHOWN_COUNTERPARTS
    if num_cut_off_candidates > 0:
        candidate_items.append(Item(f"+{num_cut_off_candidates} more"))

    children = [Item("candidates:", candidate_items)]
    counterparts_outside = entity_candidates.num_counterparts_outside_population_by_rank.get(
        entity_candidates.top_rank, 0
        )
    if counterparts_outside:
        children.append(
            Item(f"; {counterparts_outside} competing entities "
                 "outside the population share the key")
            )
    return Item(entity_displays[Side.SOURCE].show_line(entity), children)


def prefetch_ambiguous(
        result: StrategyResult,
        entities: list[IRI],
        entity_displays: dict[Side, EntityDisplay]
        ) -> None:
    entity_displays[Side.SOURCE].prefetch([
        *entities,
        *(
            rival for entity in entities
            for rivals in get_top_rank_rivals(result, entity).values()
            for rival in rivals[:NUM_SHOWN_RIVALS]
            )
        ])
    entity_displays[Side.TARGET].prefetch(
        candidate.candidate_entity for entity in entities
        for candidate in result.collected[(Side.SOURCE, entity)].candidates
        )


def anchors_of_kind(anchors: list[Anchor], kind: AnchorListKind, seed: int = 0) -> list[Anchor]:
    match kind:
        case AnchorListKind.HEAD:
            heads = [a for a in anchors if a.specificity == AnchorSpecificity.HEAD]
            return sorted(heads, key=lambda a: (a.information_content or 0.0, a.pair))
        case AnchorListKind.EXTENDED:
            return [a for a in anchors if a.extended]
        case _:
            shuffled = list(anchors)
            random.Random(seed).shuffle(shuffled)
            return shuffled


def format_anchor_result(
        run_id: str,
        result: StrategyResult,
        anchors: list[Anchor],
        entity_displays: dict[Side, EntityDisplay],
        ) -> Section:
    sizes = {side: result.signals[side].population_size for side in SIDES}
    with_match = {side: sum(1 for (s, _), selected in result.selected.items()
                            if s == side and selected.collected.candidates) for side in SIDES}
    specificity_counts = {specificity: sum(a.specificity == specificity for a in anchors)
                          for specificity in AnchorSpecificity}
    sampled_side = result.signals[Side.SOURCE].sampled_side
    random_examples = anchors_of_kind(anchors, AnchorListKind.RANDOM)[:NUM_EXAMPLE_ANCHORS]
    head_examples = anchors_of_kind(anchors, AnchorListKind.HEAD)[:NUM_EXAMPLE_ANCHORS]
    ambiguous = choose_ambiguous_entities(result, NUM_AMBIGUOUS_EXAMPLES)
    prefetch_anchors(random_examples + head_examples, entity_displays)
    prefetch_ambiguous(result, ambiguous, entity_displays)
    prefetch_shown_entities([], result.hubs, entity_displays)

    statistics_content = [
        f"Found {len(anchors)} anchors:",
        *(Item(
            f"""{side}: matched {
                format_share(len(anchors) / sizes[side]) if sizes[side] else '-'
                } of {sizes[side]} entities"""
            + (" (a sample)" if sampled_side == side else "")
            + f"; {with_match[side]} share a key with a competing entity, "
            f"{with_match[side] - len(anchors)} of them without an anchor (ambiguous)"
            ) for side in SIDES),
        Item(
            f"{specificity_counts[AnchorSpecificity.SPECIFIC]} names are specific, "
            f"{specificity_counts[AnchorSpecificity.HEAD]} names are extended in the ontologies ('head' names); "
            f"{specificity_counts[AnchorSpecificity.NOT_APPLICABLE]} other values (not names); "
            f"{sum(a.extended for a in anchors)} names are unique only on a coarse normalization level ('extended')"
            )
        ]

    found_again = format_found_again(result)
    if found_again:
        statistics_content.append(Item(found_again))

    statistics_subsection = Section("Statistics", statistics_content)

    subsections = [statistics_subsection]
    normalization_section = format_normalization_examples(result)
    if normalization_section:
        subsections.append(normalization_section)
    if result.hubs:
        hubs_content = format_hubs(result.hubs, entity_displays)
        if hubs_content:
            subsections.append(Section("Hubs", hubs_content))
    if random_examples:
        subsections.append(Section(
            "Example anchors",
            [Item(format_anchor(anchor, entity_displays)) for anchor in random_examples]
            ))
    if head_examples:
        subsections.append(Section(
            "Example head anchors with the lowest information content",
            [Item(format_anchor(anchor, entity_displays)) for anchor in head_examples]
            ))
    if ambiguous:
        subsections.append(Section(
            "Example entities without an anchor",
            [format_ambiguous_entity(result, entity, entity_displays) for entity in ambiguous]
            ))
    return Section(f"Result {run_id}", subsections=subsections)


def format_anchor_merge(
        anchor_merge: AnchorMerge, entity_displays: dict[Side, EntityDisplay]
        ) -> list[SectionContent]:
    strata = get_anchor_strata(anchor_merge)
    conflict_examples = anchor_merge.conflicts[:NUM_CONFLICT_EXAMPLES]
    entity_displays[Side.SOURCE].prefetch(source for source, _, _ in conflict_examples)
    entity_displays[Side.TARGET].prefetch(target for _, target, _ in conflict_examples)
    content: list[SectionContent] = [
        Item(f"{len(anchor_merge.anchors)} anchors"),
        Item(
            "strata (families that found the anchors / head, specific, extended): anchors",
            [
                Item(f"{stratum.name}: {len(stratum.pairs)}")
                for stratum in sorted(strata, key=lambda stratum: -len(stratum.pairs))
                ]
            ),
        Item(
            f"{len(anchor_merge.conflicts)} conflicts: an entity has different partners in different runs "
            "(left out of the anchors)",
            [
                Item(
                    f"{entity_displays[Side.SOURCE].show_line(source)} = "
                    f"{entity_displays[Side.TARGET].show_line(target)} "
                    f"({'+'.join(sorted(families))})"
                    )
                for source, target, families in conflict_examples
                ]
            ),
        ]
    return content


# ---------------------------------------------------------------- functions for the LLM


class AnchorStrategyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # the population, flat: models put the group refs at the top level
    source_group_refs: Annotated[list[GroupRef], WithJsonSchema(
        GENERATOR_SCHEMA["properties"]["population"]["properties"]["source_group_refs"])
        ] = Field(
        default=[POPULATION_REST], min_length=1,
        description="The source entities that get anchors: groups of the profile by name, 'rest' for all "
        "groups that are not excluded (default), or a group defined inline for this strategy (a new name "
        "and a SPARQL pattern over ?s). Entities that already have an equivalence get none. Uniqueness is "
        "checked against all competing entities of both knowledge graphs, not only these.")
    target_group_refs: Annotated[list[GroupRef], WithJsonSchema(
        GENERATOR_SCHEMA["properties"]["population"]["properties"]["target_group_refs"])
        ] = Field(
        default=[POPULATION_REST], min_length=1,
        description="Like source_group_refs, for the target entities.")
    source_values: ValuePatternSource = Field(
        default="names",
        description="The values compared, as a key shared by two entities: role names of the profile (names, "
        "synonyms, definitions, codes, cross_references), or an own pattern over ?s (the entity) "
        "and ?o (a value of the entity itself, such as a name or a code)."
        )
    target_values: ValuePatternSource | None = Field(
        default=None,
        description="Like source_values; omit it for the same values as the source."
        )
    normalization_levels: list[NormalizationLevel] = Field(
        default=list(STANDARD_LEVELS), min_length=1,
        description="Levels from fine to coarse, in ladder order, on "
        "which values are compared; an anchor "
        "records the finest level on which it is unique. "
        + NORMALIZATION_LEVELS_DESCRIPTION
        )


ANCHOR_ENSEMBLE_SCHEMA = {
    "type": "array",
    "description": "Tried runs whose anchors are merged. Runs of one family count as one kind of evidence "
    "when the anchors are grouped for certification (e.g. two runs on names in different groups).",
    "items": {
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "family": {"type": "string", "description": "Name of the kind of evidence; default: the run id."},
        },
        "required": ["run_id"],
        "additionalProperties": False,
    },


}

ANCHOR_FINISH_DEFINITION = {
    "name": AnchorFunctionName.FINISH,
    "description": "Ends the task: selects the tried runs whose anchors are computed in full later (on all "
    "entities, not a sample) and certified. Call it exactly once, at the end.",
    "parameters": {
        "type": "object",
        "properties": {
            "runs": {**ANCHOR_ENSEMBLE_SCHEMA,
                     "description": "The selected runs; empty if no strategy gives reliable anchors. "
                     + ANCHOR_ENSEMBLE_SCHEMA["description"]},
            "rationale": {"type": "string",
                          "description": "Which keys for which groups, why, and which anchors are risky."},
        },
        "required": ["runs", "rationale"],
        "additionalProperties": False,
    },
}


def anchor_function_definitions() -> list[dict]:
    return [
        {
            "name": AnchorFunctionName.TRY_ANCHOR_STRATEGY,
            "description": (
                "Tries a key strategy in a trial run and shows its "
                "anchors: pairs whose key no other competing entity of either knowledge graph shares. It "
                "shows how many entities get an anchor, how many share a key but stay ambiguous, how many "
                "anchors are head anchors (their name is part of longer names, e.g. 'paper' of 'poster "
                "paper', so they may name a broader or narrower thing), the existing equivalences found "
                "again, keys shared by too many entities, and examples. The run gets an id (run-N)."
                ),
            "parameters": inline_schema(AnchorStrategyArguments),
            },
        {
            "name": AnchorFunctionName.SHOW_ANCHORS,
            "description": "Shows more anchors of a run, or source entities without an anchor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "kind": {"type": "string", "enum": [str(kind) for kind in AnchorListKind],
                             "description": "random (default), head (lowest information content first), "
                                            "extended (unique only on a coarse level) or ambiguous (entities "
                                            "that share a key with several entities)."},
                    "n": {"type": "integer", "minimum": 1, "maximum": MAX_SHOWN_ANCHORS},
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
            },
        {
            "name": AnchorFunctionName.EVALUATE_ANCHOR_ENSEMBLE,
            "description": "Merges the anchors of tried runs and shows the merge: the anchors per stratum "
                           "of the certification and the conflicts between the runs (an entity with "
                           "different partners in different runs).",
            "parameters": {"type": "object", "properties": {"runs": ANCHOR_ENSEMBLE_SCHEMA},
                           "required": ["runs"], "additionalProperties": False},
            },
        ANCHOR_FINISH_DEFINITION,
    ]


def try_anchor_strategy(state: CandidateTaskState, fn_args: dict) -> str:
    return format_display_content(run_tried_anchor_strategy(state, fn_args))


def run_tried_anchor_strategy(state: CandidateTaskState, fn_args: dict) -> Section:
    """
    A trial run of the key strategy in the arguments of try_anchor_strategy,
    added to the runs of the task: its anchors and statistics.
    """
    generator = anchor_generator(fn_args)
    try:
        result = run_strategy(
            state.kg_managers, state.alignment_state, generator,
            state.entity_memberships,
            MAX_ENTITY_PAIRS_PER_BUCKET[CandidateStrategyKind.KEY], trial=True,
            )
    except (ValueError, NotImplementedError) as error:
        raise FunctionCallException(str(error)) from error
    run_id = state.add_run(generator, result)
    return format_anchor_result(run_id, result, run_anchors_of(state, run_id), state.entity_displays)


def default_anchor_arguments(alignment_state: AlignmentState) -> list[dict]:
    """
    The arguments of try_anchor_strategy for the baseline runs, as far as the
    profiles allow: names against names on all groups (pairs of partitions
    that do not compete are left out anyway), names against synonyms in each
    direction a side has synonyms, and cross-references if both KGs have them.
    """
    source, target = (alignment_state.get_kg_profile(side) for side in SIDES)
    arguments = []
    if source.is_role_set("names") and target.is_role_set("names"):
        arguments.append({"source_values": "names"})
        if target.is_role_set("synonyms"):
            arguments.append({"source_values": "names", "target_values": "synonyms"})
        if source.is_role_set("synonyms"):
            arguments.append({"source_values": "synonyms", "target_values": "names"})
    if source.is_role_set("cross_references") and target.is_role_set("cross_references"):
        arguments.append({"source_values": "cross_references"})
    return arguments


def show_anchors(state: CandidateTaskState, fn_args: dict) -> str:
    run_id = fn_args.get("run_id")
    if run_id not in state.runs:
        raise FunctionCallException(f"Unknown run_id {run_id!r}; existing runs: {list(state.runs)}")
    try:
        kind = AnchorListKind(fn_args.get("kind") or AnchorListKind.RANDOM)
    except ValueError as error:
        raise FunctionCallException(f"Invalid kind: {error}") from error
    n = min(int(fn_args.get("n") or NUM_EXAMPLE_ANCHORS), MAX_SHOWN_ANCHORS)
    _, result = state.runs[run_id]
    if kind == AnchorListKind.AMBIGUOUS:
        entities = choose_ambiguous_entities(result, n)
        prefetch_ambiguous(result, entities, state.entity_displays)
        lines = [format_ambiguous_entity(result, entity, state.entity_displays) for entity in entities]
    else:
        anchors = anchors_of_kind(run_anchors_of(state, run_id), kind)[:n]
        prefetch_anchors(anchors, state.entity_displays)
        lines = [format_anchor(anchor, state.entity_displays) for anchor in anchors]
    return format_display_content(Section(f"{kind} of {run_id}", lines or ["none"]))


def evaluate_anchor_ensemble(state: CandidateTaskState, fn_args: dict) -> str:
    run_ensemble = parse_run_ensemble(state, fn_args.get("runs"))
    return format_display_content(Section(
        "Ensemble " + ", ".join(run_ensemble.run_ids()),
        format_anchor_merge(merge_anchors(run_ensemble, state.runs), state.entity_displays),
        ))


def finish(state: CandidateTaskState, fn_args: dict) -> str:
    runs = fn_args.get("runs")
    rationale = fn_args.get("rationale")
    if not isinstance(runs, list):
        raise FunctionCallException("runs must be a list of runs (it may be empty)")
    if not rationale:
        raise FunctionCallException("A rationale is required")
    state.run_ensemble = parse_run_ensemble(state, runs) if runs else None
    state.rationale = rationale
    state.finished = True
    if state.run_ensemble is None:
        return "Selected runs: none (no anchors)."
    return "Selected runs: " + ", ".join(state.run_ensemble.run_ids()) + "."


ANCHOR_FUNCTIONS: dict[str, Callable[[CandidateTaskState, dict], str]] = {
    AnchorFunctionName.TRY_ANCHOR_STRATEGY: try_anchor_strategy,
    AnchorFunctionName.SHOW_ANCHORS: show_anchors,
    AnchorFunctionName.EVALUATE_ANCHOR_ENSEMBLE: evaluate_anchor_ensemble,
    AnchorFunctionName.FINISH: finish,
}


class OMAnchorTask(GraspTask):
    """
    The anchor stage. Input: a dict with source_kg, target_kg (names of the
    configured KGs; may be omitted with exactly two KGs) and alignment_state
    (a serialized AlignmentState with the profiles and the equivalences so far).
    """
    name = "om-anchors"
    state: CandidateTaskState

    def setup(self, input: Any) -> str:
        alignment_state = resolve_alignment_state(input)
        kg_managers = managers_by_side(self, alignment_state.source_kg_name, alignment_state.target_kg_name)
        entity_memberships = resolve_entity_memberships(input, kg_managers, alignment_state)
        self.state = CandidateTaskState(
            kg_managers=kg_managers,
            alignment_state=alignment_state,
            entity_memberships=entity_memberships,
            strategy_history=resolve_strategy_history(input),
            entity_displays={
                side: EntityDisplay(
                    kg_managers[side], alignment_state.get_kg_profile(side),
                    entity_memberships.group_names_by_entity[side])
                for side in SIDES
            },
        )
        baseline_runs = None if self.state.strategy_history.has_stage("anchors") else format_baseline_runs(
            self.state, default_anchor_arguments(alignment_state), run_tried_anchor_strategy,
            AnchorFunctionName.TRY_ANCHOR_STRATEGY)
        if baseline_runs is None:
            return format_start_message(self.state) + "\n\nNo anchor strategy has been tried yet."
        return format_start_message(self.state, baseline_runs)

    def system_information(self) -> str:
        info = fill_function_name_placeholders(
            add_equivalence_condition(load_prompt("anchor_task.txt"), self.config), AnchorFunctionName
            )
        task_instructions = load_task_instructions(self.config)
        if task_instructions:
            info += "\n\n" + format_section("Task-specific instructions", task_instructions)
        return info

    def rules(self) -> list[str]:
        return [
            BLANK_NODE_RULE,
            f"Try every strategy you want to select with {AnchorFunctionName.TRY_ANCHOR_STRATEGY}; you can only select tried runs.",
            f"Look at the examples, head anchors and ambiguous entities with {AnchorFunctionName.SHOW_ANCHORS} before you judge "
            "a strategy.",
            "You can execute multiple tool calls in one step.",
            f"You have at most {self.config.max_steps} steps; one step is one response of yours, with "
            f"all tool calls in it. Call {AnchorFunctionName.FINISH} exactly once, well before the limit.",
        ]

    def function_definitions(self) -> list[dict]:
        return anchor_function_definitions()

    def final_step_functions(self) -> list[dict] | None:
        return [ANCHOR_FINISH_DEFINITION]

    def call_function(self, fn_name: str, fn_args: dict, known: set[str], example_indices: dict | None) -> str:
        function = ANCHOR_FUNCTIONS.get(fn_name)
        if function is None:
            raise FunctionCallException(f"Unknown function {fn_name}")
        return function(self.state, fn_args)

    def done(self, fn_name: str) -> bool:
        return fn_name == AnchorFunctionName.FINISH and self.state.finished

    def output(self, messages: list[Message]) -> dict:
        state = self.state
        return AnchorStageResult(
            source_kg=state.kg_managers[Side.SOURCE].kg,
            target_kg=state.kg_managers[Side.TARGET].kg,
            finished=state.finished,
            rationale=state.rationale,
            run_ensemble=state.run_ensemble,
            runs={
                run_id: TriedRun(
                    generator=generator,
                    signals=result.signals,
                    confirmed_equivalences=len(result.confirmed_equivalences),
                    existing_equivalences=result.num_existing_equivalences_in_population,
                    num_hubs=len(result.hubs),
                )
                for run_id, (generator, result) in state.runs.items()
            },
        ).model_dump(mode="json")
