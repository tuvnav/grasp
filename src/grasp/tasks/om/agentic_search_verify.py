import json
import random
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from pydantic import BaseModel, model_validator
from universal_ml_utils.table import generate_table

from grasp.configs import GraspConfig, ShapeConfig
from grasp.functions import (
    find_manager,
    list_triples,
    update_known_from_shape_iris,
    SHAPE_CAVEAT,
)
from grasp.manager import KgManager, format_kgs
from grasp.model import Message
from grasp.sparql.types import AskResult, ObjType, SelectResult
from grasp.sparql.utils import prepare_identifier_for_sparql
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.tasks.entities import Entity, prepare_entity
from grasp.utils import FunctionCallException, format_list, format_notes, format_section
from grasp.build.shapes import collect_iris, compute_shape, emit_pseudo_shex
from grasp.tasks.om.model import (
    allows_many, format_entity, PotentialCorrespondences,
    Correspondence, RelationProfile,
    RELATION_PROFILES, CorrespondenceSide,
    RelationSpec, get_relation_profile
    )


class AlignmentTaskInput(BaseModel):
    unmatched_entities: list[Entity] = []
    potential_correspondences: list[PotentialCorrespondences] = []
    input_alignment: list[Correspondence] = []

    source_kg: str
    target_kg: str

    description: str | None = None


class AlignmentState:

    # class-level on purpose: shared across all jsonl lines processed in one
    # process, so single-entity runs build on each other's correspondences
    __global_correspondences: dict[tuple[str, str, str], Correspondence] = {}

    def __init__(self, profile: RelationProfile | None = None):
        self.task_input: AlignmentTaskInput | None = None
        self.profile = profile or RELATION_PROFILES["equivalence"]
        # mapping from (source IRI, target IRI, relation) to the correspondence;
        # an entity pair may hold several relations
        self.correspondences: dict[tuple[str, str, str], Correspondence] = AlignmentState.__global_correspondences
        # TODO: Dicts for entity -> Correspondences
        # consistency warnings already shown to the LLM: key of the proposed
        # correspondence -> keys of the existing correspondences the warning
        # was about. Per conversation, unlike the shared correspondences.
        self.pending_warnings: dict[tuple[str, str, str], frozenset[tuple[str, str, str]]] = {}
        # results of the scope checks: (side, IRI) -> whether it is in scope
        self.scope_cache: dict[tuple[CorrespondenceSide, str], bool] = {}

    @classmethod
    def clear_global(cls):
        cls.__global_correspondences.clear()

    def add_correspondence(self, correspondence: Correspondence):
        """Adds the correspondence without any checks."""
        self.correspondences[correspondence.key] = correspondence

    def get_correspondence(
        self, entity_source: str, entity_target: str, relation: str
    ) -> Correspondence | None:
        return self.correspondences.get((entity_source, entity_target, relation))

    def remove_correspondence(
        self, entity_source: str, entity_target: str, relation: str
    ) -> Correspondence | None:
        return self.correspondences.pop((entity_source, entity_target, relation), None)

    def correspondences_of_pair(self, entity_source: str, entity_target: str) -> list[Correspondence]:
        """All correspondences between `entity_source` and `entity_target`, of any relation."""
        return [
            c for key, c in self.correspondences.items()
            if key[0] == entity_source and key[1] == entity_target
        ]

    # TODO: To be optimized by using a dict entity -> Correspondences
    def correspondences_of(
        self,
        entity: str,
        side: CorrespondenceSide,
        relation: str | None = None,
    ) -> list[Correspondence]:
        """
        All correspondences in which the full IRI `entity` takes part as
        source or target entity (`side`), optionally only those of `relation`.
        """
        idx = 0 if side == CorrespondenceSide.SOURCE else 1
        return [
            c for key, c in self.correspondences.items()
            if key[idx] == entity and (relation is None or c.relation == relation)
        ]

    def discard_correspondences_of(self, entity: str, side: CorrespondenceSide, relation: str | None = None):
        for c in self.correspondences_of(entity, side, relation):
            self.correspondences.pop(c.key, None)

    def connected_correspondences(
        self,
        entity_source: str,
        entity_target: str,
        relation: str,
    ) -> list[Correspondence]:
        """
        All existing correspondences of `relation` that would end up in one
        connected component with the (possibly not yet existing) pair
        (`entity_source`, `entity_target`). For a symmetric and transitive
        relation all entities of this component are implied to be pairwise
        related by it.
        """
        # nodes are (side, IRI), since the same IRI may occur in both ontologies
        start: list[tuple[CorrespondenceSide, str]] = [(CorrespondenceSide.SOURCE, entity_source), (CorrespondenceSide.TARGET, entity_target)]
        seen = set(start)
        queue = deque(start)
        found: dict[tuple[str, str, str], Correspondence] = {}
        while queue:
            side, iri = queue.popleft()
            for c in self.correspondences_of(iri, side, relation):
                found[c.key] = c
                other: tuple[CorrespondenceSide, str] = (
                    (CorrespondenceSide.TARGET, c.entity_target.identifier) if side == "source"
                    else (CorrespondenceSide.SOURCE, c.entity_source.identifier)
                )
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        found.pop((entity_source, entity_target, relation), None)
        return list(found.values())

    def to_dict(self, reduced: list[Correspondence] | None = None) -> dict:
        """
        With `reduced`, "correspondences" holds the reduced alignment (what
        evaluation reads) and "raw_correspondences" everything the LLM set.
        """
        output = {
            "formatted": self.format(),
            "correspondences": [c.model_dump() for c in self.correspondences.values()]
        }
        if reduced is not None:
            output["raw_correspondences"] = output["correspondences"]
            output["correspondences"] = [c.model_dump() for c in reduced]
        return output

    def format(self, num: int | None = None) -> str:
        correspondences = list(self.correspondences.values())

        if not correspondences:
            return "No correspondences established yet."

        if num is not None and num < len(correspondences):
            correspondences = random.sample(correspondences, num)

        data = [
            [c.entity_source.identifier, c.entity_target.identifier, c.relation]
            for c in correspondences
        ]
        headers = [["Source Entity (Ontology 1)", "Target Entity (Ontology 2)", "Relation"]]

        return generate_table(data=data, headers=headers)


def set_correspondence_description(profile: RelationProfile) -> str:
    description = """\
Establishes a semantic correspondence between an entity from the source \
ontology and one from the target ontology."""
    if len(profile.relations) > 1:
        description += """ An entity pair may hold several relations.

The system checks the new correspondence against the relations already set for \
the same entity pair (contradicting or implied relations) and, for symmetric and \
transitive relations (such as equivalence), against the existing correspondences \
it would be connected to, since that would imply relations between entities of \
the same ontology."""
    else:
        description += """

For symmetric and transitive relations (such as equivalence), the system checks \
the new correspondence against the existing correspondences it would be connected \
to, since that would imply relations between entities of the same ontology."""
    description += """ In case of such issues the correspondence is not set yet. \
Instead, a warning lists the involved correspondences and your options. To set \
the correspondence anyway, call this function again with the same arguments.
Use this to lock in your alignment decisions."""
    return description


def relation_property(profile: RelationProfile) -> dict:
    return {
        "type": "string",
        "enum": [r.iri for r in profile.relations],
        "description": "The relation holding from the source to the target entity:\n"
        + "\n".join(f"- '{r.iri}': {r.description}" for r in profile.relations),
    }


def resolve_relation(fn_args: dict, profile: RelationProfile) -> RelationSpec:
    # with a single relation, the functions have no relation parameter
    relation_iri = fn_args.get("relation", profile.relations[0].iri)
    relation = profile.get(relation_iri)
    if relation is None:
        raise FunctionCallException(
            f"Unknown relation '{relation_iri}', must be one of "
            f"{[r.iri for r in profile.relations]}."
        )
    return relation


def functions(managers: list[KgManager], profile: RelationProfile) -> list[dict]:
    set_properties: dict[str, dict] = {
        "source_entity": {
            "type": "string",
            "description": "The exact IRI of the source entity",
        },
        "target_entity": {
            "type": "string",
            "description": "The exact IRI of the target entity",
        },
    }
    # with a single relation the LLM has nothing to choose
    if len(profile.relations) > 1:
        set_properties["relation"] = relation_property(profile)

    delete_properties: dict[str, dict] = {
        "source_entity": {
            "type": "string",
            "description": "The full IRI of the source entity whose correspondence should be removed",
        },
        "target_entity": {
            "type": "string",
            "description": "The full IRI of the target entity whose correspondence should be removed",
        },
    }
    if len(profile.relations) > 1:
        delete_properties["relation"] = relation_property(profile)

    fns = [
        {
            "name": "set_correspondence",
            "description": set_correspondence_description(profile),
            "parameters": {
                "type": "object",
                "properties": set_properties,
                "required": list(set_properties),
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_correspondence",
            "description": "Delete an existing correspondence.",
            "parameters": {
                "type": "object",
                "properties": delete_properties,
                "required": list(delete_properties),
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_correspondences",
            "description": "Displays the current state of established mappings. If 'num' is provided, "
            "it shows a random sample; otherwise, it lists all correspondences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num": {
                        "type": "integer",
                        "description": "The number of random mappings to display for review",
                        },
                    },
                "required": [],
                "additionalProperties": False,
                },
            "strict": False,
        },
        {
            "name": "stop",
            "description": "Finalize your mappings and stop the alignment process. May be "
            "rejected if there is still an unresolved logical conflict in the current "
            "alignment - resolve it and call this again.",
        },
    ]
    return fns


def load_task_instructions(om_kwargs: dict) -> str | None:
    """
    Free-text instructions specific to the alignment task at hand (e.g. domain
    knowledge needed to decide on the relations), given inline or as the path
    to a .md/.txt file in the `task_instructions` om task kwargs.
    """
    spec = om_kwargs.get("task_instructions")
    if not spec:
        return None
    if "\n" not in spec and Path(spec).suffix in (".md", ".txt"):
        spec = Path(spec).expanduser().read_text(encoding="utf-8")
    return spec.strip() or None


def system_information(
    profile: RelationProfile,
    task_instructions: str | None = None,
) -> str:
    relations = "\n".join(f"- '{r.iri}': {r.description}" for r in profile.relations)
    if profile.description:
        relations += f"\n\n{profile.description.strip()}"
    if len(profile.relations) == 1:
        relation_intro = "Each correspondence relates them by the following relation:"
    else:
        relation_intro = "You can relate them by the following relations:"
    info = f"""\
You are an expert ontology alignment assistant. Your mission is to establish \
correspondences between the entities (e.g. classes, properties and instances) of a source \
and a target ontology. A correspondence <source entity> <relation> <target entity> \
is always read from the source to the target entity. {relation_intro}
{relations}

Whether a correspondence holds is ultimately a functional question: ontology \
alignments are used to support cross-ontology queries and interoperability \
between different systems, one using each ontology. Ask yourself what a query \
for an entity in one system should return in the other one - the relation \
descriptions above state this for each relation. Lexical information is your \
primary evidence for answering this question; logical signals support or \
relativize it.

Follow this step-by-step approach:
1. Analyze the source and target ontology to understand their core domain and \
conceptual scope. Consider how concepts in the source ontology might be \
represented, structured or labeled differently in the target ontology.
2. For each given source entity:
a. Search the target ontology for entities that are lexically related to it. \
The lexical information of an entity comprises its own labels, synonyms and \
other lexical information as well as those of its neighboring entities (e.g. \
related entities and the classes it instantiates). If an entity has no \
meaningful label of its own (e.g. only a label derived from an identifier-like \
IRI), the labels of its neighbors are its lexical description.
b. Check logical signals for support or contradiction: the hierarchies of both \
ontologies, and the logical consequences a correspondence would have, e.g. \
which relations it would imply between their sub- and superclasses or between \
entities within the same ontology.
c. Decide from the functional point of view which relation, if any, holds \
between the source entity and each candidate, and set the corresponding \
correspondences. If no relation holds, leave the source entity unmatched.
3. Once all given source entities have been processed, use the stop function \
to finalize the alignment."""
    if task_instructions:
        info += "\n\n" + format_section("Task-specific instructions", task_instructions)
    return info


def rules(
    profile: RelationProfile,
    alignment_form: str = "raw",
    scope_description: str | None = None,
) -> list[str]:
    """
    Only rules needed to understand the task. Rules derived from relation
    properties explain the consistency checks the system performs.
    """
    rules = []

    # alignment-level cardinality in Alignment API notation: '*' or '+' on a
    # side means its entities may take part in several correspondences
    many = allows_many(profile)
    if many:
        directed = next((r.iri for r in profile.relations if not r.symmetric), None)
        example = (
            f" (e.g. a source entity can be related by '{directed}' to several target entities)"
            if directed else ""
        )
        rules.append(
            "Multiple correspondences: An entity may take part in several correspondences, "
            f"also of different relations{example}. Set every correspondence that holds."
        )

    reducible = [r.iri for r in profile.relations if r.iri in REDUCIBLE]
    if alignment_form == "reduced" and reducible:
        rules.append(
            "Implied correspondences: Correspondences that follow from other correspondences "
            "together with the hierarchies of the ontologies are removed automatically from "
            "the final alignment, so you do not need to set them. E.g. 's < t2' follows from "
            "'s < t1' if t1 is below t2 in the target hierarchy, and 's > t1' follows from "
            "'s > t2' or 's = t2' in the same case."
        )

    for r in profile.relations:
        if r.symmetric and r.transitive and profile.duplicates_within:
            rules.append(
                f"Transitivity of '{r.iri}': If several source entities are related by "
                f"'{r.iri}' to the same target entity (or one source entity to several "
                f"target entities), they are implied to be pairwise '{r.iri}' to each "
                "other within their own ontology. The ontologies may contain duplicate "
                "entities, so this can be correct: set all of these correspondences if "
                "the implied relations within the ontologies hold as well."
            )
        elif r.symmetric and r.transitive:
            rules.append(
                f"Transitivity and near-injectivity of '{r.iri}': If several source entities "
                f"are related by '{r.iri}' to the same target entity (or one source entity to "
                f"several target entities), they are implied to be pairwise '{r.iri}' to each "
                "other within their own ontology. In most cases, only one of them is correct."
            )

    # mutually exclusive relations, grouped where possible, e.g. {=, <, >}
    def excluded(a: str, b: str) -> bool:
        return b in profile.get(a).excludes or a in profile.get(b).excludes  # type: ignore

    covered: set[frozenset[str]] = set()
    for r in profile.relations:
        group = [x.iri for x in profile.relations if x.iri == r.iri or excluded(r.iri, x.iri)]
        pairs = {frozenset(p) for p in combinations(group, 2)}
        if len(group) < 2 or pairs <= covered:
            continue
        if all(excluded(a, b) for a, b in combinations(group, 2)):
            rules.append(
                "Mutually exclusive relations: At most one of "
                + ", ".join(f"'{iri}'" for iri in group)
                + " holds for an entity pair."
            )
            covered |= pairs
        else:
            others = [iri for iri in group if iri != r.iri]
            rules.append(
                f"Exclusion: '{r.iri}' cannot hold for an entity pair together with "
                + ", ".join(f"'{iri}'" for iri in others) + "."
            )
            covered |= {frozenset((r.iri, o)) for o in others}

    completeness = (
        "Completeness: Perform a final comprehensive review of all established "
        "correspondences before concluding the task. You must explicitly evaluate every "
        "single entity provided in the source list."
    )
    if many:
        completeness += (
            " For each of them, check every relation against all relevant target entities "
            "and do not stop after the first correspondence found."
        )
    no_match = "has no correspondence" if many else "has no match"
    completeness += (
        f" If an entity truly {no_match}, explain briefly why, but do not simply skip it."
    )
    rules.append(completeness)

    if scope_description:
        rules.append(f"Scope: {scope_description}")
    return rules


def entity_exists_in_kg(manager: KgManager, iri: str) -> bool:
    # subject and object position cover classes/instances; predicate position
    # is needed too since a property may appear only there (e.g. used in
    # triples without a separate rdf:type owl:ObjectProperty/DatatypeProperty
    # declaration triple of its own)
    wrapped = prepare_identifier_for_sparql(iri, manager.iri_literal_parser)
    sparql = (
        f"ASK {{ {{ {wrapped} ?p ?o }} UNION {{ ?s ?p {wrapped} }} "
        f"UNION {{ ?s {wrapped} ?o }} }}"
    )
    result = manager.execute_sparql(sparql)
    assert isinstance(result, AskResult)
    return result.boolean


def ensure_known(manager: KgManager, iri: str, known: set[str]) -> None:
    """
    Fast path: skip the SPARQL round-trip if the IRI was already surfaced by
    an earlier function call result during this conversation. Otherwise fall
    back to an actual live existence check against the KG, and only report
    the IRI as invalid to the LLM once that check has genuinely failed.
    """
    if iri in known:
        return

    if entity_exists_in_kg(manager, iri):
        known.add(iri)
        return

    raise FunctionCallException(
        f"The entity {iri} does not seem to exist in the knowledge graph. "
        "Double check the IRI."
    )


def scope_sparql(scope: dict, side: CorrespondenceSide) -> str | None:
    """
    The ASK query template with an {IRI} placeholder that restricts the
    entities of `side` which may take part in correspondences, given inline or
    as the path to a .sparql/.rq file in the `scope` om task kwargs. None if
    the side is not restricted.
    """
    spec = scope.get(side)
    if spec is None:
        return None
    if "\n" not in spec and Path(spec).suffix in (".sparql", ".rq"):
        spec = Path(spec).expanduser().read_text(encoding="utf-8")
    if "{IRI}" not in spec:
        raise ValueError(f"Scope SPARQL for the {side} side must contain an {{IRI}} placeholder")
    return spec


def ensure_in_scope(
    manager: KgManager,
    iri: str,
    side: CorrespondenceSide,
    scope: dict,
    state: "AlignmentState",
) -> None:
    # set-based restriction by IRI prefixes (e.g. the entity space of the anchor
    # stage), checked before the per-IRI ASK query
    namespaces = scope.get("namespaces", {}).get(side)
    if namespaces and not iri.startswith(tuple(namespaces)):
        message = (
            f"The entity {iri} is not among the entities to be aligned in the "
            f"{side} ontology {manager.kg}: its IRI starts with none of {namespaces}."
        )
        if scope.get("description"):
            message += f" {scope['description']}"
        raise FunctionCallException(message)

    sparql = scope_sparql(scope, side)
    if sparql is None:
        return

    key = (side, iri)
    if key not in state.scope_cache:
        wrapped = prepare_identifier_for_sparql(iri, manager.iri_literal_parser)
        result = manager.execute_sparql(sparql.replace("{IRI}", wrapped))
        assert isinstance(result, AskResult), "Scope SPARQL must be an ASK query"
        state.scope_cache[key] = result.boolean

    if not state.scope_cache[key]:
        message = (
            f"The entity {iri} is not among the entities to be aligned in the "
            f"{side} ontology {manager.kg}."
        )
        if scope.get("description"):
            message += f" {scope['description']}"
        raise FunctionCallException(message)


def get_entity_shape_text(
    manager: KgManager,
    iri: str,
    known: set[str] | None = None,
) -> str | None:
    """
    Look up (or compute on the fly) the shape for the given, already-expanded
    IRI, rendered as pseudo-SHEx text. Returns None if no shape is available
    for it (e.g. the entity is a property/instance rather than a class, or no
    shape index/patterns exist for this KG at all). If `known` is given, IRIs
    referenced by the shape are added to it, same as when a shape is surfaced
    via the interactive get_shape/search_shape functions.
    """
    if manager.shapes is None:
        return None

    shapes = manager.shapes
    shape_config = manager.shape_config or ShapeConfig()

    sample = shapes.index.get_by_iri(iri) if shapes.index is not None else None
    if sample is not None:
        profile = sample.profile
    elif shapes.instance_pattern is not None or shapes.schema_pattern is not None:
        try:
            profile = compute_shape(
                iri,
                manager,
                instance_pattern=shapes.instance_pattern,
                schema_pattern=shapes.schema_pattern,
                shape_config=shape_config,
            )
        except Exception:
            return None
    else:
        return None

    if known is not None:
        update_known_from_shape_iris(
            known, collect_iris(profile, manager, shape_config), manager,
        )

    return emit_pseudo_shex(profile, manager, shape_config)


class EntityContextConfig(BaseModel):
    """
    Triples around an entity shown to the LLM together with the entity, as
    returned by the list function, e.g. for entities without meaningful labels.
    """
    # list(subject=<entity>) and list(object=<entity>)
    outgoing: bool = True
    incoming: bool = False
    # rows per list call, defaults to the list_k of the config
    k: int | None = None
    # predicates along which neighbors without a label are described, e.g.
    # the type predicate of the knowledge graph; empty means not at all
    expand_unlabeled: list[str] = []
    # maximum number of neighbors without a label to describe
    max_expanded: int = 5
    # no context for entities whose shape is shown (off by default, since
    # shapes computed for instances may be empty)
    skip_if_shape: bool = False


def get_entity_context_config(om_kwargs: dict) -> EntityContextConfig | None:
    cfg = om_kwargs.get("entity_context")
    if cfg is None:
        return None
    return EntityContextConfig(**cfg)


def unlabeled_neighbors(
    manager: KgManager,
    iri: str,
    cfg: EntityContextConfig,
) -> list[str]:
    """
    IRIs of the neighbors of `iri` (in the directions set in `cfg`) that have
    no label in the entities index, in deterministic order.
    """
    wrapped = prepare_identifier_for_sparql(iri, manager.iri_literal_parser)
    patterns = []
    if cfg.outgoing:
        patterns.append(f"{{ {wrapped} ?p ?n }}")
    if cfg.incoming:
        patterns.append(f"{{ ?n ?p {wrapped} }}")
    if not patterns:
        return []
    sparql = (
        f"SELECT DISTINCT ?n WHERE {{ {' UNION '.join(patterns)} "
        f"FILTER(isIRI(?n) && ?n != {wrapped}) }} ORDER BY ?n LIMIT 1000"
    )
    result = manager.execute_sparql(sparql)
    assert isinstance(result, SelectResult)
    neighbors = [row["n"].identifier() for row in result.rows() if "n" in row]
    info = manager.get_info_for_identifiers_from_index(
        neighbors, ObjType.ENTITY.index_name
    )
    unlabeled = [n for n in neighbors if not info.get(n, {}).get("label")]
    return unlabeled[:cfg.max_expanded]


def get_entity_context_text(
    manager: KgManager,
    entity: Entity,
    cfg: EntityContextConfig,
    config: GraspConfig,
    known: set[str],
) -> str | None:
    """
    The context of `entity` as returned by list calls, each preceded by the
    call itself so that the LLM can continue from there (e.g. with the next
    page). None if no call returned anything.
    """
    k = cfg.k or config.list_k
    timeouts = (config.sparql_request_timeout, config.sparql_read_timeout)
    calls: list[tuple[str, dict]] = []
    if cfg.outgoing:
        calls.append((f'subject="{entity.entity}"', dict(subject=entity.identifier)))
    if cfg.incoming:
        calls.append((f'object="{entity.entity}"', dict(obj=entity.identifier)))
    if cfg.expand_unlabeled:
        for neighbor in unlabeled_neighbors(manager, entity.identifier, cfg):
            for pred in cfg.expand_unlabeled:
                calls.append((
                    f'subject="{manager.format_iri(neighbor)}", property="{pred}"',
                    dict(subject=neighbor, property=pred),
                ))

    blocks = []
    for call, args in calls:
        result = list_triples(
            [manager],
            manager.kg,
            args.get("subject"),
            args.get("property"),
            args.get("obj"),
            1,
            False,
            k,
            known,
            *timeouts,
        )
        if result.startswith("No triples"):
            continue
        blocks.append(f'list(kg="{manager.kg}", {call}):\n{result}')

    return "\n\n".join(blocks) or None


def pair_inconsistencies(
        relation: RelationSpec,
        pair: list[Correspondence],
        profile: RelationProfile,
        ) -> tuple[list[Correspondence], list[Correspondence], list[Correspondence]]:
    """
    Checks a new `relation` against the relations already set for the same
    entity pair. Returns the existing correspondences that (1) contradict it
    (`excludes`, in either direction), (2) make it redundant because they
    imply it, and (3) become redundant because it implies them.
    """
    contradicting, implying, implied = [], [], []
    for c in pair:
        other = profile.get(c.relation)
        if other is None:
            continue
        if other.iri in relation.excludes or relation.iri in other.excludes:
            contradicting.append(c)
        elif relation.iri in other.implies:
            implying.append(c)
        elif other.iri in relation.implies:
            implied.append(c)
    return contradicting, implying, implied


def format_consistency_warning(
        entity_source: str,
        entity_target: str,
        full_iri_source: str,
        full_iri_target: str,
        kg_source: str,
        kg_target: str,
        relation: RelationSpec,
        contradicting: list[Correspondence],
        implying: list[Correspondence],
        implied: list[Correspondence],
        connected: list[Correspondence],
        ) -> str:
    """
    Warning for a proposed correspondence that is not consistent with, or
    redundant to, existing ones. `entity_source` and `entity_target` are the
    IRIs as given by the LLM.
    """
    r = relation.iri
    new = f"'{entity_source}' {r} '{entity_target}'"

    def line(c: Correspondence) -> str:
        return f"- '{c.entity_source.entity}' {c.relation} '{c.entity_target.entity}'\n"

    message = f"Consistency warning for {new}:\n"

    if contradicting:
        message += (
            f"\nThe following existing relations of the same entity pair contradict '{r}', "
            "i.e. they cannot hold at the same time:\n"
        )
        message += "".join(line(c) for c in contradicting)
    if implying:
        message += (
            f"\nThe following existing relations of the same entity pair already imply '{r}', "
            "so the new correspondence would be redundant:\n"
        )
        message += "".join(line(c) for c in implying)
    if implied:
        message += (
            f"\n'{r}' implies the following existing relations of the same entity pair, "
            "so they would become redundant:\n"
        )
        message += "".join(line(c) for c in implied)

    if connected:
        # all entities of the resulting component per side, as displayed to the LLM
        sources = {full_iri_source: entity_source}
        targets = {full_iri_target: entity_target}
        for c in connected:
            sources.setdefault(c.entity_source.identifier, c.entity_source.entity)
            targets.setdefault(c.entity_target.identifier, c.entity_target.entity)

        implied_within = []
        for names, kg, side in [(sources, kg_source, CorrespondenceSide.SOURCE), (targets, kg_target, CorrespondenceSide.TARGET)]:
            displayed = list(names.values())
            for i, a in enumerate(displayed):
                for b in displayed[i + 1:]:
                    implied_within.append(f"- '{a}' {r} '{b}' (within the {side} ontology {kg})\n")

        message += (
            f"\n'{r}' is symmetric and transitive. The new correspondence would be connected "
            f"to these existing '{r}' correspondences:\n"
        )
        message += "".join(line(c) for c in connected)
        message += "Together they would imply the following relations within the same ontology:\n"
        message += "".join(implied_within)

    message += (
        "\nThe new correspondence has NOT been set yet. The following options are "
        "equally valid - compare all involved entities semantically and structurally and "
        "choose deliberately:\n"
        "- Do not set the new correspondence: no further action needed, the existing "
        "correspondences are kept.\n"
        "- Replace (some of) the existing correspondences: remove those you consider wrong "
        "with `delete_correspondence`, then call `set_correspondence` again with the same "
        "arguments.\n"
        "- Keep all: if all relations listed above actually hold together, call "
        "`set_correspondence` again with the same arguments. The new correspondence is then "
        "set in addition to the existing ones."
    )
    return message


def add_correspondence(
        managers: list[KgManager],
        kg_source: str,
        kg_target: str,
        entity_source: str,
        entity_target: str,
        relation: RelationSpec,
        state: AlignmentState,
        known: set[str],
        know_before_use: bool = True,
        scope: dict | None = None,
        ) -> str:

    try:
        manager_source, _ = find_manager(managers, kg_source)
        manager_target, _ = find_manager(managers, kg_target)
        entity_object_source = prepare_entity(manager_source, entity_source)
        entity_object_target = prepare_entity(manager_target, entity_target)
        full_iri_source = entity_object_source.identifier
        full_iri_target = entity_object_target.identifier

        if know_before_use:
            ensure_known(manager_source, full_iri_source, known)
            ensure_known(manager_target, full_iri_target, known)

        if scope:
            ensure_in_scope(manager_source, full_iri_source, CorrespondenceSide.SOURCE, scope, state)
            ensure_in_scope(manager_target, full_iri_target, CorrespondenceSide.TARGET, scope, state)

        if state.get_correspondence(full_iri_source, full_iri_target, relation.iri) is not None:
            return (
                f"{entity_source} from {kg_source} is already aligned with "
                f"{entity_target} from {kg_target} by '{relation.iri}'."
            )

        # cheap local checks: other relations of the same pair, and for
        # symmetric and transitive relations the connected component
        contradicting, implying, implied = pair_inconsistencies(
            relation,
            state.correspondences_of_pair(full_iri_source, full_iri_target),
            state.profile,
        )
        connected = []
        if relation.symmetric and relation.transitive:
            connected = state.connected_correspondences(
                full_iri_source, full_iri_target, relation.iri
            )
        involved = frozenset(c.key for c in [*contradicting, *implying, *implied, *connected])

        key = (full_iri_source, full_iri_target, relation.iri)
        warned = state.pending_warnings.get(key)
        # warn on the first call, and again on a repeated call if the existing
        # correspondences changed in a way the LLM has not seen yet; a repeated
        # call after a warning means "keep all"
        if involved and (warned is None or not involved <= warned):
            state.pending_warnings[key] = involved
            return format_consistency_warning(
                entity_source, entity_target,
                full_iri_source, full_iri_target,
                kg_source, kg_target,
                relation, contradicting, implying, implied, connected,
            )
        state.pending_warnings.pop(key, None)

        correspondence = Correspondence(
            entity_source=entity_object_source,
            entity_target=entity_object_target,
            relation=relation.iri,
        )
        state.add_correspondence(correspondence)

        note = ""
        if involved:
            note = (
                f" It is set in addition to the {len(involved)} existing "
                "correspondence(s) from the warning."
            )
        if len(state.profile.relations) == 1:
            return f"Aligned {entity_source} from {kg_source} with {entity_target} from {kg_target}.{note}"
        return (
            f"Aligned {entity_source} from {kg_source} with {entity_target} "
            f"from {kg_target} by '{relation.iri}'.{note}"
        )

    except ValueError as e:
        raise FunctionCallException(str(e)) from e


def delete_correspondence(
        entity1: str,
        entity2: str,
        relation: str,
        state: AlignmentState,
        ) -> str:
    c = state.remove_correspondence(entity1, entity2, relation)
    if c is None:
        raise FunctionCallException(
            f"There is no Correspondence aligning {entity1} to {entity2} by '{relation}' yet. "
            "Notice you have to use the full IRI as entity reference."
            )
    return f"Deleted correspondence with relation {c.relation} between {entity1} and {entity2}"


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: AlignmentState | None = None,
    example_indices: dict | None = None,
) -> str:
    assert isinstance(state, AlignmentState), (
        "Alignments must be provided as state for OM task"
    )
    assert not example_indices, "Example indices are not supported for OM task"

    om_kwargs = config.task_kwargs.get("om", {})

    if fn_name == "set_correspondence":
        if state is None:
            return "No AlignmentState was provided. Cannot set correspondences without AlignmentState"
        kg1 = state.task_input.source_kg  # type: ignore
        kg2 = state.task_input.target_kg  # type: ignore
        ent1, ent2 = fn_args["source_entity"], fn_args["target_entity"]
        know_before_use = om_kwargs.get("know_before_use", True)

        relation = resolve_relation(fn_args, state.profile)

        return add_correspondence(
            managers, kg1, kg2, ent1, ent2, relation, state, known,
            know_before_use, om_kwargs.get("scope"),
        )

    elif fn_name == "delete_correspondence":
        return delete_correspondence(
            fn_args["source_entity"],
            fn_args["target_entity"],
            resolve_relation(fn_args, state.profile).iri,
            state
            )

    elif fn_name == "show_correspondences":
        return state.format(num=fn_args.get("num"))

    elif fn_name == "stop":
        # TODO: Consistency warnings the LLM never answered (state.pending_warnings)
        # are dropped silently on stop, i.e. "do not set the new correspondence"
        # is the implicit default and the three options are not equal in effect.
        # Options: send the pending warnings back and refuse to stop (would need
        # `OmTask.done()` to return False while warnings are pending - om.py only),
        # or resolve them automatically. Note that a conversation also ends when
        # the LLM answers without any tool call, which bypasses `done()` entirely
        # (see the `should_stop` handling in core.py) - covering that path needs
        # a change in core.py, to be decided with the GRASP maintainers. The
        # feedback loop (`feedback_instructions`) could be another place to
        # surface pending warnings.
        return "Stopping"

    else:
        raise ValueError(f"Unknown function {fn_name}")


def input_instructions(
    task_input: AlignmentTaskInput,
    state: AlignmentState,
    managers: list[KgManager],
    known: set[str],
    config: GraspConfig | None = None,
) -> str:
    instructions = ""
    source_manager, _ = find_manager(managers, task_input.source_kg)
    target_manager, _ = find_manager(managers, task_input.target_kg)
    shape_shown = False
    block_num = 0
    context_cfg = (
        get_entity_context_config(config.task_kwargs.get("om", {}))
        if config is not None else None
    )

    def entity_block(entity: Entity, manager: KgManager, label: str) -> str:
        # fenced code block for the shape: self-delimiting regardless of
        # surrounding indentation, and semantically fitting since pseudo-SHEx
        # is itself a small code-like syntax
        nonlocal shape_shown
        block = f"**{label}**: {format_entity(entity)}\n"
        shape_text = get_entity_shape_text(manager, entity.identifier, known)
        show_shape = shape_text is not None and "no properties found" not in shape_text
        if show_shape:
            shape_shown = True
            block += f"\n```shape: \n{shape_text}\n```\n"
        if context_cfg is not None and not (show_shape and context_cfg.skip_if_shape):
            assert config is not None
            try:
                context = get_entity_context_text(
                    manager, entity, context_cfg, config, known
                )
            except Exception as e:
                # never lose a run because of the context
                context = None
                manager.logger.warning(
                    f"Failed to get the context of {entity.identifier}: {e}"
                )
            if context is not None:
                block += f"\nContext:\n{context}\n"
        return block

    # source entities with candidates, by default from string matching (om_pretask);
    # other candidate generators describe themselves via om.candidates_description
    len_pc = len(task_input.potential_correspondences)
    if len_pc > 0:
        filler = "entities" if len_pc > 1 else "entity"
        found_by = "simple string matching found entities with the same normalized name"
        if config is not None:
            found_by = config.task_kwargs.get("om", {}).get("candidates_description", found_by)
        instructions += f"""\
For the following {filler} from the source ontology {task_input.source_kg}, \
{found_by} in the target ontology {task_input.target_kg}. They are only \
candidates: decide which relation, if any, holds for each of them, and search for \
further correspondences as for any other source entity:
"""
        for corr in task_input.potential_correspondences:
            block_num += 1
            instructions += f"### Entity {block_num}\n"
            instructions += entity_block(corr.source_entity, source_manager, "Source")
            for j, c in enumerate(corr.candidates, start=1):
                instructions += entity_block(c, target_manager, f"Candidate {j}")
            instructions += "\n"

    # source entities without candidates
    len_corrs = len(task_input.unmatched_entities)
    if len_corrs > 0:
        filler_1 = "entities" if len_corrs > 1 else "entity"
        if allows_many(state.profile):
            instructions += f"""\
Find all correspondences between the following {filler_1} from the source ontology \
{task_input.source_kg} and entities from the target ontology {task_input.target_kg}, \
using any of the given relations, or verify that none exist:
"""
        else:
            filler_2 = "entities"
            filler_3 = "they have no appropriate matches"
            if len_corrs == 1:
                filler_2 = "an entity"
                filler_3 = "it has no appropriate match"
            instructions += f"""\
Align the following {filler_1} from the source ontology {task_input.source_kg} \
with {filler_2} from the target ontology {task_input.target_kg} or verify {filler_3}:
"""
        if task_input.description:
            instructions += f"Context: {task_input.description}\n\n"

        for entity in task_input.unmatched_entities:
            block_num += 1
            instructions += f"### Entity {block_num}\n"
            instructions += entity_block(entity, source_manager, "Source")
            instructions += "\n"

    if shape_shown:
        instructions += f"\n{SHAPE_CAVEAT}\n"

    return instructions


def input_and_state(
    input: Any,
    config: GraspConfig,
    managers: list[KgManager],
    known: set[str],
) -> tuple[str, AlignmentState]:
    try:
        task_input = AlignmentTaskInput(**input)
    except Exception as e:
        raise ValueError("OM task input must match AlignmentTaskInput schema") from e

    profile = get_relation_profile(config)
    state = AlignmentState(profile)
    state.task_input = task_input
    for correspondence in state.task_input.input_alignment:
        relation = profile.get(correspondence.relation)
        if relation is None:
            raise ValueError(
                f"Input correspondence {correspondence.key} has relation "
                f"'{correspondence.relation}', which is not part of the relation "
                f"profile {profile.name}"
            )
        # input correspondences win over conflicting ones in the shared state,
        # by replacing the conflicting ones, without asking the LLM
        if relation.symmetric and relation.transitive:
            state.discard_correspondences_of(correspondence.key[0], CorrespondenceSide.SOURCE, relation.iri)
            state.discard_correspondences_of(correspondence.key[1], CorrespondenceSide.TARGET, relation.iri)
        state.add_correspondence(correspondence)

    instructions = input_instructions(task_input, state, managers, known, config)
    return instructions, state


RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"

# relations reducible via the ontology hierarchies, with the relations that can
# imply them: (a < b) follows from (a' r b') with r in {<, =}, a below-or-equal
# a' in the source and b' below-or-equal b in the target hierarchy; '>' the
# other way round. '=' never follows from hierarchies alone.
REDUCIBLE: dict[str, tuple[str, ...]] = {"<": ("<", "="), ">": (">", "=")}


def hierarchy_predicate(om_kwargs: dict, kg: str) -> str:
    """
    The predicate forming the hierarchy of `kg`, e.g. rdfs:subClassOf or
    skos:broader. `hierarchy_predicate` in the om task kwargs is either one
    IRI for all knowledge graphs or a mapping from knowledge graph name to IRI.
    """
    pred = om_kwargs.get("hierarchy_predicate", RDFS_SUBCLASS_OF)
    if isinstance(pred, dict):
        return pred.get(kg, RDFS_SUBCLASS_OF)
    return pred


def fetch_ancestors(
    manager: KgManager,
    iris: set[str],
    predicate: str,
    chunk_size: int = 200,
) -> dict[str, set[str]]:
    """Transitive ancestors of each of the given IRIs along `predicate`."""
    ancestors: dict[str, set[str]] = {iri: set() for iri in iris}
    wrapped_pred = prepare_identifier_for_sparql(predicate, manager.iri_literal_parser)
    ordered = sorted(iris)
    for i in range(0, len(ordered), chunk_size):
        values = " ".join(
            prepare_identifier_for_sparql(iri, manager.iri_literal_parser)
            for iri in ordered[i:i + chunk_size]
        )
        sparql = (
            f"SELECT ?c ?p WHERE {{ VALUES ?c {{ {values} }} "
            f"?c {wrapped_pred}+ ?p . FILTER(isIRI(?p)) }}"
        )
        result = manager.execute_sparql(sparql)
        assert isinstance(result, SelectResult)
        for row in result.rows():
            if "c" in row and "p" in row:
                ancestors.setdefault(row["c"].identifier(), set()).add(row["p"].identifier())
    return ancestors


def reduce_alignment(
    correspondences: list[Correspondence],
    source_ancestors: dict[str, set[str]],
    target_ancestors: dict[str, set[str]],
) -> list[Correspondence]:
    """
    Removes '<' and '>' correspondences that are implied by another
    correspondence together with the source and target hierarchies. If two
    correspondences imply each other (e.g. via cyclic hierarchies), the one
    with the smaller key is kept.
    """
    def leq(ancestors: dict[str, set[str]], x: str, y: str) -> bool:
        return x == y or y in ancestors.get(x, set())

    def implies(d: Correspondence, c: Correspondence) -> bool:
        (a, b, r), (a2, b2, r2) = c.key, d.key
        if (a, b) == (a2, b2) or r2 not in REDUCIBLE.get(r, ()):
            return False
        if r == "<":
            return leq(source_ancestors, a, a2) and leq(target_ancestors, b2, b)
        return leq(source_ancestors, a2, a) and leq(target_ancestors, b, b2)

    def redundant(c: Correspondence) -> bool:
        if c.relation not in REDUCIBLE:
            return False
        return any(
            implies(d, c) and not (implies(c, d) and c.key < d.key)
            for d in correspondences if d is not c
        )

    return [c for c in correspondences if not redundant(c)]


def reduced_correspondences(
    state: AlignmentState,
    managers: list[KgManager],
    om_kwargs: dict,
) -> list[Correspondence]:
    correspondences = list(state.correspondences.values())
    if not any(c.relation in REDUCIBLE for c in correspondences):
        return correspondences
    assert state.task_input is not None
    sides = []
    for kg, idx in [(state.task_input.source_kg, 0), (state.task_input.target_kg, 1)]:
        manager, _ = find_manager(managers, kg)
        iris = {c.key[idx] for c in correspondences}
        sides.append(fetch_ancestors(manager, iris, hierarchy_predicate(om_kwargs, kg)))
    return reduce_alignment(correspondences, sides[0], sides[1])


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
    profile: RelationProfile,
    alignment_form: str = "raw",
    scope_description: str | None = None,
    task_instructions: str | None = None,
) -> str:
    task_info = ""
    if profile.description:
        task_info += (
            "The system was provided the following explanation of the relations:\n"
            f"{profile.description.strip()}\n\n"
        )
    if task_instructions:
        task_info += (
            "The system was provided the following task-specific instructions:\n"
            f"{task_instructions}\n\n"
        )
    return f"""\
You are an expert ontology matching judge. Your task is to evaluate if the assistant has correctly identified and set correspondences between the entities.

The system has access to the following knowledge graphs:
{format_kgs(managers, kg_notes)}

The system was provided the following notes across all knowledge graphs:
{format_notes(notes)}

{task_info}The system was provided the following rules to follow:
{format_list(rules(profile, alignment_form, scope_description))}

Provide your feedback with the give_feedback function.\
"""


def feedback_instructions(inputs: list[str], output: dict) -> str:
    assert inputs, "At least one input is required for feedback"

    if len(inputs) > 1:
        prompt = (
            "Previous inputs:\n" + "\n\n".join(i.strip() for i in inputs[:-1]) + "\n\n"
        )
    else:
        prompt = ""

    prompt += f"Input:\n{inputs[-1].strip()}"
    prompt += f"Current Alignment State:\n{output['formatted']}\n\n"
    prompt += (
        "Does the current state fulfill the task? If the system has" +
        "not finished the matching process yet, provide prompts describing" +
        "the concrete next step for the system to continue and fininalize the matching.")
    return prompt


class OmTask(GraspTask, FeedbackTask):
    name = "om"

    def setup(self, input: Any) -> str:
        instructions, self.state = input_and_state(
            input, self.config, self.managers, self.known
        )
        return instructions

    def system_information(self) -> str:
        return system_information(
            get_relation_profile(self.config),
            load_task_instructions(self.config.task_kwargs.get("om", {})),
        )

    def rules(self) -> list[str]:
        om_kwargs = self.config.task_kwargs.get("om", {})
        return rules(
            get_relation_profile(self.config),
            om_kwargs.get("alignment_form", "raw"),
            om_kwargs.get("scope", {}).get("description"),
        )

    def function_definitions(self) -> list[dict]:
        return functions(self.managers, get_relation_profile(self.config))

    def output(self, messages: list[Message]) -> dict:
        om_kwargs = self.config.task_kwargs.get("om", {})
        if om_kwargs.get("alignment_form", "raw") != "reduced":
            return self.state.to_dict()
        try:
            reduced = reduced_correspondences(self.state, self.managers, om_kwargs)
        except Exception as e:
            # never lose a run because of the reduction
            output = self.state.to_dict()
            output["reduction_error"] = str(e)
            return output
        return self.state.to_dict(reduced)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
            ) -> str:
        return call_function(
            self.config,
            self.managers,
            fn_name,
            fn_args,
            known,
            self.state,
            example_indices,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    @property
    def default_input_field(self) -> str | None:
        return None

    def feedback_system_message(
        self, kg_notes: dict[str, list[str]], notes: list[str]
    ) -> str:
        om_kwargs = self.config.task_kwargs.get("om", {})
        return feedback_system_message(
            self.managers, kg_notes, notes, get_relation_profile(self.config),
            om_kwargs.get("alignment_form", "raw"),
            om_kwargs.get("scope", {}).get("description"),
            load_task_instructions(om_kwargs),
        )

    def feedback_instructions(self, inputs: list[str], output: dict) -> str:
        return feedback_instructions(inputs, output)
