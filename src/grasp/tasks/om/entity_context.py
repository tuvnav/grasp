from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

from grasp.manager import KgManager
from grasp.sparql.types import Binding
from grasp.tasks.om.model import (
    IRI, VOCABULARY, BindingType, BlankNodeTree, EntityContextConfig, local_name, sparql_path,
)
from grasp.tasks.om.render import clip, display_iri, render_blank_node
from grasp.tasks.om.retrieve import (
    retrieve_blank_nodes, retrieve_outgoing_edges, select_object_iris, select_rows, to_sparql_term,
    values_blocks,
)

@dataclass
class IncomingEdgesSummary:
    """One for each predicate"""
    predicate: IRI
    count: int
    sample: list[IRI]  # subjects, rarest first


class EdgeType(StrEnum):
    NAME = "name"
    SYNONYM = "synonym"
    ANNOTATION = "annotation"
    TYPE = "type"
    CLASS_EXPRESSION = "class_expression"
    OTHER_RELATION = "other_relation"


@dataclass(frozen=True)
class Edge:
    """An outgoing edge of the entity, sorted into a category."""
    type: EdgeType
    predicate: IRI
    obj: Binding


@dataclass
class TypeUsage:
    """Entity is used as a type/class"""
    instance_count: int
    shape: str | None = None


@dataclass
class PredicateUsage:
    """Entity is used as a predicate/property"""
    triple_count: int
    subject_types: list[tuple[IRI, int]]  # most common subject types with count
    object_types: list[tuple[IRI, int]]  # most common object types with count
    num_literal_objects: int
    examples: list[tuple[Binding, Binding]]  # subject - object pairs


@dataclass
class EntityContext:
    iri: IRI
    # other relations of a predicate are ordered rarest object first
    outgoing_edges: list[Edge]
    incoming_edges: list[IncomingEdgesSummary]
    used_as_type: TypeUsage | None
    used_as_predicate: PredicateUsage | None
    blank_nodes: BlankNodeTree = field(default_factory=dict)
    # labels of all IRIs that are shown
    labels: dict[IRI, str] = field(default_factory=dict)


def find_helper_nodes(
        manager: KgManager,
        iris: Iterable[IRI],
        cfg: EntityContextConfig,
        ) -> set[IRI]:
    """
    The IRIs that only carry a value: at least one name, and otherwise only
    name, synonym or type edges (e.g. definition and synonym nodes in the OAEI
    anatomy ontologies). Their label stands for them like a literal. IRIs
    without any name are no helper nodes, there would be nothing to show.
    """
    plain_preds = [*cfg.name_predicates, *cfg.synonym_predicates, *cfg.type_predicates]
    not_plain = ", ".join(f"<{p}>" for p in plain_preds)
    named: set[IRI] = set()
    with_other_edges: set[IRI] = set()
    for values in values_blocks(iris):
        named |= select_object_iris(
            manager,
            f"SELECT DISTINCT ?o WHERE {{ VALUES ?o {{ {values} }} "
            f"?o {sparql_path(cfg.name_predicates)} ?name }}",
        )
        with_other_edges |= select_object_iris(
            manager,
            f"SELECT DISTINCT ?o WHERE {{ VALUES ?o {{ {values} }} "
            f"?o ?p ?x FILTER(?p NOT IN ({not_plain})) }}",
        )
    return named - with_other_edges


def categorize_outgoing_edge(
        edge: tuple[Binding, Binding],
        cfg: EntityContextConfig,
        helper_nodes: set[IRI],
        ) -> Edge:
    """
    Sorts one outgoing edge into a category. Rules 1-3 look at the predicate,
    rules 4-7 at the object; the first matching rule wins.
    """
    pred, obj = edge
    predicate = IRI(pred.value)

    # 1-3: by predicate (types before helper nodes: classes often only carry a label)
    if predicate in cfg.name_predicates:
        edge_type = EdgeType.NAME
    elif predicate in cfg.synonym_predicates:
        edge_type = EdgeType.SYNONYM
    elif predicate in cfg.type_predicates:
        edge_type = EdgeType.TYPE
    # 4-7: by object
    elif obj.typ == BindingType.URI and IRI(obj.value) in helper_nodes:
        edge_type = EdgeType.ANNOTATION
    elif obj.typ == BindingType.LITERAL:
        edge_type = EdgeType.ANNOTATION
    elif obj.typ == BindingType.BNODE:
        edge_type = EdgeType.CLASS_EXPRESSION
    else:
        edge_type = EdgeType.OTHER_RELATION

    return Edge(edge_type, predicate, obj)


def categorize_outgoing_edges(
        manager: KgManager,
        entity: IRI,
        cfg: EntityContextConfig,
        ) -> list[Edge]:
    """Retrieves the outgoing edges of the entity and sorts them into categories."""
    edges = retrieve_outgoing_edges(manager, entity)
    # helper node candidates: IRI objects of edges that rules 1-3 do not sort by predicate
    sorted_by_predicate = {*cfg.name_predicates, *cfg.synonym_predicates, *cfg.type_predicates}
    candidates = {
        IRI(obj.value) for pred, obj in edges
        if obj.typ == BindingType.URI and pred.value not in sorted_by_predicate
    }
    helper_nodes = find_helper_nodes(manager, candidates, cfg)
    return [categorize_outgoing_edge(edge, cfg, helper_nodes) for edge in edges]


# ---------------------------------------------------------------- gathering

def order_by_rarity(manager: KgManager, edges: list[Edge], cfg: EntityContextConfig) -> list[Edge]:
    """
    Orders the other relations of each predicate with more than k objects by
    how many edges of that predicate point to the object, rarest first: objects
    many entities share (e.g. the most linked wiki page) say little about this one.
    """
    by_pred: dict[IRI, list[Edge]] = {}
    for edge in edges:
        if edge.type == EdgeType.OTHER_RELATION and edge.obj.typ == BindingType.URI:
            by_pred.setdefault(edge.predicate, []).append(edge)
    rank: dict[tuple[IRI, str], int] = {}
    for pred, pred_edges in by_pred.items():
        if len(pred_edges) <= cfg.k_values:
            continue
        for values in values_blocks(IRI(e.obj.value) for e in pred_edges):
            for row in select_rows(
                manager,
                f"SELECT ?o (COUNT(?x) AS ?n) WHERE {{ VALUES ?o {{ {values} }} ?x <{pred}> ?o }} GROUP BY ?o",
            ):
                rank[(pred, row["o"].value)] = int(row["n"].value)
    # stable sort: edges without a rank keep their order
    return sorted(edges, key=lambda e: rank.get((e.predicate, e.obj.value), 0))


def retrieve_incoming_edges(
        manager: KgManager, entity: IRI, cfg: EntityContextConfig,
        ) -> list[IncomingEdgesSummary]:
    """Incoming edges per predicate (type predicates excluded, see TypeUsage),
    most frequent predicates first, with the k rarest subjects."""
    term = to_sparql_term(manager, entity)
    not_types = ", ".join(f"<{p}>" for p in cfg.type_predicates)
    rows = select_rows(
        manager,
        # two separate filters: the combination with && triggers an internal QLever error
        f"SELECT ?p (COUNT(?s) AS ?n) WHERE {{ ?s ?p {term} FILTER(isIRI(?s)) FILTER(?p NOT IN ({not_types})) }} "
        f"GROUP BY ?p ORDER BY DESC(?n) LIMIT {cfg.max_incoming_predicates}",
    )
    summaries = []
    for row in rows:
        pred = IRI(row["p"].value)
        sample = select_rows(
            manager,
            f"SELECT ?s (COUNT(?y) AS ?d) WHERE {{ ?s <{pred}> {term} FILTER(isIRI(?s)) "
            f"OPTIONAL {{ ?y ?q ?s }} }} GROUP BY ?s ORDER BY ?d ?s LIMIT {cfg.k_values}",
        )
        summaries.append(IncomingEdgesSummary(pred, int(row["n"].value), [IRI(r["s"].value) for r in sample]))
    return summaries


def retrieve_type_usage(manager: KgManager, entity: IRI, cfg: EntityContextConfig) -> TypeUsage | None:
    term = to_sparql_term(manager, entity)
    rows = select_rows(
        manager, f"SELECT (COUNT(?x) AS ?n) WHERE {{ ?x {sparql_path(cfg.type_predicates)} {term} }}"
    )
    count = int(rows[0]["n"].value) if rows else 0
    return TypeUsage(count) if count else None


def retrieve_predicate_usage(manager: KgManager, entity: IRI, cfg: EntityContextConfig) -> PredicateUsage | None:
    term = to_sparql_term(manager, entity)
    rows = select_rows(manager, f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s {term} ?o }}")
    count = int(rows[0]["n"].value) if rows else 0
    if not count:
        return None
    types = sparql_path(cfg.type_predicates)

    def top_types(position: str) -> list[tuple[IRI, int]]:
        return [
            (IRI(r["t"].value), int(r["n"].value)) for r in select_rows(
                manager,
                f"SELECT ?t (COUNT(*) AS ?n) WHERE {{ ?s {term} ?o . ?{position} {types} ?t }} "
                f"GROUP BY ?t ORDER BY DESC(?n) LIMIT 3",
            )
        ]

    literals = select_rows(manager, f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s {term} ?o FILTER(isLiteral(?o)) }}")
    examples = select_rows(manager, f"SELECT ?s ?o WHERE {{ ?s {term} ?o }} LIMIT 3")
    return PredicateUsage(
        triple_count=count,
        subject_types=top_types("s"),
        object_types=top_types("o"),
        num_literal_objects=int(literals[0]["n"].value) if literals else 0,
        examples=[(r["s"], r["o"]) for r in examples],
    )


def retrieve_labels(manager: KgManager, iris: Iterable[IRI], cfg: EntityContextConfig) -> dict[IRI, str]:
    """One name per IRI (the first found), for all IRIs at once."""
    labels: dict[IRI, str] = {}
    for values in values_blocks(iri for iri in iris if not iri.startswith(VOCABULARY)):
        for row in select_rows(
            manager,
            f"SELECT ?o ?l WHERE {{ VALUES ?o {{ {values} }} ?o {sparql_path(cfg.name_predicates)} ?l }}",
        ):
            labels.setdefault(IRI(row["o"].value), row["l"].value)
    return labels


def shown_iris(ctx: EntityContext) -> set[IRI]:
    """All IRIs whose label may be shown."""
    iris = {IRI(e.obj.value) for e in ctx.outgoing_edges if e.obj.typ == BindingType.URI}
    for edges in ctx.blank_nodes.values():
        for pred, obj in edges:
            iris.add(pred)
            if obj.typ == BindingType.URI:
                iris.add(IRI(obj.value))
    for summary in ctx.incoming_edges:
        iris.update(summary.sample)
    if ctx.used_as_predicate:
        usage = ctx.used_as_predicate
        iris.update(t for t, _ in usage.subject_types + usage.object_types)
        for s, o in usage.examples:
            iris.update(IRI(b.value) for b in (s, o) if b.typ == BindingType.URI)
    return iris


def build_entity_context(manager: KgManager, entity: IRI, cfg: EntityContextConfig) -> EntityContext:
    """Gathers everything shown about the entity; all SPARQL queries happen here."""
    edges = order_by_rarity(manager, categorize_outgoing_edges(manager, entity, cfg), cfg)
    has_blank_nodes = any(e.type == EdgeType.CLASS_EXPRESSION for e in edges)
    ctx = EntityContext(
        iri=entity,
        outgoing_edges=edges,
        incoming_edges=retrieve_incoming_edges(manager, entity, cfg),
        used_as_type=retrieve_type_usage(manager, entity, cfg),
        used_as_predicate=retrieve_predicate_usage(manager, entity, cfg),
        blank_nodes=retrieve_blank_nodes(manager, entity, cfg.bn_resolution_max_depth) if has_blank_nodes else {},
    )
    ctx.labels = retrieve_labels(manager, shown_iris(ctx), cfg)
    return ctx


# ---------------------------------------------------------------- rendering

def value_text(obj: Binding, ctx: EntityContext, cfg: EntityContextConfig) -> str:
    """Text of a name or annotation value: a literal, or the label of a helper node."""
    if obj.typ == BindingType.URI:
        return ctx.labels.get(IRI(obj.value)) or display_iri(obj.value, ctx.labels)
    text = clip(obj.value, cfg.max_chars_per_literal)
    return f"{text}@{obj.lang}" if obj.lang else text


def render_entity_context(
        ctx: EntityContext,
        cfg: EntityContextConfig,
        entity_space: tuple[str, ...] | None = None,
        ) -> str:
    """Text for the LLM, sections in a fixed order of importance, cut at max_chars.
    Pure: uses only the gathered context, no SPARQL."""
    lines: list[str] = []
    by_type: dict[EdgeType, list[Edge]] = {}
    for edge in ctx.outgoing_edges:
        by_type.setdefault(edge.type, []).append(edge)

    def grouped(edge_type: EdgeType) -> dict[IRI, list[Edge]]:
        groups: dict[IRI, list[Edge]] = {}
        for edge in by_type.get(edge_type, []):
            groups.setdefault(edge.predicate, []).append(edge)
        return groups

    # names and synonyms
    for edge_type, title in [(EdgeType.NAME, "labels"), (EdgeType.SYNONYM, "synonyms")]:
        texts = list(dict.fromkeys(value_text(e.obj, ctx, cfg) for e in by_type.get(edge_type, [])))
        if texts:
            lines.append(f"{title}: " + " | ".join(texts))

    # annotations: identical texts of several predicates once, longest first
    predicates_of: dict[str, list[str]] = {}
    for edge in by_type.get(EdgeType.ANNOTATION, []):
        text = value_text(edge.obj, ctx, cfg)
        preds = predicates_of.setdefault(text, [])
        if local_name(edge.predicate) not in preds:
            preds.append(local_name(edge.predicate))
    by_preds: dict[str, list[str]] = {}
    for text, preds in predicates_of.items():
        by_preds.setdefault(", ".join(preds), []).append(text)
    for preds, texts in sorted(by_preds.items(), key=lambda kv: -max(map(len, kv[1]))):
        texts.sort(key=len, reverse=True)
        more = f" (+{len(texts) - cfg.k_values} more)" if len(texts) > cfg.k_values else ""
        lines.append(f"{preds}: " + " | ".join(f'"{t}"' for t in texts[:cfg.k_values]) + more)

    # types
    types = list(dict.fromkeys(display_iri(e.obj.value, ctx.labels) for e in by_type.get(EdgeType.TYPE, [])))
    if types:
        lines.append("types: " + ", ".join(types))

    # class expressions
    for pred, edges in grouped(EdgeType.CLASS_EXPRESSION).items():
        expressions = [render_blank_node(e.obj.value, ctx.blank_nodes, ctx.labels) for e in edges]
        lines.append(f"{local_name(pred)}: " + "; ".join(dict.fromkeys(expressions)))

    # other relations: predicates with few objects first; objects with a label
    # first (rare objects without one are often files or stubs), then rarest first
    only_hidden = []
    for pred, edges in sorted(grouped(EdgeType.OTHER_RELATION).items(), key=lambda kv: len(kv[1])):
        objects = list(dict.fromkeys(e.obj.value for e in edges))
        shown = [
            o for o in objects
            if entity_space is None or o.startswith(entity_space)
            or o.startswith(VOCABULARY) or IRI(o) in ctx.labels
        ]
        hidden = len(objects) - len(shown)
        if not shown:
            only_hidden.append(f"{local_name(pred)} ({hidden})")
            continue
        shown.sort(key=lambda o: IRI(o) not in ctx.labels and not o.startswith(VOCABULARY))
        notes = []
        if len(shown) > cfg.k_values:
            notes.append(f"+{len(shown) - cfg.k_values} more, labeled and rarest shown")
        if hidden:
            notes.append(f"{hidden} unlabeled values outside the entity space")
        values = ", ".join(display_iri(o, ctx.labels) for o in shown[:cfg.k_values])
        lines.append(
            f"{local_name(pred)} ({len(objects)}): {values}" + (f" [{'; '.join(notes)}]" if notes else "")
        )
    if only_hidden:
        lines.append("only unlabeled values outside the entity space: " + ", ".join(only_hidden))

    # incoming edges
    for summary in ctx.incoming_edges:
        subjects = ", ".join(display_iri(s, ctx.labels) for s in summary.sample)
        lines.append(f"incoming {local_name(summary.predicate)} ({summary.count}): {subjects}")

    # usage by other entities
    if ctx.used_as_type:
        lines.append(f"used as type of {ctx.used_as_type.instance_count} entities")
        if ctx.used_as_type.shape:
            lines.append(ctx.used_as_type.shape)
    if ctx.used_as_predicate:
        usage = ctx.used_as_predicate

        def type_counts(types: list[tuple[IRI, int]]) -> str:
            return ", ".join(f"{display_iri(t, ctx.labels)} {n}" for t, n in types) or "untyped"

        def example(b: Binding) -> str:
            if b.typ == BindingType.URI:
                return display_iri(b.value, ctx.labels)
            return "_" if b.typ == BindingType.BNODE else f'"{clip(b.value, cfg.max_chars_per_literal)[:60]}"'

        examples = "; ".join(f"{example(s)} -> {example(o)}" for s, o in usage.examples)
        lines.append(
            f"used as predicate in {usage.triple_count} triples; subject types: "
            f"{type_counts(usage.subject_types)}; object types: {type_counts(usage.object_types)}; "
            f"{usage.num_literal_objects} literal objects; e.g. {examples}"
        )

    # budget: sections are ordered by importance, so the least important are cut
    out = [f"Entity {display_iri(ctx.iri, ctx.labels)} ({ctx.iri})"]
    used = len(out[0])
    for i, line in enumerate(lines):
        if used + len(line) + 3 > cfg.max_chars:
            out.append(f"  … ({len(lines) - i} more lines omitted)")
            break
        out.append(f"  {line}")
        used += len(line) + 3
    return "\n".join(out)


def entity_context(
        manager: KgManager,
        entity: IRI,
        cfg: EntityContextConfig,
        entity_space: tuple[str, ...] | None = None,
        ) -> str:
    """The context of an entity as text for the LLM."""
    return render_entity_context(build_entity_context(manager, entity, cfg), cfg, entity_space)
