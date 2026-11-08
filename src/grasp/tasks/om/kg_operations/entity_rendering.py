"""
Entities shown to the LLM, the same in all stages (EntityDisplay.show), in
three views of one concept (EntityView; presets SHORT_VIEW, MIDDLE_VIEW,
LONG_VIEW):
    line     'large intestine <mouse:MA_1>', optionally with further names and
             synonyms: '... (names: "Dickdarm"@de; synonyms: "colon", +1 more)'
    context  the line, then everything the KG says about the entity (names,
             synonyms, definitions, types, class expressions, relations,
             incoming edges, usage as type or predicate), cut at a budget

What the KG means is taken from its profile (the roles: names, synonyms,
definitions, and the membership relations as types), so it works on any
schema; a profile without a role shows nothing for it (no fallback to
rdfs:label and co., which would hide gaps of the profile). Two steps:
prefetch retrieves (SPARQL, in batches, cached for the whole task), show only
formats what was retrieved.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Iterable

from grasp.manager import KgManager
from grasp.sparql.types import Binding
from grasp.tasks.om.data_models.patterns import StructureRelationKind
from grasp.tasks.om.data_models.profile import KgProfile, RoleName, Roles
from grasp.tasks.om.foundation.types import IRI, BindingType, BlankNodeTree
from grasp.tasks.om.foundation.vocabulary import (
    OWL_ON_PROPERTY,
    RDF_FIRST,
    RDF_REST,
    RDF_TYPE,
    VOCABULARY,
    VOCABULARY_PREFIXES,
    local_name,
)
from grasp.tasks.om.kg_operations.value_retrieval import value_branches
from grasp.tasks.om.sparql_access.execution import select_object_iris, select_rows
from grasp.tasks.om.sparql_access.lookups import retrieve_blank_nodes, retrieve_outgoing_edges, to_sparql_term
from grasp.tasks.om.sparql_access.query_building import down_pushed_union, sparql_path
from grasp.tasks.om.sparql_access.values import values_blocks

# ---------------------------------------------------------------- building blocks

# restriction predicates and their keyword in Manchester syntax
RESTRICTIONS = {
    IRI("http://www.w3.org/2002/07/owl#someValuesFrom"): "some",
    IRI("http://www.w3.org/2002/07/owl#allValuesFrom"): "only",
    IRI("http://www.w3.org/2002/07/owl#hasValue"): "value",
}

# levels of nested blank nodes resolved below an entity (class expressions);
# only a guard, render_blank_node also stops at cycles
BLANK_NODE_MAX_DEPTH = 6

# roles retrieved for every shown entity, and for the entity of a context
LINE_ROLES: tuple[RoleName, ...] = ("names", "synonyms")
CONTEXT_ROLES: tuple[RoleName, ...] = ("names", "synonyms", "definitions")


def label_with_reference(label: str | None, reference: str) -> str:
    """The one shape of a labelled entity: 'Label <reference>', or '<reference>'
    without a label."""
    return f"{label} <{reference}>" if label else f"<{reference}>"


def clip(text: str, max_chars: int) -> str:
    """Whitespace collapsed, cut at max_chars."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …"


def vocabulary_name(iri: str) -> str | None:
    """owl:Class for an IRI of the standard vocabulary, else None."""
    for namespace, prefix in VOCABULARY_PREFIXES.items():
        if iri.startswith(namespace):
            return f"{prefix}:{iri[len(namespace):]}"
    return None


def render_blank_node(
        node: str,
        tree: BlankNodeTree,
        show_iri: Callable[[str], str],
        seen: frozenset[str] = frozenset(),
        ) -> str:
    """A blank node as a class expression: RDF lists as [a, b], restrictions
    as (property some filler), anything else as (predicate value; ...)."""
    if node in seen:
        return "(…)"
    seen = seen | {node}
    by_pred: dict[IRI, list[Binding]] = {}
    for pred, obj in tree.get(node, []):
        by_pred.setdefault(pred, []).append(obj)

    def render(obj: Binding) -> str:
        if obj.typ == BindingType.BNODE:
            return render_blank_node(obj.value, tree, show_iri, seen)
        if obj.typ == BindingType.URI:
            return show_iri(obj.value)
        return f'"{obj.value}"'

    if RDF_FIRST in by_pred:
        items: list[str] = []
        visited: set[str] = set()
        current: str | None = node
        while current is not None and current not in visited:
            visited.add(current)
            current_edges = dict(tree.get(current, []))
            first = current_edges.get(RDF_FIRST)
            if first is None:
                break
            items.append(render(first))
            rest = current_edges.get(RDF_REST)
            current = rest.value if rest is not None and rest.typ == BindingType.BNODE else None
        return "[" + ", ".join(items) + "]"
    if OWL_ON_PROPERTY in by_pred:
        prop = render(by_pred[OWL_ON_PROPERTY][0])
        for pred, keyword in RESTRICTIONS.items():
            if pred in by_pred:
                return f"({prop} {keyword} {render(by_pred[pred][0])})"
    parts = [
        f"{show_iri(pred)} {', '.join(render(o) for o in objs)}"
        for pred, objs in by_pred.items()
        if pred != RDF_TYPE  # e.g. owl:Restriction, implied by the rendering
    ]
    return "(" + "; ".join(parts) + ")"


# ---------------------------------------------------------------- views

@dataclass(frozen=True)
class EntityView:
    """
    What is shown of an entity.
    Line: iri: its reference (prefixed IRI if a prefix is known); num_names:
    names shown, the first as label (0: none); num_synonyms: synonyms shown.
    Values that are not shown are counted.
    context: the line, then the context of the entity, with: num_values: values
    shown per predicate (and definitions); max_chars_per_literal; max_incoming_predicates;
    max_chars: budget of the whole context (the least important lines are cut).
    """
    iri: bool = True
    num_names: int = 1
    num_synonyms: int = 0
    context: bool = False
    num_values: int = 5
    max_chars_per_literal: int = 300
    max_incoming_predicates: int = 8
    max_chars: int = 2000


SHORT_VIEW = EntityView()                                # Label <mouse:MA_1>
MIDDLE_VIEW = EntityView(num_names=3, num_synonyms=3)   # ... (names: ...; synonyms: ...)
LONG_VIEW = EntityView(num_names=3, num_synonyms=10, context=True)


@dataclass(frozen=True, order=True)
class NameValue:
    """A name, synonym or definition of an entity; ordered without language
    first, then by language and text (so that the label is the same in every run)."""
    language: str
    text: str

    def render(self, max_chars: int | None = None) -> str:
        text = clip(self.text, max_chars) if max_chars is not None else self.text
        return f'"{text}"' + (f"@{self.language}" if self.language else "")


def shown_values(title: str, values: list[NameValue], num_shown: int, max_chars: int | None = None) -> str | None:
    """'title: "a", "b", +2 more', None if nothing is to be shown."""
    if num_shown <= 0 or not values:
        return None
    shown = ", ".join(value.render(max_chars) for value in values[:num_shown])
    more = f", +{len(values) - num_shown} more" if len(values) > num_shown else ""
    return f"{title}: {shown}{more}"


# ---------------------------------------------------------------- what the profile says

def plain_predicates(roles: Roles, role_names: Iterable[RoleName]) -> set[IRI]:
    """The plain predicates of the roles (not their SPARQL patterns)."""
    patterns = [roles.predicate_pattern_by_name(role_name) for role_name in role_names]
    return {predicate for pattern in patterns if pattern is not None for predicate in pattern.predicates}


def membership_predicates(roles: Roles) -> list[IRI]:
    """The plain predicates of the membership relations: the types of an entity."""
    return sorted({
        predicate for relation in roles.structure if relation.kind == StructureRelationKind.MEMBERSHIP
        for predicate in relation.predicate_pattern.predicates
    })


# ---------------------------------------------------------------- context: data

@dataclass
class IncomingEdgesSummary:
    """One for each predicate"""
    predicate: IRI
    count: int
    sample: list[IRI]  # subjects, rarest first


class EdgeType(StrEnum):
    ROLE_VALUE = "role_value"   # a plain predicate of a shown role: shown from the role values
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
    # IRIs that only carry a value (e.g. synonym nodes): their name stands for them
    helper_nodes: set[IRI] = field(default_factory=set)


# ---------------------------------------------------------------- context: gathering

def find_helper_nodes(manager: KgManager, iris: Iterable[IRI], roles: Roles) -> set[IRI]:
    """
    The IRIs that only carry a value: at least one name, and otherwise only
    edges of plain predicates of the names, synonyms and membership roles (e.g.
    definition and synonym nodes in the OAEI anatomy ontologies). Their name
    stands for them like a literal. Patterns of the profile cannot be checked
    here, only plain predicates.
    """
    names = roles.predicate_pattern_by_name("names")
    if names is None:
        return set()
    plain = sorted(plain_predicates(roles, LINE_ROLES) | set(membership_predicates(roles)))
    named: set[IRI] = set()
    with_other_edges: set[IRI] = set()
    for values in values_blocks(iris):
        has_name = down_pushed_union(f"VALUES ?entity {{ {values} }}", names.branches_for_union())
        named |= {IRI(row["entity"].value)
                  for row in select_rows(manager, f"SELECT DISTINCT ?entity WHERE {{\n{has_name}\n}}")}
        not_plain = f"FILTER(?p NOT IN ({', '.join(f'<{p}>' for p in plain)}))" if plain else ""
        with_other_edges |= select_object_iris(
            manager, f"SELECT DISTINCT ?o WHERE {{ VALUES ?o {{ {values} }} ?o ?p ?x {not_plain} }}",
        )
    return named - with_other_edges


def categorize_outgoing_edge(
        edge: tuple[Binding, Binding],
        role_predicates: set[IRI],
        type_predicates: set[IRI],
        helper_nodes: set[IRI],
        ) -> Edge:
    """
    Sorts one outgoing edge into a category. Rules 1-2 look at the predicate,
    rules 3-5 at the object; the first matching rule wins.
    """
    pred, obj = edge
    predicate = IRI(pred.value)
    if predicate in role_predicates:
        edge_type = EdgeType.ROLE_VALUE
    elif predicate in type_predicates:
        edge_type = EdgeType.TYPE
    elif obj.typ == BindingType.LITERAL or (obj.typ == BindingType.URI and IRI(obj.value) in helper_nodes):
        edge_type = EdgeType.ANNOTATION
    elif obj.typ == BindingType.BNODE:
        edge_type = EdgeType.CLASS_EXPRESSION
    else:
        edge_type = EdgeType.OTHER_RELATION
    return Edge(edge_type, predicate, obj)


def order_by_rarity(manager: KgManager, edges: list[Edge], view: EntityView) -> list[Edge]:
    """
    Orders the other relations of each predicate with more than num_values
    objects by how many edges of that predicate point to the object, rarest
    first: objects many entities share (e.g. the most linked wiki page) say
    little about this one.
    """
    by_pred: dict[IRI, list[Edge]] = {}
    for edge in edges:
        if edge.type == EdgeType.OTHER_RELATION and edge.obj.typ == BindingType.URI:
            by_pred.setdefault(edge.predicate, []).append(edge)
    rank: dict[tuple[IRI, str], int] = {}
    for pred, pred_edges in by_pred.items():
        if len(pred_edges) <= view.num_values:
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
        manager: KgManager, entity: IRI, type_predicates: list[IRI], view: EntityView,
        ) -> list[IncomingEdgesSummary]:
    """Incoming edges per predicate (type predicates excluded, see TypeUsage),
    most frequent predicates first, with the num_values rarest subjects."""
    term = to_sparql_term(manager, entity)
    not_types = f"FILTER(?p NOT IN ({', '.join(f'<{p}>' for p in type_predicates)}))" if type_predicates else ""
    rows = select_rows(
        manager,
        # two separate filters: the combination with && triggers an internal QLever error
        f"SELECT ?p (COUNT(?s) AS ?n) WHERE {{ ?s ?p {term} FILTER(isIRI(?s)) {not_types} }} "
        f"GROUP BY ?p ORDER BY DESC(?n) LIMIT {view.max_incoming_predicates}",
    )
    summaries = []
    for row in rows:
        pred = IRI(row["p"].value)
        sample = select_rows(
            manager,
            f"SELECT ?s (COUNT(?y) AS ?d) WHERE {{ ?s <{pred}> {term} FILTER(isIRI(?s)) "
            f"OPTIONAL {{ ?y ?q ?s }} }} GROUP BY ?s ORDER BY ?d ?s LIMIT {view.num_values}",
        )
        summaries.append(IncomingEdgesSummary(pred, int(row["n"].value), [IRI(r["s"].value) for r in sample]))
    return summaries


def retrieve_type_usage(manager: KgManager, entity: IRI, type_predicates: list[IRI]) -> TypeUsage | None:
    if not type_predicates:
        return None
    term = to_sparql_term(manager, entity)
    rows = select_rows(manager, f"SELECT (COUNT(?x) AS ?n) WHERE {{ ?x {sparql_path(type_predicates)} {term} }}")
    count = int(rows[0]["n"].value) if rows else 0
    return TypeUsage(count) if count else None


def retrieve_predicate_usage(manager: KgManager, entity: IRI, type_predicates: list[IRI]) -> PredicateUsage | None:
    term = to_sparql_term(manager, entity)
    rows = select_rows(manager, f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s {term} ?o }}")
    count = int(rows[0]["n"].value) if rows else 0
    if not count:
        return None

    def top_types(position: str) -> list[tuple[IRI, int]]:
        if not type_predicates:
            return []
        return [
            (IRI(r["t"].value), int(r["n"].value)) for r in select_rows(
                manager,
                f"SELECT ?t (COUNT(*) AS ?n) WHERE {{ ?s {term} ?o . ?{position} {sparql_path(type_predicates)} ?t }} "
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


def build_entity_context(manager: KgManager, roles: Roles, entity: IRI, view: EntityView) -> EntityContext:
    """Gathers everything shown about the entity besides names (EntityDisplay);
    all SPARQL queries of a context happen here."""
    type_predicates = membership_predicates(roles)
    edges = retrieve_outgoing_edges(manager, entity)
    role_predicates = plain_predicates(roles, CONTEXT_ROLES)
    # helper node candidates: IRI objects of edges that are no role values or types
    candidates = {
        IRI(obj.value) for pred, obj in edges
        if obj.typ == BindingType.URI and pred.value not in role_predicates | set(type_predicates)
    }
    helper_nodes = find_helper_nodes(manager, candidates, roles)
    categorized = [
        categorize_outgoing_edge(edge, role_predicates, set(type_predicates), helper_nodes) for edge in edges
    ]
    categorized = order_by_rarity(manager, categorized, view)
    has_blank_nodes = any(edge.type == EdgeType.CLASS_EXPRESSION for edge in categorized)
    return EntityContext(
        iri=entity,
        outgoing_edges=categorized,
        incoming_edges=retrieve_incoming_edges(manager, entity, type_predicates, view),
        used_as_type=retrieve_type_usage(manager, entity, type_predicates),
        used_as_predicate=retrieve_predicate_usage(manager, entity, type_predicates),
        blank_nodes=retrieve_blank_nodes(manager, entity, BLANK_NODE_MAX_DEPTH) if has_blank_nodes else {},
        helper_nodes=helper_nodes,
    )


def shown_iris(context: EntityContext) -> set[IRI]:
    """All IRIs of a context that may be shown with their name."""
    iris = {IRI(e.obj.value) for e in context.outgoing_edges if e.obj.typ == BindingType.URI}
    for edges in context.blank_nodes.values():
        for pred, obj in edges:
            iris.add(pred)
            if obj.typ == BindingType.URI:
                iris.add(IRI(obj.value))
    for summary in context.incoming_edges:
        iris.update(summary.sample)
    if context.used_as_predicate:
        usage = context.used_as_predicate
        iris.update(t for t, _ in usage.subject_types + usage.object_types)
        for s, o in usage.examples:
            iris.update(IRI(b.value) for b in (s, o) if b.typ == BindingType.URI)
    return iris


# ---------------------------------------------------------------- the display

@dataclass
class EntityDisplay:
    """
    How the entities of one KG are shown. prefetch retrieves what a view needs
    (names, and for the long view the context); show formats without queries.
    An entity never prefetched is shown by its IRI only. entity_space: objects
    outside it without a name are only counted in a context (e.g. images).
    """
    manager: KgManager
    roles: Roles
    view: EntityView = SHORT_VIEW
    entity_space: tuple[str, ...] = ()
    # entity -> role -> values, sorted; retrieved entities without any value
    # have an empty entry, so they are not retrieved again
    values_by_entity: dict[IRI, dict[str, list[NameValue]]] = field(default_factory=dict)
    # entity -> its context, for the long view
    contexts_by_entity: dict[IRI, EntityContext] = field(default_factory=dict)

    @classmethod
    def of_profile(
            cls,
            manager: KgManager,
            profile: KgProfile,
            view: EntityView = SHORT_VIEW
            ) -> "EntityDisplay":
        return cls(manager, profile.roles, view, tuple(profile.entity_space))

    # ------------------------------------------------------------ retrieval

    def prefetch(self, entities: Iterable[IRI], view: EntityView | None = None) -> None:
        """Retrieves what the view (None: the display's own) needs to show the
        entities: their names and synonyms, and for a context also their
        definitions, their context and the names of all IRIs in it."""
        view = view or self.view
        entities = set(entities)
        if not view.context:
            self.retrieve_values(entities, LINE_ROLES)
            return
        self.retrieve_values(entities, CONTEXT_ROLES, refresh=True)
        for entity in sorted(entities - self.contexts_by_entity.keys()):
            self.contexts_by_entity[entity] = build_entity_context(self.manager, self.roles, entity, view)
        self.retrieve_values(
            {iri for entity in entities for iri in shown_iris(self.contexts_by_entity[entity])}, LINE_ROLES)

    def retrieve_values(self, entities: set[IRI], role_names: tuple[RoleName, ...], refresh: bool = False) -> None:
        """The values of the roles, for the entities not retrieved yet (refresh:
        for all of them, to add roles), in blocks; vocabulary IRIs are skipped."""
        missing = sorted(
            entity for entity in entities
            if (refresh or entity not in self.values_by_entity) and not entity.startswith(VOCABULARY)
        )
        if not missing:
            return
        branches = value_branches(list(role_names), self.roles)
        values_by_entity: dict[IRI, dict[str, set[NameValue]]] = {entity: defaultdict(set) for entity in missing}
        if branches:  # none of the roles in the profile: nothing to retrieve
            for values in values_blocks(missing):
                query = (
                    "SELECT ?entity ?o ?role WHERE {\n"
                    + down_pushed_union(f"VALUES ?entity {{ {values} }}", branches)
                    + "\n}"
                )
                for row in select_rows(self.manager, query):
                    value = row["o"]
                    if value.typ != BindingType.LITERAL:
                        continue  # a helper node (IRI) or blank node: no text to show
                    values_by_entity[IRI(row["entity"].value)][row["role"].value].add(
                        NameValue(language=value.lang or "", text=value.value))
        for entity, by_role in values_by_entity.items():
            self.values_by_entity[entity] = {role: sorted(values) for role, values in by_role.items()}

    # ------------------------------------------------------------ showing

    def values(self, entity: IRI, role_name: RoleName) -> list[NameValue]:
        return self.values_by_entity.get(entity, {}).get(role_name, [])

    def has_name(self, iri: str) -> bool:
        return bool(self.values(IRI(iri), "names"))

    def show(self, entity: IRI, view: EntityView | None = None) -> str:
        """The entity by the view (None: the display's own view): a line, or
        the line with the context (if prefetched with a context view)."""
        view = view or self.view
        if view.context and entity in self.contexts_by_entity:
            return self.render_context(self.contexts_by_entity[entity], view)
        return self.render_line(entity, view)

    def render_line(self, entity: str, view: EntityView = SHORT_VIEW) -> str:
        """e.g. 'large intestine <mouse:MA_1> (names: "Dickdarm"@de; synonyms: "colon", +1 more)';
        a vocabulary IRI as owl:Class."""
        vocabulary = vocabulary_name(entity)
        if vocabulary is not None:
            return vocabulary
        names = self.values(IRI(entity), "names")
        label = names[0].text if names and view.num_names > 0 else None
        reference = self.manager.format_iri(entity) if view.iri or label is None else None
        text = label_with_reference(label, reference) if reference is not None else str(label)
        details = [
            detail for detail in (
                shown_values("names", names[1:], view.num_names - 1),
                shown_values("synonyms", self.values(IRI(entity), "synonyms"), view.num_synonyms),
            ) if detail
        ]
        return text + (f" ({'; '.join(details)})" if details else "")

    def render_value(self, obj: Binding, view: EntityView) -> str:
        """Text of an annotation value: a literal, or the name of a helper node."""
        if obj.typ == BindingType.URI:
            names = self.values(IRI(obj.value), "names")
            return names[0].text if names else self.render_line(obj.value)
        text = clip(obj.value, view.max_chars_per_literal)
        return f"{text}@{obj.lang}" if obj.lang else text

    def render_context(self, context: EntityContext, view: EntityView) -> str:
        """The line of the entity, then its context in a fixed order of
        importance, cut at view.max_chars. Pure: no SPARQL."""
        lines: list[str] = []
        by_type: dict[EdgeType, list[Edge]] = {}
        for edge in context.outgoing_edges:
            by_type.setdefault(edge.type, []).append(edge)

        def grouped(edge_type: EdgeType) -> dict[IRI, list[Edge]]:
            groups: dict[IRI, list[Edge]] = {}
            for edge in by_type.get(edge_type, []):
                groups.setdefault(edge.predicate, []).append(edge)
            return groups

        # names (besides the label in the header), synonyms (not equal to a
        # name), definitions: from the roles of the profile
        names = self.values(context.iri, "names")
        name_texts = {name.text for name in names}
        synonyms = [synonym for synonym in self.values(context.iri, "synonyms") if synonym.text not in name_texts]
        definitions = self.values(context.iri, "definitions")
        for line in (
            shown_values("further names", names[1:], len(names)),
            shown_values("synonyms", synonyms, view.num_synonyms),
            shown_values("definitions", definitions, view.num_values, view.max_chars_per_literal),
        ):
            if line:
                lines.append(line)
        role_texts = name_texts | {value.text for value in synonyms + definitions}

        # annotations not shown as role values: identical texts of several
        # predicates once, longest first
        predicates_of: dict[str, list[str]] = {}
        for edge in by_type.get(EdgeType.ANNOTATION, []):
            text = self.render_value(edge.obj, view)
            if text in role_texts or edge.obj.value in role_texts:
                continue
            preds = predicates_of.setdefault(text, [])
            if local_name(edge.predicate) not in preds:
                preds.append(local_name(edge.predicate))
        by_preds: dict[str, list[str]] = {}
        for text, preds in predicates_of.items():
            by_preds.setdefault(", ".join(preds), []).append(text)
        for preds, texts in sorted(by_preds.items(), key=lambda kv: -max(map(len, kv[1]))):
            texts.sort(key=len, reverse=True)
            more = f" (+{len(texts) - view.num_values} more)" if len(texts) > view.num_values else ""
            lines.append(f"{preds}: " + " | ".join(f'"{t}"' for t in texts[:view.num_values]) + more)

        # types
        types = list(dict.fromkeys(self.render_line(e.obj.value) for e in by_type.get(EdgeType.TYPE, [])))
        if types:
            lines.append("types: " + ", ".join(types))

        # class expressions
        for pred, edges in grouped(EdgeType.CLASS_EXPRESSION).items():
            expressions = [render_blank_node(e.obj.value, context.blank_nodes, self.render_line) for e in edges]
            lines.append(f"{local_name(pred)}: " + "; ".join(dict.fromkeys(expressions)))

        # other relations: predicates with few objects first; objects with a name
        # first (rare objects without one are often files or stubs), then rarest first
        only_hidden = []
        for pred, edges in sorted(grouped(EdgeType.OTHER_RELATION).items(), key=lambda kv: len(kv[1])):
            objects = list(dict.fromkeys(e.obj.value for e in edges))
            shown = [
                o for o in objects
                if not self.entity_space or o.startswith(self.entity_space)
                or o.startswith(VOCABULARY) or self.has_name(o)
            ]
            hidden = len(objects) - len(shown)
            if not shown:
                only_hidden.append(f"{local_name(pred)} ({hidden})")
                continue
            shown.sort(key=lambda o: not self.has_name(o) and not o.startswith(VOCABULARY))
            notes = []
            if len(shown) > view.num_values:
                notes.append(f"+{len(shown) - view.num_values} more, named and rarest shown")
            if hidden:
                notes.append(f"{hidden} unnamed values outside the entity space")
            values = ", ".join(self.render_line(o) for o in shown[:view.num_values])
            lines.append(
                f"{local_name(pred)} ({len(objects)}): {values}" + (f" [{'; '.join(notes)}]" if notes else "")
            )
        if only_hidden:
            lines.append("only unnamed values outside the entity space: " + ", ".join(only_hidden))

        # incoming edges
        for summary in context.incoming_edges:
            subjects = ", ".join(self.render_line(s) for s in summary.sample)
            lines.append(f"incoming {local_name(summary.predicate)} ({summary.count}): {subjects}")

        # usage by other entities
        if context.used_as_type:
            lines.append(f"used as type of {context.used_as_type.instance_count} entities")
        if context.used_as_predicate:
            usage = context.used_as_predicate

            def type_counts(types: list[tuple[IRI, int]]) -> str:
                return ", ".join(f"{self.render_line(t)} {n}" for t, n in types) or "untyped"

            def example(binding: Binding) -> str:
                if binding.typ == BindingType.URI:
                    return self.render_line(binding.value)
                if binding.typ == BindingType.BNODE:
                    return "_"
                return f'"{clip(binding.value, view.max_chars_per_literal)[:60]}"'

            examples = "; ".join(f"{example(s)} -> {example(o)}" for s, o in usage.examples)
            lines.append(
                f"used as predicate in {usage.triple_count} triples; subject types: "
                f"{type_counts(usage.subject_types)}; object types: {type_counts(usage.object_types)}; "
                f"{usage.num_literal_objects} literal objects; e.g. {examples}"
            )

        # budget: sections are ordered by importance, so the least important are cut
        header = "Entity " + self.render_line(context.iri, EntityView(num_names=1))
        out = [header]
        used = len(header)
        for index, line in enumerate(lines):
            if used + len(line) + 3 > view.max_chars:
                out.append(f"  … ({len(lines) - index} more lines omitted)")
                break
            out.append(f"  {line}")
            used += len(line) + 3
        return "\n".join(out)
