"""Shared building blocks for showing KG content to the LLM, so that entities
look the same in all stages. Task-specific views (e.g. of an anchor result or an
entity context) stay with their task and use these blocks."""
from grasp.sparql.types import Binding
from grasp.tasks.entities import Entity
from grasp.tasks.om.model import (
    IRI, OWL_ON_PROPERTY, RDF_FIRST, RDF_REST, RDF_TYPE, VOCABULARY_PREFIXES,
    BindingType, BlankNodeTree, local_name,
)

# restriction predicates and their keyword in Manchester syntax
RESTRICTIONS = {
    IRI("http://www.w3.org/2002/07/owl#someValuesFrom"): "some",
    IRI("http://www.w3.org/2002/07/owl#allValuesFrom"): "only",
    IRI("http://www.w3.org/2002/07/owl#hasValue"): "value",
}


def display_iri(iri: str, labels: dict[IRI, str]) -> str:
    """owl:Class for vocabulary, otherwise 'Label <local name>' or '<local name>'."""
    for namespace, prefix in VOCABULARY_PREFIXES.items():
        if iri.startswith(namespace):
            return f"{prefix}:{iri[len(namespace):]}"
    label = labels.get(IRI(iri))
    local = local_name(iri)
    return f"{label} <{local}>" if label and label != local else f"<{local}>"


def clip(text: str, max_chars: int) -> str:
    """Whitespace collapsed, cut at max_chars."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …"


def render_blank_node(
        node: str,
        tree: BlankNodeTree,
        labels: dict[IRI, str],
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
            return render_blank_node(obj.value, tree, labels, seen)
        if obj.typ == BindingType.URI:
            return display_iri(obj.value, labels)
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
        f"{display_iri(pred, labels)} {', '.join(render(o) for o in objs)}"
        for pred, objs in by_pred.items()
        if pred != RDF_TYPE  # e.g. owl:Restriction, implied by the rendering
    ]
    return "(" + "; ".join(parts) + ")"


def format_entity(entity: Entity) -> str:
    output = f"Full IRI: {entity.identifier}"
    if entity.entity != entity.identifier:
        output += f", shortened IRI: {entity.entity}"
    if entity.label is not None:
        output += f", label: {entity.label}"
    if entity.aliases and entity.aliases != [entity.label]:
        output += f", aliases: {entity.aliases}"
    if entity.infos is not None and entity.infos != []:
        output += f", infos: {entity.infos}"
    return output
