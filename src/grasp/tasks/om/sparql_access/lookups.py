"""
Small lookups of single entities: existence, outgoing edges, blank node trees.
"""
from grasp.manager import KgManager
from grasp.sparql.types import AskResult, Binding, SelectResult
from grasp.sparql.utils import prepare_identifier_for_sparql
from grasp.tasks.om.foundation.types import IRI, BindingType, BlankNodeTree, SparqlTerm
from grasp.tasks.om.sparql_access.execution import select_rows
from grasp.utils import FunctionCallException


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


def to_sparql_term(manager: KgManager, iri: IRI) -> SparqlTerm:
    """IRI prepared for insertion into SPARQL text (e.g. <...>, escaped)."""
    return SparqlTerm(prepare_identifier_for_sparql(iri, manager.iri_literal_parser))


def retrieve_outgoing_edges(manager: KgManager, entity: IRI) -> list[tuple[Binding, Binding]]:
    """(predicate, object) of all triples with the entity as subject."""
    result = manager.execute_sparql(
        f"SELECT ?p ?o WHERE {{ {to_sparql_term(manager, entity)} ?p ?o }}",
        force_select_result=True,
    )
    if not isinstance(result, SelectResult):
        return []
    # bindings() drops unbound variables; for ?p ?o of a triple pattern both are always bound
    return [row for row in result.bindings() if len(row) == 2]  # type: ignore[misc]


def retrieve_blank_nodes(manager: KgManager, entity: IRI, max_depth: int) -> BlankNodeTree:
    """
    The blank nodes below the entity with their edges, level by level: the query
    for depth d is a chain from the entity whose every link is a blank node.
    A property path would also walk through named entities into their blank nodes.
    """
    tree: BlankNodeTree = {}
    start = to_sparql_term(manager, entity)
    for depth in range(1, max_depth + 1):
        chain = f"{start} ?p0 ?b0 . FILTER(isBlank(?b0)) " + " ".join(
            f"?b{i - 1} ?p{i} ?b{i} . FILTER(isBlank(?b{i}))" for i in range(1, depth)
        )
        last = f"b{depth - 1}"
        rows = select_rows(manager, f"SELECT DISTINCT ?{last} ?p ?o WHERE {{ {chain} ?{last} ?p ?o }}")
        new_edges = False
        for row in rows:
            edge = (IRI(row["p"].value), row["o"])
            edges = tree.setdefault(row[last].value, [])
            if edge not in edges:
                edges.append(edge)
                new_edges = True
        if not new_edges or not any(row["o"].typ == BindingType.BNODE for row in rows):
            break
    return tree
