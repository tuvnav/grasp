"""
A function of the judges (om-search-verify, the agentic om-verify):
show_entities shows entities of one KG the way the presented ones are shown
(LONG_VIEW: names, synonyms, definitions and the other roles, the context with
the names of the neighbours), so that the judge needs no steps to query their
triples. Every IRI on its own: a malformed or unknown one gives its error, one
without anything to show says so, the others their view. What is shown becomes
known (know_before_use), like the presented entities.
"""
from grasp.tasks.entities import prepare_entity
from grasp.tasks.om.foundation.rendering import format_display_content
from grasp.tasks.om.foundation.types import IRI
from grasp.tasks.om.kg_operations.entity_rendering import LONG_VIEW, EntityDisplay
from grasp.tasks.om.sparql_access import entity_exists_in_kg
from grasp.utils import FunctionCallException

SHOW_ENTITIES = "show_entities"
MAX_SHOWN_ENTITIES = 5  # per call: the long view is long


def show_entities_definition(kg_names: list[str]) -> dict:
    return {
        "name": SHOW_ENTITIES,
        "description": (
            "Shows entities of one knowledge graph the way the presented entities are shown: "
            "their names, synonyms, definitions and other values, and their related entities with names."
            ),
        "parameters": {
            "type": "object",
            "properties": {
                "kg": {"type": "string", "enum": kg_names, "description": "The knowledge graph of the entities."},
                "iris": {"type": "array", "items": {"type": "string"},
                         "description": f"The IRIs of 1 to {MAX_SHOWN_ENTITIES} entities."},
            },
            "required": ["kg", "iris"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def show_entities(displays_by_kg: dict[str, EntityDisplay], fn_args: dict, known: set[str]) -> str:
    kg_name = fn_args.get("kg")
    display = displays_by_kg.get(kg_name) if isinstance(kg_name, str) else None
    if display is None:
        raise FunctionCallException(f"Unknown knowledge graph {kg_name!r}, use one of {sorted(displays_by_kg)}")
    iris = fn_args.get("iris")
    if not isinstance(iris, list) or not 1 <= len(iris) <= MAX_SHOWN_ENTITIES:
        raise FunctionCallException(f"Give the IRIs of 1 to {MAX_SHOWN_ENTITIES} entities")

    # every IRI on its own: its entity, or the error of this IRI
    entities_or_errors: list[tuple[IRI | None, str]] = []
    for iri in iris:
        try:
            entity = IRI(prepare_entity(display.manager, str(iri)).identifier)
        except ValueError as error:
            entities_or_errors.append((None, f"Error for {iri}: {error}"))
            continue
        if entity not in known and not entity_exists_in_kg(display.manager, entity):
            entities_or_errors.append((None, f"Error for {iri}: no entity {entity} in the knowledge graph {kg_name}"))
            continue
        entities_or_errors.append((entity, ""))

    display.prefetch(entity for entity, _ in entities_or_errors if entity is not None)
    shown = []
    for entity, error in entities_or_errors:
        if entity is None:
            shown.append(error)
        elif not display.has_content(entity):
            shown.append(f"No triples for {entity}.")
        else:
            known.add(entity)
            if entity in display.contexts_by_entity:
                known.update(display.shown_iris(entity))
            shown.append(format_display_content(display.show_section(entity, LONG_VIEW), level=3))
    return "\n\n".join(shown)
