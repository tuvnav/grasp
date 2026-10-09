import json
import logging
from pathlib import Path

import re
import unicodedata

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.utils import derive_label_from_iri, camel_case_split
from grasp.tasks.entities import Entity
from grasp.tasks.om import AlignmentTaskInput, PotentialCorrespondences, entity_exists_in_kg
from grasp.sparql.types import ObjType, SelectResult


def normalize_name(name: str) -> str:
    _WHITESPACE_RE = re.compile(r"\s+")

    name = unicodedata.normalize("NFKC", name)
    name = name.replace("_", " ").replace("-", " ")
    name = camel_case_split(name)
    name = _WHITESPACE_RE.sub(" ", name).strip()
    return name.casefold()


def entity_names(entity: Entity) -> list[str]:
    names = [entity.label] if entity.label else []
    if entity.aliases:
        names.extend(entity.aliases)
    return names


def _entities_from_ids(
    kg_manager: KgManager, ids: list[str], info: dict[str, dict]
) -> dict[str, Entity]:
    result: dict[str, Entity] = {}
    for iri in ids:
        entry = info.get(iri, {})
        label = entry.get("label") or derive_label_from_iri(iri, kg_manager.prefixes)

        result[iri] = Entity(
            identifier=iri,
            entity=kg_manager.format_iri(iri),
            label=label,
            aliases=entry.get("alias", []),
            infos=entry.get("other", []),
        )

    return result


def _filter_own_namespace(identifiers: list[str], kg_manager: KgManager) -> list[str]:
    """
    Keeps only identifiers within the KG's own namespace(s).
    """
    own_namespaces = tuple(kg_manager.kg_prefixes.values())
    if not own_namespaces:
        # no declared KG-specific namespace to filter against - fail open
        # rather than risk filtering out everything
        return identifiers
    return [i for i in identifiers if i.startswith(own_namespaces)]


def retrieve_entities_and_properties(kg_manager: KgManager) -> dict[str, Entity]:
    """
    Retrieves all classes and properties of a KG using GRASP's own, already
    built entities/properties indices.
    Does not cover literals.
    """
    entities_data = kg_manager.get_data(ObjType.ENTITY.index_name)
    properties_data = kg_manager.get_data(ObjType.PROPERTY.index_name)

    property_ids = _filter_own_namespace(
        [identifier for identifier, _ in properties_data], kg_manager
    )
    property_id_set = set(property_ids)

    # the entities index may already include properties (depends on how the KG
    # was set up) - exclude those here so they are only queried/enriched once,
    # via the properties index's own, property-specific info SPARQL
    entity_ids = _filter_own_namespace(
        [identifier for identifier, _ in entities_data if identifier not in property_id_set],
        kg_manager,
    )

    entity_info = kg_manager.get_info_for_identifiers_from_index(
        entity_ids, ObjType.ENTITY.index_name
    )
    property_info = kg_manager.get_info_for_identifiers_from_index(
        property_ids, ObjType.PROPERTY.index_name
    )

    result = _entities_from_ids(kg_manager, entity_ids, entity_info)
    result.update(_entities_from_ids(kg_manager, property_ids, property_info))
    return result


def load_iris_file(path: str | Path) -> list[str]:
    """
    Reads IRIs from a plain text file, one IRI per line. Empty lines and lines
    starting with '#' are skipped, optional angle brackets are stripped.
    """
    iris = []
    with Path(path).expanduser().open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            iris.append(line.removeprefix("<").removesuffix(">"))
    return iris


def select_iris_with_sparql(kg_manager: KgManager, sparql: str) -> list[str]:
    """
    Executes a single-column SELECT query against the KG and returns the IRIs
    it yields, in result order. Non-IRI results (literals, blank nodes) are
    skipped with a warning.
    """
    result = kg_manager.execute_sparql(sparql)
    if not isinstance(result, SelectResult):
        raise ValueError("source_selection_sparql must be a SELECT query")
    if result.num_columns != 1:
        raise ValueError("source_selection_sparql must return a single column")

    iris = []
    skipped = 0
    for bindings in result.bindings():
        binding = next(iter(bindings), None)
        if binding is None:
            continue
        if binding.typ != "uri":
            skipped += 1
            continue
        iris.append(binding.value)

    if skipped:
        logging.warning(
            f"Skipped {skipped} non-IRI results of source_selection_sparql."
        )
    return iris


def retrieve_selected_entities(kg_manager: KgManager, iris: list[str]) -> dict[str, Entity]:
    """
    Retrieves only the given IRIs (in the given order) instead of all entities
    and properties of the KG. They do not need to be part of GRASP's indices,
    info is retrieved live via the indices' info SPARQL. IRIs that are in the
    properties index are enriched with the property-specific info SPARQL.
    """
    properties_data = kg_manager.try_get_data(ObjType.PROPERTY.index_name)

    def is_property(iri: str) -> bool:
        return (
            properties_data is not None
            and properties_data.id_from_identifier(iri) is not None
        )

    entity_ids = [iri for iri in iris if not is_property(iri)]
    property_ids = [iri for iri in iris if is_property(iri)]

    entity_info = kg_manager.get_info_for_identifiers_from_index(
        entity_ids, ObjType.ENTITY.index_name
    )
    property_info = kg_manager.get_info_for_identifiers_from_index(
        property_ids, ObjType.PROPERTY.index_name
    ) if property_ids else {}

    entities = _entities_from_ids(kg_manager, entity_ids, entity_info)
    entities.update(_entities_from_ids(kg_manager, property_ids, property_info))
    # restore the given order
    return {iri: entities[iri] for iri in iris}


def get_source_iris(kg_manager: KgManager, configs: dict) -> list[str] | None:
    """
    Collects the source IRIs selected via source_iris (list), source_iris_file
    (plain text file, one IRI per line) and source_selection_sparql (single-column
    SELECT query). Several options can be combined, their IRIs are concatenated
    in this order without duplicates. Returns None if no option is set, i.e. all
    entities and properties of the source KG should be used.
    """
    source_iris = configs.get("source_iris")
    source_iris_file = configs.get("source_iris_file")
    source_selection_sparql = configs.get("source_selection_sparql")

    if source_iris is None and source_iris_file is None and source_selection_sparql is None:
        return None

    # explicitly given IRIs might be mistyped, so check that they exist;
    # IRIs from the selection query exist in the KG by construction
    explicit = list(source_iris or [])
    if source_iris_file is not None:
        explicit.extend(load_iris_file(source_iris_file))

    iris = []
    missing = []
    for iri in explicit:
        if entity_exists_in_kg(kg_manager, iri):
            iris.append(iri)
        else:
            missing.append(iri)
    if missing:
        logging.warning(
            f"Skipping {len(missing)} selected IRIs not found in "
            f"{kg_manager.kg}:\n" + "\n".join(missing)
        )

    if source_selection_sparql is not None:
        iris.extend(select_iris_with_sparql(kg_manager, source_selection_sparql))

    # deduplicate, keep first occurrence
    iris = list(dict.fromkeys(iris))
    logging.info(f"Selected {len(iris)} source IRIs from {kg_manager.kg}.")
    return iris


def build_string_matching_dict(entities: dict[str, Entity]) -> dict[str, list[Entity]]:
    result: dict[str, list[Entity]] = {}
    for entity in entities.values():
        names = entity_names(entity)
        for name in names:
            name = normalize_name(name)
            if result.get(name) is None:
                result[name] = []
            result[name].append(entity)

    return result


def perform_string_matching(
    entities_source: dict[str, Entity],
    entities_target: dict[str, Entity]
        ) -> tuple[list[PotentialCorrespondences], list[Entity]]:
    target_matching_dict = build_string_matching_dict(entities_target)
    matches: list[PotentialCorrespondences] = []
    unmatched: list[Entity] = []

    for src_entity in entities_source.values():
        src_names = entity_names(src_entity)
        matched_tgt_entities: list[Entity] = []
        for src_name in src_names:
            src_name = normalize_name(src_name)
            if src_name in target_matching_dict:
                for tgt_entity in target_matching_dict[src_name]:
                    if tgt_entity in matched_tgt_entities:
                        continue
                    matched_tgt_entities.append(tgt_entity)

        if len(matched_tgt_entities) == 0:
            unmatched.append(src_entity)

        else:
            matches.append(PotentialCorrespondences(source_entity=src_entity, candidates=matched_tgt_entities))

    return matches, unmatched


def write_jsonl_input(
    string_matches: list[PotentialCorrespondences],
    unmatched_entities: list[Entity],
    source_kg: str,
    target_kg: str,
    output_file: Path,
    batch_size: int = 1,
    limit: int | None = None,
) -> Path:

    entities = string_matches + unmatched_entities
    if limit is not None:
        logging.info(f"Limiting source entities to the first {limit} entities.")
        entities = entities[:limit]

    with output_file.open("w", encoding="utf-8") as jsonl_file:
        for i in range(0, len(entities), batch_size):

            if len(string_matches) >= i + batch_size:
                potential_corrs = entities[i:i + batch_size]
                unmatched = []
            elif len(string_matches) > i:
                potential_corrs = entities[i:len(string_matches)]
                unmatched = entities[len(string_matches):i + batch_size]
            else:
                potential_corrs = []
                unmatched = entities[i:i + batch_size]

            record = AlignmentTaskInput(
                unmatched_entities=unmatched,
                potential_correspondences=potential_corrs,
                source_kg=source_kg,
                target_kg=target_kg
                ).model_dump()

            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(
        f"Wrote OM task with {len(entities)} entities "
        f"in batches of {batch_size} to {output_file}"
        )

    return output_file


def om_pretask(source_manager: KgManager, target_manager: KgManager, output_file: Path, config: GraspConfig):
    configs = config.task_kwargs.get("om_pretask", {})
    skip_prematching = configs.get("skip_prematching")
    batch_size = configs.get("batch_size")
    limit = configs.get("limit")

    source_iris = get_source_iris(source_manager, configs)
    if source_iris is None:
        source_entities = retrieve_entities_and_properties(source_manager)
    else:
        source_entities = retrieve_selected_entities(source_manager, source_iris)

    if skip_prematching:
        matches = []
        unmatched = list(source_entities.values())
    else:
        target_entities = retrieve_entities_and_properties(target_manager)
        matches, unmatched = perform_string_matching(source_entities, target_entities)

    write_jsonl_input(
        matches,
        unmatched,
        source_manager.kg,
        target_manager.kg,
        output_file,
        (batch_size if batch_size else 1),
        limit
    )
