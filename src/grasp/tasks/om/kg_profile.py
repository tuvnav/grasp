"""
Default profile of a KG (KgProfile): the default canon of predicates and
structure relations (model.py), restricted to what each partition actually
uses, plus statistics as evidence for the profile task (PartitionStats).

Scales with the KG: counting is done by the SPARQL server (GROUP BY), so only
aggregates are transferred, and the local name sample is drawn while streaming
the entities (reservoir sampling), so memory stays at the sample size.
"""
from collections import Counter, defaultdict
import random
import re
from typing import Iterable, Iterator
from urllib.parse import unquote

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.tasks.om.model import (
    CODE_PREDICATES, CROSS_REFERENCE_PREDICATES, DEFAULT_STRUCTURE_RELATIONS, DEFINITION_PREDICATES,
    IRI, KIND_PARTITION_SPARQL, LABEL_PREDICATES, SYNONYM_PREDICATES,
    KgPartitionProfile, KgProfile, NameType, StructureRelation,
    configured_namespaces, local_name, split_prologue, sparql_path,
)
from grasp.tasks.om.retrieve import determine_entity_space, select_rows, stream_rows, values_blocks
from pydantic import BaseModel

# local names per partition compared with labels; enough to tell a share of
# 0.2 from one of 0.8 (95 % interval at most +-7 points)
SAMPLE_SIZE = 200
SAMPLE_SEED = 0
# minimum labelled entities in the sample to judge local names by their labels
MIN_LABELLED = 20
# local name agrees with a label: share of its tokens found in the label
MIN_TOKEN_AGREEMENT = 0.5
# share of local names that agree with labels / look speaking, above which
# local names count as names and below which (speaking only) as codes
NAME_SHARE = 0.8
AGREEMENT_SHARE = 0.5
CODE_SHARE = 0.2

ROLE_PREDICATES = {
    "name": LABEL_PREDICATES,
    "synonym": SYNONYM_PREDICATES,
    "definition": DEFINITION_PREDICATES,
    "code": CODE_PREDICATES,
    "cross_reference": CROSS_REFERENCE_PREDICATES,
}

# SPARQL equivalent of namespace_of: scheme and host, or up to the last # or /
NAMESPACE_EXPRESSION = (
    'IF(CONTAINS(STR(?entity), "://"), '
    'REPLACE(STR(?entity), "^([^:/]+://[^/#?]+).*$", "$1"), '
    'REPLACE(STR(?entity), "[^#/]*$", ""))'
)
# characters not allowed in a SPARQL IRI reference
INVALID_IRI_CHARACTERS = re.compile(r'[\s<>"{}|^`\\]')


class PartitionStats(BaseModel):
    """Evidence for a partition profile, shown to the profile task."""
    size: int
    # entities per predicate (default canon only) and per structure relation name
    predicate_counts: dict[IRI, int] = {}
    structure_counts: dict[str, int] = {}
    # local name sample: labelled entities, share agreeing with a label, share speaking
    sample_size: int = 0
    labelled: int = 0
    local_name_agreement: float | None = None
    local_name_speaking: float | None = None


def partition_query(manager: KgManager, select: str, where: str, tail: str = "") -> str:
    """Query over the entities of the kind partitions (?entity ?partition, see
    KIND_PARTITION_SPARQL), joined with the given patterns."""
    prefixes, body = split_prologue(manager, KIND_PARTITION_SPARQL)
    prologue = "\n".join(f"PREFIX {short}: <{long}>" for short, long in prefixes.items())
    return (f"{prologue}\nSELECT {select} WHERE {{\n"
            f"{{ SELECT DISTINCT ?entity ?partition WHERE {{\n{body}\n}} }}\n"
            f"{where}\n}} {tail}")


def space_filter(space: tuple[str, ...]) -> str:
    if not space:
        return ""
    return "FILTER(" + " || ".join(f'STRSTARTS(STR(?entity), "{ns}")' for ns in space) + ")"


def count_rows(rows: list[dict], *keys: str) -> dict[tuple[str, ...], int]:
    return {tuple(row[k].value for k in keys): int(row["n"].value) for row in rows if all(k in row for k in keys)}


def namespace_counts_by_server(manager: KgManager, named_only: bool) -> Counter[str]:
    """Entities per namespace over all partitions, optionally only entities with
    a default name or synonym (keeps out e.g. image files of wiki KGs)."""
    named = (f"FILTER EXISTS {{ ?entity {sparql_path(LABEL_PREDICATES + SYNONYM_PREDICATES)} ?name }}"
             if named_only else "")
    sparql = partition_query(
        manager, "?ns (COUNT(DISTINCT ?entity) AS ?n)",
        f"{named}\nBIND({NAMESPACE_EXPRESSION} AS ?ns)", "GROUP BY ?ns",
    )
    return Counter({ns: n for (ns,), n in count_rows(select_rows(manager, sparql), "ns").items()})


def entity_space(manager: KgManager, config: GraspConfig) -> tuple[str, ...]:
    """Configured namespaces of the KG, otherwise the namespace with the most
    named entities (all entities if none has a default name)."""
    configured = configured_namespaces(config, manager.kg)
    if configured:
        return tuple(configured)
    counts = namespace_counts_by_server(manager, named_only=True) or namespace_counts_by_server(manager, named_only=False)
    return determine_entity_space(counts, None)


def partition_sizes(manager: KgManager, space: tuple[str, ...]) -> dict[str, int]:
    sparql = partition_query(manager, "?partition (COUNT(DISTINCT ?entity) AS ?n)", space_filter(space), "GROUP BY ?partition")
    return {partition: n for (partition,), n in count_rows(select_rows(manager, sparql), "partition").items()}


def predicate_counts(manager: KgManager, space: tuple[str, ...], predicates: list[IRI]) -> dict[tuple[str, str], int]:
    """(partition, predicate) -> entities with at least one value."""
    values = " ".join(f"<{p}>" for p in predicates)
    sparql = partition_query(
        manager, "?partition ?p (COUNT(DISTINCT ?entity) AS ?n)",
        f"{space_filter(space)}\nVALUES ?p {{ {values} }}\n?entity ?p ?value .",
        "GROUP BY ?partition ?p",
    )
    return count_rows(select_rows(manager, sparql), "partition", "p")


def structure_counts(manager: KgManager, space: tuple[str, ...], relation: StructureRelation) -> dict[str, int]:
    """partition -> entities that are ?s of the relation."""
    sparql = partition_query(
        manager, "?partition (COUNT(DISTINCT ?entity) AS ?n)",
        f"{space_filter(space)}\nBIND(?entity AS ?s)\n{relation.predicate_pattern.predicates_to_sparql()}",
        "GROUP BY ?partition",
    )
    return {partition: n for (partition,), n in count_rows(select_rows(manager, sparql), "partition").items()}


def sample_entities(manager: KgManager, space: tuple[str, ...], k: int = SAMPLE_SIZE,
                    seed: int = SAMPLE_SEED) -> dict[str, list[IRI]]:
    """Up to k random entities per partition, drawn while streaming (reservoir
    sampling), so memory stays at k per partition whatever the KG size."""
    rng = random.Random(seed)
    samples: dict[str, list[IRI]] = defaultdict(list)
    seen: Counter[str] = Counter()
    sparql = partition_query(manager, "?entity ?partition", space_filter(space), "ORDER BY ?entity")
    for row in stream_rows(manager, sparql):
        entity, partition = IRI(row["entity"]["value"]), row["partition"]["value"]
        if INVALID_IRI_CHARACTERS.search(entity):
            continue
        seen[partition] += 1
        if len(samples[partition]) < k:
            samples[partition].append(entity)
        elif (j := rng.randrange(seen[partition])) < k:
            samples[partition][j] = entity
    return dict(samples)


def retrieve_names(manager: KgManager, entities: Iterable[IRI], predicates: list[IRI]) -> dict[IRI, list[str]]:
    names: dict[IRI, list[str]] = defaultdict(list)
    if not predicates:
        return names
    for values in values_blocks(entities):
        sparql = (f"SELECT ?entity ?name WHERE {{ VALUES ?entity {{ {values} }} "
                  f"?entity {sparql_path(predicates)} ?name FILTER(isLiteral(?name)) }}")
        for row in select_rows(manager, sparql):
            names[IRI(row["entity"].value)].append(row["name"].value)
    return names


NAME_TOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def name_tokens(text: str) -> list[str]:
    """Lowercased word and number tokens; splits camelCase, _ - ( ) and percent escapes."""
    return [t.lower() for t in NAME_TOKEN.findall(unquote(text))]


def is_speaking_name(name: str) -> bool:
    """Whether a local name looks like words rather than a code (MA_0000001,
    NCI_C12345, relation_22321): more than half of its tokens are words of at
    least three letters with a vowel. Single letters are not counted (Class_N),
    numbers are. Form only: a speaking name may still be an identifier."""
    tokens = [t for t in name_tokens(name) if len(t) > 1 or t.isdigit()]
    if not tokens:
        return False
    words = sum(t.isalpha() and len(t) >= 3 and re.search(r"[aeiouy]", t) is not None for t in tokens)
    return words / len(tokens) > 0.5


def agrees_with_label(name: str, labels: list[str]) -> bool:
    """Whether most tokens of the local name occur in one of the labels."""
    tokens = set(name_tokens(name))
    if not tokens:
        return False
    return any(len(tokens & set(name_tokens(label))) / len(tokens) >= MIN_TOKEN_AGREEMENT for label in labels)


def local_name_type(stats: PartitionStats) -> NameType:
    """Local names agreeing with labels, or speaking without labels to compare
    with, count as names. A speaking local name that differs from the label
    (e.g. another language) is reported in the statistics."""
    # small partitions (e.g. a handful of properties) count if all are labelled
    enough_labelled = stats.labelled >= min(MIN_LABELLED, stats.sample_size)
    if enough_labelled and stats.local_name_agreement is not None \
            and stats.local_name_agreement >= AGREEMENT_SHARE:
        return NameType.LABEL
    if stats.local_name_speaking is None:
        return NameType.OTHER
    if stats.local_name_speaking >= NAME_SHARE:
        return NameType.LABEL
    if stats.local_name_speaking <= CODE_SHARE:
        return NameType.CODE
    return NameType.OTHER


def local_name_stats(manager: KgManager, stats: PartitionStats, sample: list[IRI], name_predicates: list[IRI]) -> None:
    names = retrieve_names(manager, sample, name_predicates)
    local_names = {entity: local_name(entity) for entity in sample}
    stats.sample_size = len(sample)
    stats.labelled = sum(entity in names for entity in sample)
    if sample:
        stats.local_name_speaking = sum(map(is_speaking_name, local_names.values())) / len(sample)
    if stats.labelled:
        stats.local_name_agreement = sum(
            agrees_with_label(local_names[entity], names[entity]) for entity in sample if entity in names
        ) / stats.labelled


def default_profile(manager: KgManager, config: GraspConfig) -> tuple[KgProfile, dict[str, PartitionStats]]:
    """The default canon restricted to what each partition uses, and the
    statistics behind it."""
    space = entity_space(manager, config)
    sizes = partition_sizes(manager, space)
    used = predicate_counts(manager, space, [p for ps in ROLE_PREDICATES.values() for p in ps])
    structure = {relation.name: structure_counts(manager, space, relation) for relation in DEFAULT_STRUCTURE_RELATIONS}
    samples = sample_entities(manager, space)

    profiles, all_stats = {}, {}
    for partition, size in sorted(sizes.items()):
        def in_use(role: str) -> list[IRI]:
            return [p for p in ROLE_PREDICATES[role] if used.get((partition, p))]

        stats = PartitionStats(
            size=size,
            predicate_counts={IRI(p): n for (part, p), n in used.items() if part == partition},
            structure_counts={name: counts[partition] for name, counts in structure.items() if counts.get(partition)},
        )
        local_name_stats(manager, stats, samples.get(partition, []), in_use("name"))
        profiles[partition] = KgPartitionProfile(
            local_name_type=local_name_type(stats),
            name_predicates=in_use("name"),
            synonym_predicates=in_use("synonym"),
            code_predicates=in_use("code"),
            definition_predicates=in_use("definition"),
            cross_references=in_use("cross_reference"),
            structure_predicates=[r for r in DEFAULT_STRUCTURE_RELATIONS if r.name in stats.structure_counts],
        )
        all_stats[partition] = stats
    return KgProfile(kg_name=manager.kg, alignment_namespaces=list(space), partitions=profiles), all_stats
