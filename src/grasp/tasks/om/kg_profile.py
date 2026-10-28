"""
Queries and statistics behind the staged KG profile (profile_model.py): entity
space, partitions, predicate coverage, the default canon restricted to what the
KG uses, and local names.

Scales with the KG: counting is done by the SPARQL server (GROUP BY), so only
aggregates are transferred, and the local name sample is drawn while streaming
the entities (reservoir sampling), so memory stays at the sample size.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import random
import re
from typing import Iterable
from urllib.parse import unquote

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.tasks.om.model import (
    CODE_PREDICATES, CROSS_REFERENCE_PREDICATES, DEFAULT_STRUCTURE_RELATIONS, DEFINITION_PREDICATES,
    IRI, KIND_PARTITION_SPARQL, LABEL_PREDICATES, RDF_TYPE, SYNONYM_PREDICATES, UNCATEGORIZED,
    NameType, PredicatePattern, configured_namespaces, local_name, split_prologue, sparql_path,
)
from grasp.tasks.om.profile_model import (
    ElementKey, PartitionDefinition, Roles, Usage, role_elements, roles_from_elements,
)
from grasp.tasks.om.kg_sample import current_sample, values_block
from grasp.tasks.om.retrieve import select_rows, stream_rows, values_blocks
from pydantic import BaseModel

# partitions of the default partitioning by kind (KIND_PARTITION_SPARQL), in order
KINDS = ("class", "property", "concept", "instance")

# canon predicates per role field of Roles
CANON = {
    "names": LABEL_PREDICATES,
    "synonyms": SYNONYM_PREDICATES,
    "definitions": DEFINITION_PREDICATES,
    "codes": CODE_PREDICATES,
    "cross_references": CROSS_REFERENCE_PREDICATES,
}
# a canon element enters the default roles if it covers this share of a partition
# (keeps out e.g. two stray hasDbXref in an ontology of thousands of classes)
MIN_DEFAULT_COVERAGE = 0.01

# features whose shares inside and outside differ less are not shown
MIN_CONTRAST = 0.05

# local names per partition compared with names. 200 is enough to tell a share
# of 0.2 from one of 0.8: the 95 % Clopper-Pearson interval of a share is at
# most +-7 percentage points wide at n=200, whatever the share.
SAMPLE_SIZE = 200
SAMPLE_SEED = 0
# minimum named entities in the sample to judge local names by their names
MIN_LABELLED = 20
# local name agrees with a name: share of its tokens found in the name
MIN_TOKEN_AGREEMENT = 0.5
# share of local names that agree with names / look speaking, above which
# local names count as names and below which (speaking only) as codes
NAME_SHARE = 0.8
AGREEMENT_SHARE = 0.5
CODE_SHARE = 0.2

# prefix of an entity: scheme, host and the first path segment if more path
# follows (http://dbkwik.webdatacommons.org/stexpanded.wikia.com/), otherwise
# scheme and host (http://mouse.owl#MA_0000001 -> http://mouse.owl); without a
# host, up to the last # or / (as namespace_of)
_HOST = 'REPLACE(STR(?entity), "^([^:/]+://[^/#?]+).*$", "$1")'
_HOST_AND_SEGMENT = 'REPLACE(STR(?entity), "^([^:/]+://[^/#?]+/[^/#?]+/).*$", "$1")'
PREFIX_EXPRESSION = (
    f'IF(REGEX(STR(?entity), "^[^:/]+://[^/#?]+/[^/#?]+/"), {_HOST_AND_SEGMENT}, '
    f'IF(CONTAINS(STR(?entity), "://"), {_HOST}, REPLACE(STR(?entity), "[^#/]*$", "")))'
)
# characters not allowed in a SPARQL IRI reference
INVALID_IRI_CHARACTERS = re.compile(r'[\s<>"{}|^`\\]')

type Partitions = dict[str, PartitionDefinition]


def kind_partitions(kinds: Iterable[str] = KINDS) -> Partitions:
    return {kind: PartitionDefinition() for kind in kinds}


# --- partition queries ----------------------------------------------------------

def space_filter(space: Iterable[str]) -> str:
    space = tuple(space)
    if not space:
        return ""
    return "FILTER(" + " || ".join(f'STRSTARTS(STR(?entity), "{ns}")' for ns in space) + ")"


def prologue_of(prefixes: dict[str, str]) -> str:
    return "\n".join(f"PREFIX {short}: <{long}>" for short, long in prefixes.items())


def partition_subquery(manager: KgManager, partitions: Partitions | None,
                       space: Iterable[str] = ()) -> tuple[dict[str, str], str]:
    """(prefixes, group pattern) binding ?entity ?partition within the entity
    space. Without partitions: the kind partitions of KIND_PARTITION_SPARQL.
    Otherwise each entity gets the first partition, in order, whose membership
    matches (a membership pattern over ?entity, or for a default partition its
    kind), else UNCATEGORIZED. An entity of several kinds (punning) may get
    several partitions."""
    prefixes, kind_query = split_prologue(manager, KIND_PARTITION_SPARQL)
    space_part = space_filter(space)
    sample = current_sample(manager)
    if sample is not None and partitions is not None:
        # large KG: the sampled members of the partitions, assigned as below
        entities = sample.entities([d.membership for d in partitions.values() if d.membership is not None])
        return prefixes, (f"{{ SELECT DISTINCT ?entity ?partition WHERE {{\n{values_block(entities)}\n"
                          f"BIND({partition_choice(partitions)} AS ?partition)\n}} }}")
    if partitions is None:
        return prefixes, f"{{ SELECT DISTINCT ?entity ?partition WHERE {{\n{kind_query}\n}} }}\n{space_part}"
    entities = [f"{{ SELECT DISTINCT ?entity (?partition AS ?kind) WHERE {{\n{kind_query}\n}} }}"]
    entities += [f"{{ SELECT DISTINCT ?entity WHERE {{ {d.membership} }} }}"
                 for d in partitions.values() if d.membership is not None]
    body = (" UNION ".join(entities) + f"\nFILTER(isIRI(?entity))\n{space_part}\n"
            f"BIND({partition_choice(partitions)} AS ?partition)")
    return prefixes, f"{{ SELECT DISTINCT ?entity ?partition WHERE {{\n{body}\n}} }}"


def partition_choice(partitions: Partitions) -> str:
    """Expression of the partition of ?entity: the first, in order, whose
    membership matches (a pattern, or for a default partition its ?kind)."""
    choice = f'"{UNCATEGORIZED}"'
    for name, definition in reversed(partitions.items()):
        condition = (f"EXISTS {{ {definition.membership} }}" if definition.membership is not None
                     else f'COALESCE(?kind, "") = "{name}"')
        choice = f'IF({condition}, "{name}", {choice})'
    return choice


def estimated_sizes(manager: KgManager, partitions: Partitions, space: Iterable[str]) -> dict[str, int]:
    """Entities per partition: counted in a small KG, estimated from the
    samples of the memberships in a large one (kg_sample)."""
    sample = current_sample(manager)
    if sample is None:
        return partition_sizes(manager, partitions, space)
    rows = select_rows(manager, partition_query(manager, partitions, space, "?entity ?partition"))
    assigned = {(r["entity"].value, r["partition"].value) for r in rows if "entity" in r and "partition" in r}
    sizes = {}
    for name, definition in partitions.items():
        if definition.membership is None:
            continue
        members = sample.sample(definition.membership).in_space
        estimate = sample.estimated_size(definition.membership, sum((m, name) in assigned for m in members))
        if estimate is not None:
            sizes[name] = estimate
    return sizes


def union_with(group: str, parts: list[str], var: str | None = "s") -> str:
    """The alternatives (UNION) of a pattern, each joined with group in its own
    group, with ?var bound to ?entity (bind_entity). A server evaluates an
    alternative in braces on its own, so it would scan the whole KG for it if
    the group were outside."""
    bound = [bind_entity(part, var) if var else part for part in parts]
    if len(bound) == 1:
        return f"{group}\n{bound[0]}"
    return " UNION ".join(f"{{ {group}\n{part} }}" for part in bound)


def partition_query(manager: KgManager, partitions: Partitions | None, space: Iterable[str],
                    select: str, where: str = "", tail: str = "",
                    parts: list[str] | None = None, var: str = "s") -> str:
    """Query over the entities of the partitions (?entity ?partition), joined
    with the given patterns; with parts, joined with the alternatives of a
    pattern whose ?var is the entity (union_with)."""
    prefixes, subquery = partition_subquery(manager, partitions, space)
    body = union_with(subquery, parts, var) if parts is not None else subquery
    return f"{prologue_of(prefixes)}\nSELECT {select} WHERE {{\n{body}\n{where}\n}} {tail}"


def count_rows(rows: list[dict], *keys: str) -> dict[tuple[str, ...], int]:
    return {tuple(row[k].value for k in keys): int(row["n"].value) for row in rows if all(k in row for k in keys)}


def counts_by_partition(manager: KgManager, partitions: Partitions | None, space: Iterable[str],
                        where: str, parts: list[str] | None = None, var: str = "s") -> dict[str, int]:
    sparql = partition_query(manager, partitions, space, "?partition (COUNT(DISTINCT ?entity) AS ?n)",
                             where, "GROUP BY ?partition", parts, var)
    return {partition: n for (partition,), n in count_rows(select_rows(manager, sparql), "partition").items()}


def partition_sizes(manager: KgManager, partitions: Partitions | None, space: Iterable[str]) -> dict[str, int]:
    return counts_by_partition(manager, partitions, space, "")


# --- entity space ---------------------------------------------------------------

def prefix_counts(manager: KgManager, named_only: bool) -> dict[tuple[str, str], int]:
    """(prefix, kind partition) -> entities, optionally only entities with a
    canon name or synonym (keeps out e.g. image files of wiki KGs)."""
    named = (f"FILTER EXISTS {{ ?entity {sparql_path(LABEL_PREDICATES + SYNONYM_PREDICATES)} ?name }}"
             if named_only else "")
    sparql = partition_query(
        manager, None, (), "?prefix ?partition (COUNT(DISTINCT ?entity) AS ?n)",
        f"{named}\nBIND({PREFIX_EXPRESSION} AS ?prefix)", "GROUP BY ?prefix ?partition",
    )
    return count_rows(select_rows(manager, sparql), "prefix", "partition")


def prefix_totals(counts: dict[tuple[str, str], int]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for (prefix, _), n in counts.items():
        totals[prefix] += n
    return totals


def default_entity_space(manager: KgManager, config: GraspConfig) -> tuple[list[str], str]:
    """(entity space, how it was chosen): the configured namespaces of the KG,
    otherwise the prefix with the most named entities, otherwise the one with
    the most entities."""
    configured = configured_namespaces(config, manager.kg)
    if configured:
        return list(configured), "configured"
    for named_only, source in ((True, "prefix with the most named entities"),
                               (False, "prefix with the most entities")):
        totals = prefix_totals(prefix_counts(manager, named_only))
        if totals:
            return [totals.most_common(1)[0][0]], source
    return [], "no entities found"


# --- predicate coverage and distinguishing features ---------------------------------

class Coverage(BaseModel):
    entities: int        # entities with at least one value
    values: int = 0      # triples
    literals: int = 0    # triples with a literal value
    blanks: int = 0      # triples with a blank node value; the rest have an IRI


def predicate_coverage(manager: KgManager, partitions: Partitions | None, space: Iterable[str]
                       ) -> dict[str, dict[IRI, Coverage]]:
    """partition -> outgoing predicate (entities of the partition as subject)
    -> coverage, with the kinds of its values (objects)."""
    sparql = partition_query(
        manager, partitions, space,
        "?partition ?p (COUNT(DISTINCT ?entity) AS ?n) (COUNT(?v) AS ?m) "
        "(SUM(IF(isLiteral(?v), 1, 0)) AS ?lit) (SUM(IF(isBlank(?v), 1, 0)) AS ?blank)",
        "?entity ?p ?v .", "GROUP BY ?partition ?p",
    )
    coverage: dict[str, dict[IRI, Coverage]] = defaultdict(dict)
    for row in select_rows(manager, sparql):
        if "partition" in row and "p" in row:
            coverage[row["partition"].value][IRI(row["p"].value)] = Coverage(
                entities=int(row["n"].value), values=int(row["m"].value),
                literals=int(row["lit"].value) if "lit" in row else 0,
                blanks=int(row["blank"].value) if "blank" in row else 0,
            )
    return dict(coverage)


def type_coverage(manager: KgManager, partitions: Partitions | None, space: Iterable[str]
                  ) -> dict[str, dict[IRI, int]]:
    """partition -> rdf:type value -> entities."""
    sparql = partition_query(
        manager, partitions, space, "?partition ?t (COUNT(DISTINCT ?entity) AS ?n)",
        f"?entity <{RDF_TYPE}> ?t . FILTER(isIRI(?t))", "GROUP BY ?partition ?t",
    )
    coverage: dict[str, dict[IRI, int]] = defaultdict(dict)
    for (partition, t), n in count_rows(select_rows(manager, sparql), "partition", "t").items():
        coverage[partition][IRI(t)] = n
    return dict(coverage)


@dataclass
class Feature:
    """A predicate or type and its share inside a group and outside of it."""
    key: str
    inside: float
    outside: float

    @property
    def contrast(self) -> float:
        return self.inside - self.outside


def distinguishing_features(inside: dict[str, int], size_inside: int,
                            outside: dict[str, int], size_outside: int, k: int) -> list[Feature]:
    """The k features with the largest difference of shares, in either direction
    (common inside and rare outside, or the reverse), at least MIN_CONTRAST."""
    if not size_inside:
        return []
    features = [
        Feature(key, inside.get(key, 0) / size_inside, outside.get(key, 0) / size_outside if size_outside else 0.0)
        for key in set(inside) | set(outside)
    ]
    features = [f for f in features if abs(f.contrast) >= MIN_CONTRAST]
    return sorted(features, key=lambda f: (-abs(f.contrast), f.key))[:k]


def complement(counts: dict[str, dict[str, int]], sizes: dict[str, int], partition: str
               ) -> tuple[dict[str, int], int]:
    """Counts and size of all partitions except one, summed."""
    total: Counter[str] = Counter()
    for other, other_counts in counts.items():
        if other != partition:
            total.update(other_counts)
    return dict(total), sum(n for other, n in sizes.items() if other != partition)


def members_query(membership: str, space: Iterable[str], manager: KgManager | None = None) -> str:
    sample = current_sample(manager) if manager is not None else None
    if sample is not None:
        # large KG: the sampled members
        return values_block(sample.sample(membership).in_space)
    return f"{{ SELECT DISTINCT ?entity WHERE {{ {membership} }} }}\nFILTER(isIRI(?entity))\n{space_filter(space)}"


def membership_overlap(manager: KgManager, partitions: Partitions | None, space: Iterable[str],
                       membership: str) -> dict[str, int]:
    """Entities matching a membership pattern per current partition
    (UNCATEGORIZED for matches outside every partition); in a large KG for the
    sampled matches."""
    if current_sample(manager) is not None and partitions is not None:
        prefixes, _ = split_prologue(manager, KIND_PARTITION_SPARQL)
        sparql = (f"{prologue_of(prefixes)}\nSELECT ?part (COUNT(DISTINCT ?entity) AS ?n) WHERE {{\n"
                  f"{members_query(membership, space, manager)}\nBIND({partition_choice(partitions)} AS ?part)\n"
                  f"}} GROUP BY ?part")
        return {part: n for (part,), n in count_rows(select_rows(manager, sparql), "part").items()}
    prefixes, subquery = partition_subquery(manager, partitions, space)
    sparql = (f"{prologue_of(prefixes)}\nSELECT ?part (COUNT(DISTINCT ?entity) AS ?n) WHERE {{\n"
              f"{members_query(membership, space, manager)}\nOPTIONAL {subquery}\n"
              f'BIND(COALESCE(?partition, "{UNCATEGORIZED}") AS ?part)\n}} GROUP BY ?part')
    return {part: n for (part,), n in count_rows(select_rows(manager, sparql), "part").items()}


def member_coverage(manager: KgManager, space: Iterable[str], membership: str
                    ) -> tuple[dict[str, int], dict[str, int]]:
    """(predicate -> entities, rdf:type value -> entities) among the matches of a membership pattern."""
    members = members_query(membership, space, manager)
    predicates = select_rows(manager, f"SELECT ?p (COUNT(DISTINCT ?entity) AS ?n) WHERE {{\n{members}\n?entity ?p ?v }} GROUP BY ?p")
    types = select_rows(manager, (f"SELECT ?t (COUNT(DISTINCT ?entity) AS ?n) WHERE {{\n{members}\n"
                                  f"?entity <{RDF_TYPE}> ?t . FILTER(isIRI(?t)) }} GROUP BY ?t"))
    return ({p: n for (p,), n in count_rows(predicates, "p").items()},
            {t: n for (t,), n in count_rows(types, "t").items()})


# --- roles: canon and usage -------------------------------------------------------

def canon_roles() -> Roles:
    """All canon predicates and structure relations, used or not."""
    return Roles(
        structure=list(DEFAULT_STRUCTURE_RELATIONS),
        **{field: PredicatePattern(predicates=list(predicates)) for field, predicates in CANON.items()},
    )


def bind_entity(pattern: str, var: str = "s") -> str:
    """The pattern with ?var replaced by ?entity, followed by BIND(?entity AS ?var).
    Equivalent to binding ?var to ?entity first, but the server can then look
    up the pattern from the entities (index scan) instead of evaluating it on the
    whole KG and joining afterwards, which does not finish on a large KG."""
    return re.sub(rf"\?{var}\b", "?entity", pattern) + f"\nBIND(?entity AS ?{var})"


def element_parts(key: ElementKey, value) -> list[str]:
    """Alternatives of the pattern over ?s and ?o of one element of a Roles."""
    _, kind, item = key
    if kind == "relation":
        return value.predicate_pattern.alternatives()
    if kind == "predicate":
        return [f"?s <{item}> ?o"]
    return [item]


def element_sparql(key: ElementKey, value) -> str:
    """Pattern over ?s and ?o of one element of a Roles."""
    parts = element_parts(key, value)
    return parts[0] if len(parts) == 1 else " UNION ".join(f"{{ {p} }}" for p in parts)


# relation key -> (entities as ?s, entities as ?o)
type Sides = dict[ElementKey, tuple[int, int]]


def element_usage(manager: KgManager, partitions: Partitions | None, space: Iterable[str], roles: Roles,
                  sides: dict[str, Sides] | None = None) -> dict[str, Usage]:
    """partition -> element of the roles -> entities using it: as ?s for
    predicates and patterns; as ?s or ?o for structure relations, whichever
    side has more (e.g. classes are only ?o of instance_of), with both sides
    in sides if given. Predicates in one query, every pattern in its own,
    every relation in two."""
    usage: dict[str, Usage] = defaultdict(dict)
    elements = role_elements(roles)
    predicates = sorted({key[2] for key in elements if key[1] == "predicate"})
    if predicates:
        values = " ".join(f"<{p}>" for p in predicates)
        sparql = partition_query(manager, partitions, space, "?partition ?p (COUNT(DISTINCT ?entity) AS ?n)",
                                 f"VALUES ?p {{ {values} }}\n?entity ?p ?value .", "GROUP BY ?partition ?p")
        by_predicate = count_rows(select_rows(manager, sparql), "partition", "p")
        for key in elements:
            if key[1] == "predicate":
                for (partition, p), n in by_predicate.items():
                    if p == key[2]:
                        usage[partition][key] = n
    for key, value in elements.items():
        if key[1] == "pattern":
            for partition, n in counts_by_partition(manager, partitions, space, "", element_parts(key, value)).items():
                usage[partition][key] = n
        elif key[1] == "relation":
            parts = element_parts(key, value)
            as_s = counts_by_partition(manager, partitions, space, "", parts, "s")
            as_o = counts_by_partition(manager, partitions, space, "", parts, "o")
            for partition in set(as_s) | set(as_o):
                n_s, n_o = as_s.get(partition, 0), as_o.get(partition, 0)
                usage[partition][key] = max(n_s, n_o)
                if sides is not None:
                    sides.setdefault(partition, {})[key] = (n_s, n_o)
    return dict(usage)


def default_roles(usage: dict[str, Usage], sizes: dict[str, int], active: Iterable[str]) -> Roles:
    """Canon elements covering at least MIN_DEFAULT_COVERAGE of an active partition."""
    keep = {
        key for partition in active if sizes.get(partition)
        for key, n in usage.get(partition, {}).items() if n / sizes[partition] >= MIN_DEFAULT_COVERAGE
    }
    return roles_from_elements({k: v for k, v in role_elements(canon_roles()).items() if k in keep})


# --- local names --------------------------------------------------------------------

class LocalNameStats(BaseModel):
    """Local names of a sample of a partition compared with the entities' names."""
    sample_size: int = 0
    labelled: int = 0
    agreement: float | None = None  # share of named entities whose local name agrees with a name
    speaking: float | None = None   # share of local names that look like words


def sample_entities(manager: KgManager, partitions: Partitions | None, space: Iterable[str],
                    k: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED, where: str = "") -> dict[str, list[IRI]]:
    """Up to k random entities per partition, drawn while streaming (reservoir
    sampling), so memory stays at k per partition whatever the KG size."""
    rng = random.Random(seed)
    samples: dict[str, list[IRI]] = defaultdict(list)
    seen: Counter[str] = Counter()
    sparql = partition_query(manager, partitions, space, "?entity ?partition", where, "ORDER BY ?entity")
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


def retrieve_values(manager: KgManager, entities: Iterable[IRI], pattern: PredicatePattern | None,
                    literals_only: bool = True) -> dict[IRI, list[str]]:
    """?o of the pattern per entity (?s)."""
    values: dict[IRI, list[str]] = defaultdict(list)
    if pattern is None:
        return values
    literal = "FILTER(isLiteral(?o))" if literals_only else ""
    for block in values_blocks(entities):
        sparql = f"SELECT ?s ?o WHERE {{ {union_with(f'VALUES ?s {{ {block} }}', pattern.alternatives(), None)} {literal} }}"
        for row in select_rows(manager, sparql):
            if "s" in row and "o" in row:
                values[IRI(row["s"].value)].append(row["o"].value)
    return values


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


def local_name_type(stats: LocalNameStats) -> NameType:
    """Local names agreeing with names, or speaking without names to compare
    with, count as names. A speaking local name that differs from the name
    (e.g. another language) is reported in the statistics."""
    # small partitions (e.g. a handful of properties) count if all are labelled
    enough_labelled = stats.labelled >= min(MIN_LABELLED, stats.sample_size)
    if enough_labelled and stats.agreement is not None and stats.agreement >= AGREEMENT_SHARE:
        return NameType.LABEL
    if stats.speaking is None:
        return NameType.OTHER
    if stats.speaking >= NAME_SHARE:
        return NameType.LABEL
    if stats.speaking <= CODE_SHARE:
        return NameType.CODE
    return NameType.OTHER


def local_name_stats(manager: KgManager, sample: list[IRI], names: PredicatePattern | None) -> LocalNameStats:
    entity_names = retrieve_values(manager, sample, names)
    local_names = {entity: local_name(entity) for entity in sample}
    stats = LocalNameStats(sample_size=len(sample), labelled=sum(entity in entity_names for entity in sample))
    if sample:
        stats.speaking = sum(map(is_speaking_name, local_names.values())) / len(sample)
    if stats.labelled:
        stats.agreement = sum(
            agrees_with_label(local_names[entity], entity_names[entity]) for entity in sample if entity in entity_names
        ) / stats.labelled
    return stats
