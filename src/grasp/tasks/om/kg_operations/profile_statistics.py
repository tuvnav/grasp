"""
Queries and statistics behind the staged KG profile (data_models/profile.py):
entity space, predicate coverage, distinguishing features, the default canon
restricted to what the KG uses, and local names. Every statistic is computed per
member of a grouping (grouping.py), partitions and groups alike.

Scales with the KG: counting is done by the SPARQL server (GROUP BY), so only
aggregates are transferred, and the local name sample is drawn while streaming
the entities (reservoir sampling), so memory stays at the sample size.
"""
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import unquote

from pydantic import BaseModel

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.tasks.om.data_models.patterns import (
    DEFAULT_STRUCTURE_RELATIONS,
    PredicatePattern,
)
from grasp.tasks.om.data_models.profile import (
    ElementKey,
    Roles,
    Usage,
    role_elements,
    roles_from_elements,
)
from grasp.tasks.om.foundation.configuration import configured_namespaces
from grasp.tasks.om.foundation.types import IRI, NameType
from grasp.tasks.om.foundation.vocabulary import (
    CODE_PREDICATES,
    CROSS_REFERENCE_PREDICATES,
    DEFINITION_PREDICATES,
    LABEL_PREDICATES,
    RDF_TYPE,
    SYNONYM_PREDICATES,
    UNCATEGORIZED,
    local_name,
)
from grasp.tasks.om.kg_operations.grouping import (
    Grouping,
    count_rows,
    counts_by_group,
    grouping_to_sparql_query,
    grouping_to_sparql_subquery,
)
from grasp.tasks.om.kg_operations.large_kg_sampling import current_sample, values_block
from grasp.tasks.om.sparql_access.execution import select_rows, stream_rows
from grasp.tasks.om.sparql_access.query_building import (
    prologue_of,
    space_filter,
    sparql_path,
    down_pushed_union,
)
from grasp.tasks.om.sparql_access.values import values_blocks

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


def prefix_counts(manager: KgManager, named_only: bool) -> dict[tuple[str, str], int]:
    """(prefix, kind) -> entities, optionally only entities with a
    canon name or synonym (keeps out e.g. image files of wiki KGs)."""
    named = (f"FILTER EXISTS {{ ?entity {sparql_path(LABEL_PREDICATES + SYNONYM_PREDICATES)} ?name }}"
             if named_only else "")
    sparql = grouping_to_sparql_query(
        manager, None, (), "?prefix ?group (COUNT(DISTINCT ?entity) AS ?n)",
        f"{named}\nBIND({PREFIX_EXPRESSION} AS ?prefix)", "GROUP BY ?prefix ?group",
    )
    return count_rows(select_rows(manager, sparql), "prefix", "group")


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


class Coverage(BaseModel):
    entities: int        # entities with at least one value
    values: int = 0      # triples
    literals: int = 0    # triples with a literal value
    blanks: int = 0      # triples with a blank node value; the rest have an IRI


def predicate_coverage(manager: KgManager, grouping: Grouping | None, space: Iterable[str]
                       ) -> dict[str, dict[IRI, Coverage]]:
    """group -> outgoing predicate (entities of the group as subject)
    -> coverage, with the kinds of its values (objects)."""
    sparql = grouping_to_sparql_query(
        manager, grouping, space,
        "?group ?p (COUNT(DISTINCT ?entity) AS ?n) (COUNT(?v) AS ?m) "
        "(SUM(IF(isLiteral(?v), 1, 0)) AS ?lit) (SUM(IF(isBlank(?v), 1, 0)) AS ?blank)",
        "?entity ?p ?v .", "GROUP BY ?group ?p",
    )
    coverage: dict[str, dict[IRI, Coverage]] = defaultdict(dict)
    for row in select_rows(manager, sparql):
        if "group" in row and "p" in row:
            coverage[row["group"].value][IRI(row["p"].value)] = Coverage(
                entities=int(row["n"].value), values=int(row["m"].value),
                literals=int(row["lit"].value) if "lit" in row else 0,
                blanks=int(row["blank"].value) if "blank" in row else 0,
            )
    return dict(coverage)


def type_coverage(manager: KgManager, grouping: Grouping | None, space: Iterable[str]
                  ) -> dict[str, dict[IRI, int]]:
    """group -> rdf:type value -> entities."""
    sparql = grouping_to_sparql_query(
        manager, grouping, space, "?group ?t (COUNT(DISTINCT ?entity) AS ?n)",
        f"?entity <{RDF_TYPE}> ?t . FILTER(isIRI(?t))", "GROUP BY ?group ?t",
    )
    coverage: dict[str, dict[IRI, int]] = defaultdict(dict)
    for (group, t), n in count_rows(select_rows(manager, sparql), "group", "t").items():
        coverage[group][IRI(t)] = n
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


def complement(counts: dict[str, dict[str, int]], sizes: dict[str, int], group: str
               ) -> tuple[dict[str, int], int]:
    """Counts and size of all members of an exclusive grouping except one, summed."""
    total: Counter[str] = Counter()
    for other, other_counts in counts.items():
        if other != group:
            total.update(other_counts)
    return dict(total), sum(n for other, n in sizes.items() if other != group)


def rest(total: dict[str, int], total_size: int, inside: dict[str, int], size: int) -> tuple[dict[str, int], int]:
    """Counts and size of the entities of a total outside a part of it (a group
    of an overlapping grouping within all its entities)."""
    return {k: max(0, n - inside.get(k, 0)) for k, n in total.items()}, max(0, total_size - size)


def members_query(membership: str, space: Iterable[str], manager: KgManager | None = None) -> str:
    sample = current_sample(manager) if manager is not None else None
    if sample is not None:
        # large KG: the sampled members
        return values_block(sample.sample(membership).in_space)
    return f"{{ SELECT DISTINCT ?entity WHERE {{ {membership} }} }}\nFILTER(isIRI(?entity))\n{space_filter(space)}"


def membership_overlap(manager: KgManager, grouping: Grouping | None, space: Iterable[str],
                       membership: str) -> tuple[int, dict[str, int]]:
    """(entities matching a membership pattern, those per current group or
    partition, UNCATEGORIZED for matches in none); in a large KG for the
    sampled matches."""
    sample = current_sample(manager)
    if sample is not None:
        matches = sample.sample(membership).in_space
        counts: Counter[str] = Counter(g for _, g in sample.assignments(grouping, matches))
        assigned = {e for e, _ in sample.assignments(grouping, matches)}
        counts[UNCATEGORIZED] += sum(e not in assigned for e in matches)
        return len(matches), {g: n for g, n in counts.items() if n}
    prefixes, subquery = grouping_to_sparql_subquery(manager, grouping, space)
    sparql = (f"{prologue_of(prefixes)}\nSELECT ?part (COUNT(DISTINCT ?entity) AS ?n) WHERE {{\n"
              f"{members_query(membership, space, manager)}\nOPTIONAL {subquery}\n"
              f'BIND(COALESCE(?group, "{UNCATEGORIZED}") AS ?part)\n}} GROUP BY ?part')
    overlap = {part: n for (part,), n in count_rows(select_rows(manager, sparql), "part").items()}
    size = select_rows(manager, f"{prologue_of(prefixes)}\nSELECT (COUNT(DISTINCT ?entity) AS ?n) WHERE {{\n"
                                f"{members_query(membership, space, manager)}\n}}")
    return (int(size[0]["n"].value) if size and "n" in size[0] else 0), overlap


def member_coverage(manager: KgManager, space: Iterable[str], membership: str
                    ) -> tuple[dict[str, int], dict[str, int]]:
    """(predicate -> entities, rdf:type value -> entities) among the matches of a membership pattern."""
    members = members_query(membership, space, manager)
    predicates = select_rows(manager, f"SELECT ?p (COUNT(DISTINCT ?entity) AS ?n) WHERE {{\n{members}\n?entity ?p ?v }} GROUP BY ?p")
    types = select_rows(manager, (f"SELECT ?t (COUNT(DISTINCT ?entity) AS ?n) WHERE {{\n{members}\n"
                                  f"?entity <{RDF_TYPE}> ?t . FILTER(isIRI(?t)) }} GROUP BY ?t"))
    return ({p: n for (p,), n in count_rows(predicates, "p").items()},
            {t: n for (t,), n in count_rows(types, "t").items()})


def canon_roles() -> Roles:
    """All canon predicates and structure relations, used or not."""
    return Roles(
        structure=list(DEFAULT_STRUCTURE_RELATIONS),
        **{field: PredicatePattern(predicates=list(predicates)) for field, predicates in CANON.items()},
    )


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


def element_usage(manager: KgManager, grouping: Grouping | None, space: Iterable[str], roles: Roles,
                  sides: dict[str, Sides] | None = None) -> dict[str, Usage]:
    """group -> element of the roles -> entities using it: as ?s for
    predicates and patterns; as ?s or ?o for structure relations, whichever
    side has more (e.g. classes are only ?o of instance_of), with both sides
    in sides if given. Predicates in one query, every pattern in its own,
    every relation in two."""
    usage: dict[str, Usage] = defaultdict(dict)
    elements = role_elements(roles)
    predicates = sorted({key[2] for key in elements if key[1] == "predicate"})
    if predicates:
        values = " ".join(f"<{p}>" for p in predicates)
        sparql = grouping_to_sparql_query(manager, grouping, space, "?group ?p (COUNT(DISTINCT ?entity) AS ?n)",
                                 f"VALUES ?p {{ {values} }}\n?entity ?p ?value .", "GROUP BY ?group ?p")
        by_predicate = count_rows(select_rows(manager, sparql), "group", "p")
        for key in elements:
            if key[1] == "predicate":
                for (group, p), n in by_predicate.items():
                    if p == key[2]:
                        usage[group][key] = n
    for key, value in elements.items():
        if key[1] == "pattern":
            for group, n in counts_by_group(manager, grouping, space, "", element_parts(key, value)).items():
                usage[group][key] = n
        elif key[1] == "relation":
            parts = element_parts(key, value)
            as_s = counts_by_group(manager, grouping, space, "", parts, "s")
            as_o = counts_by_group(manager, grouping, space, "", parts, "o")
            for group in set(as_s) | set(as_o):
                n_s, n_o = as_s.get(group, 0), as_o.get(group, 0)
                usage[group][key] = max(n_s, n_o)
                if sides is not None:
                    sides.setdefault(group, {})[key] = (n_s, n_o)
    return dict(usage)


def default_roles(usage: dict[str, Usage], sizes: dict[str, int], active: Iterable[str]) -> Roles:
    """Canon elements covering at least MIN_DEFAULT_COVERAGE of an active group or partition."""
    keep = {
        key for group in active if sizes.get(group)
        for key, n in usage.get(group, {}).items() if n / sizes[group] >= MIN_DEFAULT_COVERAGE
    }
    return roles_from_elements({k: v for k, v in role_elements(canon_roles()).items() if k in keep})


class LocalNameStats(BaseModel):
    """Local names of a sample of a group compared with the entities' names."""
    sample_size: int = 0
    labelled: int = 0
    agreement: float | None = None  # share of named entities whose local name agrees with a name
    speaking: float | None = None   # share of local names that look like words


def sample_entities(manager: KgManager, grouping: Grouping | None, space: Iterable[str],
                    k: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED, where: str = "") -> dict[str, list[IRI]]:
    """Up to k random entities per group, drawn while streaming (reservoir
    sampling), so memory stays at k per group whatever the KG size."""
    rng = random.Random(seed)
    samples: dict[str, list[IRI]] = defaultdict(list)
    seen: Counter[str] = Counter()
    sparql = grouping_to_sparql_query(manager, grouping, space, "?entity ?group", where, "ORDER BY ?entity")
    for row in stream_rows(manager, sparql):
        entity, group = IRI(row["entity"]["value"]), row["group"]["value"]
        if INVALID_IRI_CHARACTERS.search(entity):
            continue
        seen[group] += 1
        if len(samples[group]) < k:
            samples[group].append(entity)
        elif (j := rng.randrange(seen[group])) < k:
            samples[group][j] = entity
    return dict(samples)


def retrieve_values(manager: KgManager, entities: Iterable[IRI], pattern: PredicatePattern | None,
                    literals_only: bool = True) -> dict[IRI, list[str]]:
    """?o of the pattern per entity (?s)."""
    values: dict[IRI, list[str]] = defaultdict(list)
    if pattern is None:
        return values
    literal = "FILTER(isLiteral(?o))" if literals_only else ""
    for block in values_blocks(entities):
        sparql = f"SELECT ?s ?o WHERE {{ {down_pushed_union(f'VALUES ?s {{ {block} }}', pattern.alternatives(), None)} {literal} }}"
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
    # small groups (e.g. a handful of properties) count if all are labelled
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
