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
from grasp.tasks.om.data_models.profile import (
    DEFAULT_STRUCTURE_RELATIONS,
    GroupProfile,
    KgProfile,
    PredicatePattern,
    RoleElement,
    PredicateRoles,
    RoleElementKind,
    RoleElementUsage,
    effective_roles,
    role_elements_of,
    roles_from_elements,
    swap,
)
from grasp.tasks.om.foundation.configuration import configured_namespaces
from grasp.tasks.om.foundation.types import IRI, LocalNameType
from grasp.tasks.om.foundation.rdf_vocabulary import (
    CODE_PREDICATES,
    CROSS_REFERENCE_PREDICATES,
    DEFINITION_PREDICATES,
    DEFAULT_LABEL_PREDICATES,
    RDF_TYPE,
    SYNONYM_PREDICATES,
    UNCATEGORIZED,
    local_name,
)
from grasp.tasks.om.kg_operations.grouping import (
    Grouping,
    count_rows,
    retrieve_counts_by_group,
    grouping_to_sparql_query,
    grouping_to_sparql_subquery,
    single_group_grouping,
)
from grasp.tasks.om.sparql_access import (
    select_rows,
    stream_rows,
    make_sparql_prologue,
    sparql_filter_by_space,
    union_of,
    down_pushed_union,
    as_batched_values_blocks,
)

# a KG with more triples, or whose triples cannot be counted, is not supported yet
LARGE_TRIPLES = 100_000_000


def count_triples(manager: KgManager) -> int | None:
    """
    Triples of the KG, None if the server cannot count them (e.g. a timeout).
    """
    try:
        rows = select_rows(manager, "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }")
        return int(rows[0]["n"].value) if rows and "n" in rows[0] else None
    except Exception:
        return None


def is_large(triples: int | None) -> bool:
    return triples is None or triples > LARGE_TRIPLES


# canon predicates per role field of Roles
CANON = {
    "names": DEFAULT_LABEL_PREDICATES,
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
HOST_EXPRESSION = 'REPLACE(STR(?s), "^([^:/]+://[^/#?]+).*$", "$1")'
HOST_AND_SEGMENT_EXPRESSION = 'REPLACE(STR(?s), "^([^:/]+://[^/#?]+/[^/#?]+/).*$", "$1")'
PREFIX_EXPRESSION = (
    f'IF(REGEX(STR(?s), "^[^:/]+://[^/#?]+/[^/#?]+/"), {HOST_AND_SEGMENT_EXPRESSION}, '
    f'IF(CONTAINS(STR(?s), "://"), {HOST_EXPRESSION}, REPLACE(STR(?s), "[^#/]*$", "")))'
)
# characters not allowed in a SPARQL IRI reference
INVALID_IRI_CHARACTERS = re.compile(r'[\s<>"{}|^`\\]')


def prefix_per_group_counts(
        manager: KgManager, named_only: bool
        ) -> dict[tuple[str, str], int]:
    """
    (prefix, kind) -> entities, optionally only entities with a
    canon name or synonym (keeps out e.g. image files of wiki KGs).
    """
    named = (
        f"""FILTER EXISTS {{ {union_of([
            f'?s <{p}> ?name'
            for p in DEFAULT_LABEL_PREDICATES + SYNONYM_PREDICATES]
            )}  }}"""

        if named_only else ""
        )
    sparql = grouping_to_sparql_query(
        manager, None, (), "?prefix ?group (COUNT(DISTINCT ?s) AS ?n)",
        f"{named}\nBIND({PREFIX_EXPRESSION} AS ?prefix)", "GROUP BY ?prefix ?group",
    )
    return count_rows(select_rows(manager, sparql), "prefix", "group")


def prefix_totals(counts: dict[tuple[str, str], int]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for (prefix, _), n in counts.items():
        totals[prefix] += n
    return totals


def default_entity_space(manager: KgManager, config: GraspConfig) -> tuple[list[str], str]:
    """
    (entity space, how it was chosen): the configured namespaces of the KG,
    otherwise the prefix with the most named entities, otherwise the one with
    the most entities.
    """
    configured = configured_namespaces(config, manager.kg)
    if configured:
        return list(configured), "configured"
    for named_only, source in ((True, "prefix with the most named entities"),
                               (False, "prefix with the most entities")):
        totals = prefix_totals(prefix_per_group_counts(manager, named_only))
        if totals:
            return [totals.most_common(1)[0][0]], source
    return [], "no entities found"


def ensure_space(
        kg_manager: KgManager,
        kg_profile: KgProfile,
        config: GraspConfig | None = None
        ) -> list[str]:
    if kg_profile.entity_space:
        return list(kg_profile.entity_space)
    if config:
        space, _ = default_entity_space(kg_manager, config)
        return space
    raise ValueError(
        "Given KgProfile doesn't contain entity namespace and no GraspConfig was given."
        "Either is required."
        )


class PredicateCoverage(BaseModel):
    entities: int        # entities with at least one value
    values: int = 0      # triples
    literals: int = 0    # triples with a literal value
    blanks: int = 0      # triples with a blank node value; the rest have an IRI


def predicate_coverage(
    manager: KgManager, grouping: Grouping | None, space: Iterable[str]
        ) -> dict[str, dict[IRI, PredicateCoverage]]:
    """
    group -> outgoing predicate (entities of the group as subject)
    -> coverage, with the kinds of its values (objects).
    """
    sparql = grouping_to_sparql_query(
        manager, grouping, space,
        "?group ?p (COUNT(DISTINCT ?s) AS ?n) (COUNT(?o) AS ?m) "
        "(SUM(IF(isLiteral(?o), 1, 0)) AS ?lit) (SUM(IF(isBlank(?o), 1, 0)) AS ?blank)",
        "?s ?p ?o .", "GROUP BY ?group ?p",
    )
    coverage: dict[str, dict[IRI, PredicateCoverage]] = defaultdict(dict)
    for row in select_rows(manager, sparql):
        if "group" in row and "p" in row:
            coverage[row["group"].value][IRI(row["p"].value)] = PredicateCoverage(
                entities=int(row["n"].value), values=int(row["m"].value),
                literals=int(row["lit"].value) if "lit" in row else 0,
                blanks=int(row["blank"].value) if "blank" in row else 0,
            )
    return dict(coverage)


def type_coverage(
    manager: KgManager, grouping: Grouping | None, space: Iterable[str]
        ) -> dict[str, dict[IRI, int]]:
    """group -> rdf:type value -> entities."""
    sparql = grouping_to_sparql_query(
        manager, grouping, space, "?group ?t (COUNT(DISTINCT ?s) AS ?n)",
        f"?s <{RDF_TYPE}> ?t . FILTER(isIRI(?t))", "GROUP BY ?group ?t",
    )
    coverage: dict[str, dict[IRI, int]] = defaultdict(dict)
    for (group, t), n in count_rows(
            select_rows(manager, sparql), "group", "t"
            ).items():
        coverage[group][IRI(t)] = n
    return dict(coverage)


@dataclass
class PredicateUsage:
    """A predicate or type and its share inside a group and outside of it."""
    key: str
    share_inside: float
    share_outside: float

    @property
    def contrast(self) -> float:
        return self.share_inside - self.share_outside


def get_predicate_usage(
    usage_inside: dict[str, int],
    size_inside: int,
    usage_outside: dict[str, int],
    size_outside: int,
    k: int
        ) -> tuple[list[PredicateUsage], list[PredicateUsage]]:
    """
    The k predicates with the largest difference of shares, in either direction
    (common inside and rare outside, or the reverse), at least MIN_CONTRAST.
    """
    if not size_inside:
        return ([], [])
    relevant_predicates = [
        PredicateUsage(
            key,
            usage_inside.get(key, 0) / size_inside,
            usage_outside.get(key, 0) / size_outside if size_outside else 0.0
            )
        for key in set(usage_inside) | set(usage_outside)
    ]
    relevant_predicates = [
        f for f in relevant_predicates if abs(f.contrast) >= MIN_CONTRAST
        ]
    used_often = sorted(
        (p for p in relevant_predicates if p.contrast > 0),
        key=lambda f: (-abs(f.contrast), f.key)
        )
    used_seldom = sorted(
        (p for p in relevant_predicates if p.contrast < 0),
        key=lambda f: (-abs(f.contrast), f.key)
    )
    return (used_often[:k], used_seldom[:k])


def complement(
    counts: dict[str, dict[str, int]], sizes: dict[str, int], group: str
        ) -> tuple[dict[str, int], int]:
    """
    Counts and size of all members of an exclusive grouping except one, summed.
    """
    total: Counter[str] = Counter()
    for other, other_counts in counts.items():
        if other != group:
            total.update(other_counts)
    return dict(total), sum(n for other, n in sizes.items() if other != group)


def rest(total: dict[str, int], total_size: int, inside: dict[str, int], size: int) -> tuple[dict[str, int], int]:
    """
    Counts and size of the entities of a total outside a part of it (a group
    of an overlapping grouping within all its entities).
    """
    return {
        k: max(0, n - inside.get(k, 0))
        for k, n in total.items()
        }, max(0, total_size - size)


def members_query(membership: str, space: Iterable[str]) -> str:
    return f"{{ SELECT DISTINCT ?s WHERE {{ {membership} }} }}\nFILTER(isIRI(?s))\n{sparql_filter_by_space(space)}"


def membership_overlap(
    manager: KgManager,
    grouping: Grouping | None,
    space: Iterable[str],
        membership: str
        ) -> tuple[int, dict[str, int]]:
    """
    (entities matching a membership pattern, those per current group or
    partition, UNCATEGORIZED for matches in none).
    """
    prefixes, subquery = grouping_to_sparql_subquery(manager, grouping, space)
    sparql = (
        f"{make_sparql_prologue(prefixes)}\nSELECT ?part (COUNT(DISTINCT ?s) AS ?n) WHERE {{\n"
        f"{members_query(membership, space)}\nOPTIONAL {subquery}\n"
        f'BIND(COALESCE(?group, "{UNCATEGORIZED}") AS ?part)\n}} GROUP BY ?part'
        )
    overlap = {part: n for (part,), n in count_rows(select_rows(manager, sparql), "part").items()}
    size = select_rows(manager, f"{make_sparql_prologue(prefixes)}\nSELECT (COUNT(DISTINCT ?s) AS ?n) WHERE {{\n"
                       f"{members_query(membership, space)}\n}}")
    return (int(size[0]["n"].value) if size and "n" in size[0] else 0), overlap


def member_coverage(
    manager: KgManager, space: Iterable[str], membership: str
        ) -> tuple[dict[str, int], dict[str, int]]:
    """
    (predicate -> entities, rdf:type value -> entities) among the matches of a membership pattern.
    """
    members = members_query(membership, space)
    predicates = select_rows(manager, f"SELECT ?p (COUNT(DISTINCT ?s) AS ?n) WHERE {{\n{members}\n?s ?p ?o }} GROUP BY ?p")
    types = select_rows(manager, (f"SELECT ?t (COUNT(DISTINCT ?s) AS ?n) WHERE {{\n{members}\n"
                                  f"?s <{RDF_TYPE}> ?t . FILTER(isIRI(?t)) }} GROUP BY ?t"))
    return ({p: n for (p,), n in count_rows(predicates, "p").items()},
            {t: n for (t,), n in count_rows(types, "t").items()})


def canon_roles() -> PredicateRoles:
    """
    All canon predicates and structure relations, used or not.
    """
    return PredicateRoles(
        structure=list(DEFAULT_STRUCTURE_RELATIONS),
        **{field: PredicatePattern(predicates=list(predicates)) for field, predicates in CANON.items()},
    )


# element of a structure relation -> (entities as ?s, entities as ?o)
type DirectionalUsage = dict[RoleElement, tuple[int, int]]


def count_entities_by_predicate(
        manager: KgManager,
        grouping: Grouping | None,
        space: Iterable[str],
        predicates: Iterable[str],
        triple: str
        ) -> dict[str, dict[str, int]]:
    """
    predicate -> group name -> num entities of the group in the triple pattern
    (?s the entity, ?p the predicate), for all predicates in one query.
    """
    predicates = sorted(set(predicates))
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    if not predicates:
        return counts
    values = " ".join(f"<{p}>" for p in predicates)
    sparql = grouping_to_sparql_query(
        manager,
        grouping,
        space,
        "?group ?p (COUNT(DISTINCT ?s) AS ?n)",
        f"VALUES ?p {{ {values} }}\n{triple} .",
        "GROUP BY ?group ?p"
        )
    for (group, predicate), n in count_rows(
            select_rows(manager, sparql), "group", "p"
            ).items():
        counts[predicate][group] = n
    return counts


def element_usage_in_roles(
        manager: KgManager,
        grouping: Grouping | None,
        space: Iterable[str],
        roles: PredicateRoles,
        usage_by_direction: dict[str, DirectionalUsage] | None = None
        ) -> dict[str, RoleElementUsage]:
    """
    group name -> element of the roles -> number of entities using it.
    Elements of role fields count their entities as ?s; parts of structure
    relations as ?s or as ?o, whichever is more (e.g. classes are only ?o of
    instance_of), with both numbers in usage_by_direction if given. Relation
    properties are not counted. The predicates in one query per direction,
    every pattern in its own (an invalid pattern fails alone).
    """
    usage: dict[str, RoleElementUsage] = defaultdict(dict)
    elements = role_elements_of(roles)
    non_structure_predicates = [
        e for e in elements if e.kind == RoleElementKind.PREDICATE
        and not e.is_structure_relation
        ]
    structure_predicates = [
        e for e in elements if e.kind == RoleElementKind.PREDICATE
        and e.is_structure_relation
        ]
    non_structure_patterns = [
        e for e in elements if e.kind == RoleElementKind.PATTERN
        and not e.is_structure_relation
        ]
    structure_patterns = [
        e for e in elements if e.kind == RoleElementKind.PATTERN
        and e.is_structure_relation
        ]

    # non structure fields: count entities as ?s
    non_structure_counts = count_entities_by_predicate(
        manager,
        grouping,
        space,
        (str(e.definition) for e in non_structure_predicates),
        "?s ?p ?o"
        )
    for element in non_structure_predicates:
        for group, n in non_structure_counts.get(str(element.definition), {}).items():
            usage[group][element] = n
    for element in non_structure_patterns:
        for group, n in retrieve_counts_by_group(
                manager, grouping, space, str(element.sparql())
                ).items():
            usage[group][element] = n

    # structure relations: entities as ?s and as ?o
    relation_iris = [str(e.definition) for e in structure_predicates]
    as_subject = count_entities_by_predicate(
        manager, grouping, space, relation_iris, "?s ?p ?o"
        )
    as_object = count_entities_by_predicate(
        manager, grouping, space, relation_iris, "?o ?p ?s"
        )
    counts_by_direction: dict[RoleElement, tuple[dict[str, int], dict[str, int]]] = {
        element: (
            as_subject.get(str(element.definition), {}),
            as_object.get(str(element.definition), {})
            )
        for element in structure_predicates
        }
    for element in structure_patterns:
        pattern = str(element.sparql())
        counts_by_direction[element] = (
            retrieve_counts_by_group(manager, grouping, space, pattern),
            # the entities as ?o: the pattern with ?s and ?o swapped
            retrieve_counts_by_group(manager, grouping, space, swap(pattern)),
            )
    for element, (subject_counts, object_counts) in counts_by_direction.items():
        for group in set(subject_counts) | set(object_counts):
            n_s, n_o = subject_counts.get(group, 0), object_counts.get(group, 0)
            usage[group][element] = max(n_s, n_o)
            if usage_by_direction is not None:
                usage_by_direction.setdefault(group, {})[element] = (n_s, n_o)
    return dict(usage)


def default_roles(
        role_element_usage: dict[str, RoleElementUsage],
        group_sizes: dict[str, int],
        active_groups: Iterable[str]
        ) -> PredicateRoles:
    """
    Canon elements covering at least MIN_DEFAULT_COVERAGE of
    an active group or partition.
    """
    kept_role_elements_without_structure_properties = {
        role_element for group in active_groups if group_sizes.get(group)
        for role_element, n in role_element_usage.get(group, {}).items()
        if n / group_sizes[group] >= MIN_DEFAULT_COVERAGE
    }
    kept_structure_relation_names = {
        element.role_name for element in kept_role_elements_without_structure_properties
        if element.is_structure_relation
    }
    return roles_from_elements(
        frozenset(
            element for element in role_elements_of(canon_roles())
            if element in kept_role_elements_without_structure_properties
            or (
                element.kind == RoleElementKind.RELATION_PROPERTIES
                and element.role_name in kept_structure_relation_names
                )
            )
        )


class LocalNameStats(BaseModel):
    """
    Local names of a sample of a group compared with the entities' names.
    """
    sample_size: int = 0
    labelled: int = 0
    agreement: float | None = None  # share of named entities whose local name agrees with a name
    speaking: float | None = None   # share of local names that look like words


def sample_entities(
        manager: KgManager,
        grouping: Grouping | None,
        space: Iterable[str],
        k: int = SAMPLE_SIZE,
        seed: int = SAMPLE_SEED,
        where: str = "") -> dict[str, list[IRI]]:
    """
    Up to k random entities per group, drawn while streaming (reservoir
    sampling), so memory stays at k per group whatever the KG size.
    """
    rng = random.Random(seed)
    samples: dict[str, list[IRI]] = defaultdict(list)
    seen: Counter[str] = Counter()
    sparql = grouping_to_sparql_query(manager, grouping, space, "?s ?group", where, "ORDER BY ?s")
    for row in stream_rows(manager, sparql):
        entity, group = IRI(row["s"]["value"]), row["group"]["value"]
        if INVALID_IRI_CHARACTERS.search(entity):
            continue
        seen[group] += 1
        if len(samples[group]) < k:
            samples[group].append(entity)
        elif (j := rng.randrange(seen[group])) < k:
            samples[group][j] = entity
    return dict(samples)


def retrieve_values(
        manager: KgManager,
        entities: Iterable[IRI],
        pattern: PredicatePattern | None,
        literals_only: bool = True
        ) -> dict[IRI, list[str]]:
    """?o of the pattern per entity (?s)."""
    values: dict[IRI, list[str]] = defaultdict(list)
    if pattern is None:
        return values
    literal = "FILTER(isLiteral(?o))" if literals_only else ""
    for block in as_batched_values_blocks(entities):
        sparql = f"SELECT ?s ?o WHERE {{ {down_pushed_union(f'VALUES ?s {{ {block} }}', pattern.branches_for_union())} {literal} }}"
        for row in select_rows(manager, sparql):
            if "s" in row and "o" in row:
                values[IRI(row["s"].value)].append(row["o"].value)
    return values


NAME_TOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def name_tokens(text: str) -> list[str]:
    """Lowercased word and number tokens; splits camelCase, _ - ( ) and percent escapes."""
    return [t.lower() for t in NAME_TOKEN.findall(unquote(text))]


def is_speaking_name(name: str) -> bool:
    """
    Whether a local name looks like words rather than a code (MA_0000001,
    NCI_C12345, relation_22321): more than half of its tokens are words of at
    least three letters with a vowel. Single letters are not counted (Class_N),
    numbers are. Form only: a speaking name may still be an identifier.
    """
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


def local_name_type(stats: LocalNameStats) -> LocalNameType:
    """Local names agreeing with names, or speaking without names to compare
    with, count as names. A speaking local name that differs from the name
    (e.g. another language) is reported in the statistics."""
    # small groups (e.g. a handful of properties) count if all are labelled
    enough_labelled = stats.labelled >= min(MIN_LABELLED, stats.sample_size)
    if enough_labelled and stats.agreement is not None and stats.agreement >= AGREEMENT_SHARE:
        return LocalNameType.LABEL
    if stats.speaking is None:
        return LocalNameType.OTHER
    if stats.speaking >= NAME_SHARE:
        return LocalNameType.LABEL
    if stats.speaking <= CODE_SHARE:
        return LocalNameType.CODE
    return LocalNameType.OTHER


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


def group_name_stats(
    manager: KgManager,
    grouping: Grouping,
    space: list[str],
    group: str,
        roles: PredicateRoles
        ) -> LocalNameStats:
    sample = sample_entities(manager, grouping, space, where=f'FILTER(?group = "{group}")')
    return local_name_stats(manager, sample.get(group, []), roles.names)


def group_defaults(
        manager: KgManager, profile: KgProfile, group: str
        ) -> tuple[GroupProfile, LocalNameStats]:
    """
    Profile of a group without LLM: the local name type from its sample, its delta unchanged.
    """
    grouping = single_group_grouping(profile, group)
    current = profile.get_group(group)
    usage = element_usage_in_roles(manager, grouping, profile.entity_space, profile.roles).get(group, {})
    effective = effective_roles(profile.roles, current, usage)
    stats = group_name_stats(manager, grouping, profile.entity_space, group, effective)
    return current.model_copy(update={"local_name_type": local_name_type(stats)}, deep=True), stats


def unassigned_predicates(
        roles: PredicateRoles, coverage: dict[str, PredicateCoverage]
        ) -> list[tuple[str, PredicateCoverage]]:
    """
    Predicates of a group in no element of its roles (as predicate or
    inside a pattern), most frequent first.
    """
    elements = role_elements_of(roles)
    predicates = {
        str(e.definition) for e in elements if e.kind == RoleElementKind.PREDICATE
        }
    patterns = [
        str(e.definition) for e in elements if e.kind == RoleElementKind.PATTERN
    ]
    covered = lambda p: p in predicates or any(
        f"<{p}>" in pattern for pattern in patterns
        )
    return [
        (p, c) for p, c in sorted(
            coverage.items(), key=lambda x: -x[1].entities
            ) if not covered(p)
        ]
