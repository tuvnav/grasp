"""
Large KG mode of the profile (e.g. dblp with 1.6 billion triples, or Wikidata):
queries that collect all entities of a KG (the kind partitions, prefix counts)
do not finish there. Instead:
- entities are partitioned by their types (membership patterns), not by kind;
- per membership pattern, a sample of members is drawn at evenly spread
  offsets (cheap index scans) and its members are counted on the server;
- all statistics of the profile (coverage, value kinds, usage of patterns,
  local names, examples) are computed on the union of the samples: the
  partition subquery becomes VALUES of the sampled entities, so every query of
  kg_profile works unchanged and scales with the sample, not with the KG;
- partition sizes are estimated from the member counts and the share of each
  sample that falls into the partition (first match in order, as in a small KG).

The sample of a KG is registered per KG name while a stage runs
(use_sample), so the query functions of kg_profile pick it up.
"""
from collections import Counter
from dataclasses import dataclass, field
import math
import re

from grasp.manager import KgManager
from grasp.tasks.om.model import IRI, LABEL_PREDICATES, RDF_TYPE, SYNONYM_PREDICATES, local_name, sparql_path
from grasp.tasks.om.retrieve import select_rows, values_clause

# a KG with more triples, or whose triples cannot be counted, is profiled in the large mode
LARGE_TRIPLES = 100_000_000
# sampled members per membership pattern, drawn in this many chunks at spread offsets
SAMPLE_PER_MEMBERSHIP = 200
SAMPLE_CHUNKS = 10
# stage a: sampled members per class to tell its prefixes and named share
CLASS_SAMPLE = 30
# stage b: classes with fewer estimated members in the entity space get no default partition
MIN_CLASS_MEMBERS = 1000
MAX_DEFAULT_PARTITIONS = 30


def count_triples(manager: KgManager) -> int | None:
    """Triples of the KG, None if the server cannot count them (e.g. a timeout)."""
    try:
        rows = select_rows(manager, "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }")
        return int(rows[0]["n"].value) if rows and "n" in rows[0] else None
    except Exception:
        return None


def is_large(triples: int | None) -> bool:
    return triples is None or triples > LARGE_TRIPLES


def type_membership(type_iri: str) -> str:
    return f"?entity <{RDF_TYPE}> <{type_iri}>"


def class_counts(manager: KgManager) -> dict[IRI, int]:
    """rdf:type value -> typed subjects (an index scan, fast even for billions of triples)."""
    rows = select_rows(manager, f"SELECT ?t (COUNT(?s) AS ?n) WHERE {{ ?s <{RDF_TYPE}> ?t }} GROUP BY ?t")
    return {IRI(r["t"].value): int(r["n"].value) for r in rows if "t" in r and "n" in r and r["t"].typ == "uri"}


def spread_offsets(count: int, k: int, chunks: int) -> list[tuple[int, int]]:
    """(offset, limit) of chunks spread evenly over count members, k members in all."""
    if count <= k:
        return [(0, count)]
    chunks = min(chunks, k)
    per_chunk = math.ceil(k / chunks)
    step = count // chunks
    return [(i * step, per_chunk) for i in range(chunks)]


def draw_members(manager: KgManager, membership: str, count: int, k: int, chunks: int = SAMPLE_CHUNKS) -> list[IRI]:
    members: dict[IRI, None] = {}
    for offset, limit in spread_offsets(count, k, chunks):
        rows = select_rows(manager, f"SELECT DISTINCT ?entity WHERE {{ {membership} FILTER(isIRI(?entity)) }} "
                                    f"OFFSET {offset} LIMIT {limit}")
        for row in rows:
            if "entity" in row:
                members[IRI(row["entity"].value)] = None
    return list(members)


def count_members(manager: KgManager, membership: str) -> int | None:
    try:
        rows = select_rows(manager, f"SELECT (COUNT(DISTINCT ?entity) AS ?n) WHERE {{ {membership} FILTER(isIRI(?entity)) }}")
        return int(rows[0]["n"].value) if rows and "n" in rows[0] else None
    except Exception:
        return None


@dataclass
class MembershipSample:
    count: int | None      # members in the whole KG (None: not countable)
    members: list[IRI]     # sampled members, whole KG
    in_space: list[IRI]    # those within the entity space

    @property
    def space_share(self) -> float:
        return len(self.in_space) / len(self.members) if self.members else 0.0


@dataclass
class KgSample:
    """Samples of the membership patterns of one KG within its entity space."""
    manager: KgManager
    space: tuple[str, ...]
    per_membership: int = SAMPLE_PER_MEMBERSHIP
    samples: dict[str, MembershipSample] = field(default_factory=dict)

    def sample(self, membership: str) -> MembershipSample:
        membership = membership.strip()
        if membership not in self.samples:
            key = (self.manager.kg, self.space, self.per_membership, membership)
            if key not in _CACHE:
                count = count_members(self.manager, membership)
                # without a count, draw from the beginning only
                members = draw_members(self.manager, membership, count if count is not None else self.per_membership,
                                       self.per_membership, SAMPLE_CHUNKS if count is not None else 1)
                in_space = [m for m in members if not self.space or m.startswith(self.space)]
                _CACHE[key] = MembershipSample(count, members, in_space)
            self.samples[membership] = _CACHE[key]
        return self.samples[membership]

    def entities(self, memberships: list[str]) -> list[IRI]:
        """Union of the in-space samples of the memberships."""
        entities: dict[IRI, None] = {}
        for membership in memberships:
            entities.update(dict.fromkeys(self.sample(membership).in_space))
        return list(entities)

    def estimated_size(self, membership: str, assigned: int) -> int | None:
        """Members in the entity space that fall into the partition of this
        membership: the member count times the share of its sample that is in
        the space and assigned to the partition."""
        sample = self.sample(membership)
        if sample.count is None or not sample.members:
            return None
        return round(sample.count * assigned / len(sample.members))


# KG name -> sample of the stage running on it
_SAMPLES: dict[str, KgSample] = {}
# drawn samples, shared by the stages of one process (the samples are deterministic)
_CACHE: dict[tuple, MembershipSample] = {}


def use_sample(manager: KgManager, sample: KgSample | None) -> None:
    if sample is None:
        _SAMPLES.pop(manager.kg, None)
    else:
        _SAMPLES[manager.kg] = sample


def current_sample(manager: KgManager) -> KgSample | None:
    return _SAMPLES.get(manager.kg)


def values_block(entities: list[IRI]) -> str:
    return values_clause("entity", entities)


# --- stage a: prefixes from class samples ---------------------------------------------

def prefix_of(iri: str) -> str:
    """As PREFIX_EXPRESSION of kg_profile: host and first path segment if more path follows."""
    scheme, sep, rest = iri.partition("://")
    if not sep:
        cut = max(iri.rfind("#"), iri.rfind("/"))
        return iri[:cut + 1]
    host, slash, path = rest.partition("/")
    for end in "#?":
        if end in host:
            return f"{scheme}://{host.split(end)[0]}"
    segment, slash2, _ = path.partition("/")
    if slash and slash2 and segment and "#" not in segment and "?" not in segment:
        return f"{scheme}://{host}/{segment}/"
    return f"{scheme}://{host}"


# stage a: at most this many classes are sampled, the largest
MAX_SAMPLED_CLASSES = 300


@dataclass
class PrefixCounts:
    """Estimated typed entities per (prefix, class) and per prefix, from a
    small sample of each class. An entity of several classes counts once per
    class in counts, but once in totals."""
    counts: dict[tuple[str, str], int]
    named: dict[tuple[str, str], int]
    totals: Counter[str]
    named_totals: Counter[str]
    examples: dict[str, list[tuple[IRI, str]]]  # prefix -> sampled (entity, class)


def large_prefix_counts(manager: KgManager, classes: dict[IRI, int]) -> PrefixCounts:
    """Prefixes of the typed entities, estimated from a sample of each of the
    largest classes. Classes are named by their local name, like the kinds of a
    small KG."""
    counts: Counter[tuple[str, str]] = Counter()
    named: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    named_totals: Counter[str] = Counter()
    examples: dict[str, list[tuple[IRI, str]]] = {}
    names = sparql_path(LABEL_PREDICATES + SYNONYM_PREDICATES)
    sampled = dict(sorted(classes.items(), key=lambda x: -x[1])[:MAX_SAMPLED_CLASSES])
    class_values = "VALUES ?t { " + " ".join(f"<{t}>" for t in sampled) + " }"
    for type_iri, n in sampled.items():
        sample = draw_members(manager, type_membership(type_iri), n, CLASS_SAMPLE, 3)
        if not sample:
            continue
        with_name = {IRI(r["entity"].value) for r in select_rows(
            manager, f"SELECT DISTINCT ?entity WHERE {{ {values_block(sample)} ?entity {names} ?name }}") if "entity" in r}
        # number of sampled classes of each entity, so that it counts once in the totals
        n_classes = {IRI(r["entity"].value): int(r["k"].value) for r in select_rows(
            manager, f"SELECT ?entity (COUNT(DISTINCT ?t) AS ?k) WHERE {{ {values_block(sample)} {class_values} "
                     f"?entity <{RDF_TYPE}> ?t }} GROUP BY ?entity") if "entity" in r and "k" in r}
        label = local_name(type_iri) or type_iri
        weight = n / len(sample)
        for entity in sample:
            prefix = prefix_of(entity)
            share = weight / max(1, n_classes.get(entity, 1))
            counts[(prefix, label)] += weight
            totals[prefix] += share
            if entity in with_name:
                named[(prefix, label)] += weight
                named_totals[prefix] += share
            examples.setdefault(prefix, []).append((entity, label))
    rounded = lambda c: Counter({k: round(v) for k, v in c.items()})
    return PrefixCounts(dict(rounded(counts)), dict(rounded(named)), rounded(totals), rounded(named_totals), examples)


def partition_name(type_iri: str, taken: set[str]) -> str:
    """Lower case partition name from the local name of a class, unique among taken."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", local_name(type_iri)).lower()
    name = re.sub(r"[^a-z0-9_]+", "_", words).strip("_")
    name = name if name and name[0].isalpha() else f"c_{name}"
    unique, i = name, 2
    while unique in taken:
        unique, i = f"{name}_{i}", i + 1
    return unique


def default_class_partitions(manager: KgManager, space: tuple[str, ...], classes: dict[IRI, int]
                             ) -> list[tuple[str, IRI, int]]:
    """(name, class, estimated members in the entity space) of the default
    partitions of a large KG: one per class with at least MIN_CLASS_MEMBERS
    estimated members in the space, at most MAX_DEFAULT_PARTITIONS of the
    largest, ordered from small to large so that a subclass comes before its
    superclass (the first match wins)."""
    estimates = []
    for type_iri, n in sorted(classes.items(), key=lambda x: -x[1])[:MAX_SAMPLED_CLASSES]:
        if n < MIN_CLASS_MEMBERS:
            continue
        sample = draw_members(manager, type_membership(type_iri), n, CLASS_SAMPLE, 3)
        share = sum(not space or e.startswith(space) for e in sample) / len(sample) if sample else 0.0
        if n * share >= MIN_CLASS_MEMBERS:
            estimates.append((type_iri, round(n * share)))
    chosen = sorted(estimates, key=lambda x: -x[1])[:MAX_DEFAULT_PARTITIONS]
    taken: set[str] = set()
    partitions = []
    for type_iri, estimate in sorted(chosen, key=lambda x: x[1]):
        name = partition_name(type_iri, taken)
        taken.add(name)
        partitions.append((name, type_iri, estimate))
    return partitions
