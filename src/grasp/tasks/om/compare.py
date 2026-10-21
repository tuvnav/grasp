from collections import Counter
import math
from typing import Callable, Iterable
import unicodedata

from grasp.tasks.om.model import (
    UNCATEGORIZED, GroupStats, KeyAnchor, KeyAnchorResult, KeyEntityRow, NormalizationLevelStats, PairingGroup, PartitionGroup)
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re

LEMMATIZER = WordNetLemmatizer()
SEPARATOR_RE = re.compile(r"[\s_\-/,;:()\[\]{}.]+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

STANDARD_LEVELS = ["raw", "case", "words", "spaceless"]
EXTENDED_LEVELS = ["wordset", "plural", "inflection"]

NORMALIZATION_LEVELS: dict[str, Callable] = {
    "raw": lambda w: w.strip(),
    "case": lambda w: " ".join(unicodedata.normalize("NFKC", w).casefold().split()),
    "words": lambda w: " ".join(words(w)),
    "spaceless": lambda w: "".join(words(w)),
    "wordset": lambda k: " ".join(sorted(set(words(k)))),
    "plural": lambda k: " ".join(sorted({plural(w) for w in words(k)})),
    "inflection": lambda k: " ".join(sorted({lemma(w) for w in words(k)})),
}

# normalized key -> rows of one side and partition group
type KeyComparisonIndex = dict[str, list[KeyEntityRow]]


def words(key: str) -> list[str]:
    """
    Normalization steps: NFKC, CamelCase boundaries, casefold,
    split at separators.
    Symbols like + # % stay inside words (C != C++).
    """
    key = CAMEL_RE.sub(" ", unicodedata.normalize("NFKC", key))
    return [w for w in SEPARATOR_RE.split(key.casefold()) if w]


def plural(token: str) -> str:
    return token if len(token) <= 3 else LEMMATIZER.lemmatize(token, "n")


def lemma(token: str) -> str:
    """
    Normalizes verb and noun inflection.
    Doesn't manipulate very short strings, because
    WordNet lemmatizer tends to corrupts such short strings.
    Doesn't normalize adjectives because of cases like 
    "planting" -> "plant" und "decayed" -> "decay".
    """
    if len(token) <= 3:
        return token
    noun = LEMMATIZER.lemmatize(token, "n")
    if noun != token or wordnet.synsets(token, wordnet.NOUN) or wordnet.synsets(token, wordnet.ADJ):
        return noun
    return LEMMATIZER.lemmatize(token, "v")


def build_key_comparison_index(
        key_entity_rows: Iterable[KeyEntityRow],
        string_normalization_function: Callable[[str], str],
        ) -> KeyComparisonIndex:
    """
    Returns an index of the form {normalized key -> [KeyEntityRow, ...]};
    empty keys are skipped.
    """
    index: KeyComparisonIndex = {}
    for row in key_entity_rows:
        key = string_normalization_function(row.key)
        if not key:
            continue
        index.setdefault(key, []).append(row)
    return index


def collisions(index: KeyComparisonIndex) -> set[str]:
    """
    Returns the entities sharing a key with another entity
    of the same side.
    """
    result = set()
    for rows in index.values():
        # filter out unique entities
        entities = {r.entity for r in rows}
        if len(entities) > 1:
            result.update(entities)
    return result


def cross_indices(
    src: KeyComparisonIndex,
    tgt: KeyComparisonIndex,
) -> tuple[dict[tuple[str, str], str], set[str], set[str]]:
    """Anchor rule on one level for one partition group. Returns
    ({(source, target): key}, ambiguous source entities, ambiguous target entities)."""
    candidates: dict[tuple[str, str], str] = {}
    for key in sorted(src.keys() & tgt.keys()):
        src_rows, tgt_rows = src[key], tgt[key]
        # (a) only primary keys create anchors
        if not any(r.primary for r in src_rows) or not any(r.primary for r in tgt_rows):
            continue
        # (b) unique over all keys (primary and secondary) on each side
        src_entities = {r.entity for r in src_rows}
        tgt_entities = {r.entity for r in tgt_rows}
        if len(src_entities) != 1 or len(tgt_entities) != 1:
            continue
        pair = (next(iter(src_entities)), next(iter(tgt_entities)))
        candidates.setdefault(pair, key)

    # (c) at most one partner per entity
    src_count = Counter(s for s, _ in candidates)
    tgt_count = Counter(t for _, t in candidates)
    anchors = {
        (s, t): key for (s, t), key in candidates.items()
        if src_count[s] == 1 and tgt_count[t] == 1
    }
    ambiguous_src = {s for s, n in src_count.items() if n > 1}
    ambiguous_tgt = {t for t, n in tgt_count.items() if n > 1}
    return anchors, ambiguous_src, ambiguous_tgt


def extension_counter(rows: Iterable[KeyEntityRow]) -> Callable[[frozenset[str]], int]:
    """Counts the key word sets of one side and partition that properly extend a
    given word set ({paper} ⊂ {poster, paper}); all keys, primary and secondary."""
    wordsets = {frozenset(words(r.key)) for r in rows} - {frozenset()}
    by_token: dict[str, set[frozenset[str]]] = {}
    for ws in wordsets:
        for token in ws:
            by_token.setdefault(token, set()).add(ws)

    def count(ws: frozenset[str]) -> int:
        if not ws or any(t not in by_token for t in ws):
            return 0
        containing = set.intersection(*(by_token[t] for t in ws))
        return sum(1 for other in containing if len(other) > len(ws))

    count.n_wordsets = len(wordsets)  # type: ignore[attr-defined]
    return count


def lexical_information_content(extensions: int, n_wordsets: int) -> float:
    """Seco et al. (2004): 1 - log(ext + 1) / log(N)."""
    if n_wordsets <= 1:
        return 1.0
    return 1 - math.log(extensions + 1) / math.log(n_wordsets)


def anchor_wordset(key_rows: list[KeyEntityRow]) -> frozenset[str]:
    """Word set of an anchor: the largest one among the names behind its key
    (on 'spaceless', the side that writes the words separately)."""
    return max((frozenset(words(r.key)) for r in key_rows), key=len)


def primary_key(key_rows: list[KeyEntityRow]) -> str:
    """Original form of a primary key behind a normalized key."""
    return next(r.key for r in key_rows if r.primary)


def token_overlap(src_rows: list[KeyEntityRow], tgt_rows: list[KeyEntityRow]) -> float:
    """Share of word occurrences in primary keys that also occur on the other side;
    minimum over both directions."""
    a = [w for r in src_rows if r.primary for w in words(r.key)]
    b = [w for r in tgt_rows if r.primary for w in words(r.key)]
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    return min(sum(w in set_b for w in a) / len(a), sum(w in set_a for w in b) / len(b))


def by_partition(rows: Iterable[KeyEntityRow]) -> dict[str, list[KeyEntityRow]]:
    result: dict[str, list[KeyEntityRow]] = {}
    for row in rows:
        result.setdefault(row.partition, []).append(row)
    return result


def resolve_pairing(
    src_partitions: Iterable[str],
    tgt_partitions: Iterable[str],
    pairing: list[PairingGroup] | None,
) -> list[PartitionGroup]:
    """Partition groups to compare. Without pairing, each partition is compared
    with the partition of the same name, except UNCATEGORIZED (entities the
    partition query does not cover, e.g. helper nodes); it can still be paired
    explicitly. Each partition may occur in at most one group per side,
    otherwise an entity could get anchors in several groups."""
    if pairing is None:
        common = sorted((set(src_partitions) & set(tgt_partitions)) - {UNCATEGORIZED})
        return [PartitionGroup(frozenset([p]), frozenset([p])) for p in common]

    groups = [PartitionGroup(frozenset(g.source), frozenset(g.target)) for g in pairing]
    for side in ("source", "target"):
        seen: set[str] = set()
        for group in groups:
            partitions = getattr(group, side)
            if not partitions:
                raise ValueError(f"Group {group.label} has no {side} partition")
            if seen & partitions:
                raise ValueError(
                    f"{side.capitalize()} partitions {sorted(seen & partitions)} occur in several groups"
                )
            seen |= partitions
    return groups


def normalization_levels(extended: bool) -> list[str]:
    return STANDARD_LEVELS + (EXTENDED_LEVELS if extended else [])


def is_extended(anchor: KeyAnchor) -> bool:
    return anchor.string_normalization_level in EXTENDED_LEVELS


def stratum(anchor: KeyAnchor) -> tuple[str, str]:
    """Certification stratum: (group label, standard_specific | standard_head | extended)."""
    if is_extended(anchor):
        return anchor.group.label, "extended"
    return anchor.group.label, "standard_head" if anchor.head else "standard_specific"


def anchors_for_group(
    group: PartitionGroup,
    src: list[KeyEntityRow],
    tgt: list[KeyEntityRow],
    extended: bool,
) -> tuple[list[KeyAnchor], GroupStats]:
    """Runs the anchor rule over the normalization levels, from fine to coarse.
    Each anchor gets the finest level on which it is unique; coarser levels only
    add entities without an anchor."""
    stats = GroupStats(
        n_source=len({r.entity for r in src}),
        n_target=len({r.entity for r in tgt}),
        token_overlap=token_overlap(src, tgt),
    )
    src_ext, tgt_ext = extension_counter(src), extension_counter(tgt)

    anchors: list[KeyAnchor] = []
    anchored_src: set[str] = set()
    anchored_tgt: set[str] = set()
    collided_src: set[str] = set()
    collided_tgt: set[str] = set()

    for level in normalization_levels(extended):
        normalize = NORMALIZATION_LEVELS[level]
        src_index = build_key_comparison_index(src, normalize)
        tgt_index = build_key_comparison_index(tgt, normalize)
        level_stats = NormalizationLevelStats()

        # collisions that are new compared to all finer levels
        new_src = collisions(src_index) - collided_src
        new_tgt = collisions(tgt_index) - collided_tgt
        level_stats.new_collisions_source = len(new_src)
        level_stats.new_collisions_target = len(new_tgt)
        collided_src |= new_src
        collided_tgt |= new_tgt

        pairs, amb_src, amb_tgt = cross_indices(src_index, tgt_index)
        level_stats.ambiguous_source = len(amb_src)
        level_stats.ambiguous_target = len(amb_tgt)

        for (s, t), key in sorted(pairs.items()):
            if s in anchored_src or t in anchored_tgt:
                continue  # already anchored on a finer level
            ws = anchor_wordset(src_index[key] + tgt_index[key])
            e_src, e_tgt = src_ext(ws), tgt_ext(ws)
            anchors.append(KeyAnchor(
                source_iri=s,
                target_iri=t,
                group=group,
                string_normalization_level=level,
                evidence=key,
                source_key=primary_key(src_index[key]),
                target_key=primary_key(tgt_index[key]),
                head=e_src > 0 or e_tgt > 0,
                information_content=min(
                    lexical_information_content(e_src, src_ext.n_wordsets),
                    lexical_information_content(e_tgt, tgt_ext.n_wordsets)
                ),
            ))
            anchored_src.add(s)
            anchored_tgt.add(t)
            level_stats.new_anchors += 1

        stats.levels[level] = level_stats

    stats.anchors = len(anchors)
    stats.extended = sum(is_extended(a) for a in anchors)
    stats.head = sum(a.head and not is_extended(a) for a in anchors)
    stats.specific = sum(not a.head and not is_extended(a) for a in anchors)
    stats.anchor_rate = stats.anchors / max(min(stats.n_source, stats.n_target), 1)
    return anchors, stats


# main function
def key_anchors(
    src_rows: Iterable[KeyEntityRow],
    tgt_rows: Iterable[KeyEntityRow],
    extended: bool = False,
    pairing: list[PairingGroup] | None = None,
) -> KeyAnchorResult:
    src_parts = by_partition(src_rows)
    tgt_parts = by_partition(tgt_rows)
    groups = resolve_pairing(src_parts, tgt_parts, pairing)

    anchors: list[KeyAnchor] = []
    stats: dict[PartitionGroup, GroupStats] = {}
    compared_src: set[str] = set()
    compared_tgt: set[str] = set()
    for group in groups:
        # union of the group's partitions; an entity in several of them (punning)
        # has duplicate rows, which the anchor rule counts as one entity
        src = [r for p in sorted(group.source) for r in src_parts.get(p, [])]
        tgt = [r for p in sorted(group.target) for r in tgt_parts.get(p, [])]
        if not src or not tgt:
            continue
        group_anchors, stats[group] = anchors_for_group(group, src, tgt, extended)
        anchors.extend(group_anchors)
        compared_src.update(r.entity for r in src)
        compared_tgt.update(r.entity for r in tgt)

    grouped_src = set().union(*(g.source for g in groups))
    grouped_tgt = set().union(*(g.target for g in groups))
    return KeyAnchorResult(
        anchors=anchors,
        group_statistics=stats,
        unpaired_source=sorted(src_parts.keys() - grouped_src),
        unpaired_target=sorted(tgt_parts.keys() - grouped_tgt),
        missing_source=sorted(grouped_src - src_parts.keys()),
        missing_target=sorted(grouped_tgt - tgt_parts.keys()),
        source_entities=frozenset(compared_src),
        target_entities=frozenset(compared_tgt),
    )
