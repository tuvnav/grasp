"""
VALUES blocks and tables; large ones as placeholders that are filled in after the prefixes are fixed.
"""
import hashlib
import re
from typing import Iterable

from grasp.tasks.om.foundation.types import IRI

# IRIs per VALUES block in batch queries
BATCH_SIZE = 200


# VALUES blocks with more IRIs are put into a query after its prefixes are fixed
MAX_INLINE_VALUES = 100


_PLACEHOLDER = "urn:grasp-values:"


_VALUES_BLOCKS: dict[str, str] = {}


def values_clause(var: str, iris: Iterable[str]) -> str:
    """VALUES ?var { <iri> ... }; a large one as a placeholder IRI that
    prepare_query replaces by the IRIs."""
    terms = " ".join(f"<{iri}>" for iri in iris)
    if terms.count("<") <= MAX_INLINE_VALUES:
        return f"VALUES ?{var} {{ {terms} }}"
    return f"VALUES ?{var} {{ {placeholder(terms)} }}"


def values_table(variables: list[str], rows: Iterable[tuple[str, ...]]) -> str:
    """VALUES (?a ?b) { (term term) ... } of SPARQL terms; a large one as a
    placeholder row that prepare_query replaces by the rows."""
    head = "(" + " ".join(f"?{v}" for v in variables) + ")"
    rows = list(rows)
    terms = " ".join("(" + " ".join(row) + ")" for row in rows)
    if len(rows) <= MAX_INLINE_VALUES:
        return f"VALUES {head} {{ {terms} }}"
    undef = " UNDEF" * (len(variables) - 1)
    return f"VALUES {head} {{ ({placeholder(terms)}{undef}) }}"


def placeholder(terms: str) -> str:
    key = f"{_PLACEHOLDER}{hashlib.sha1(terms.encode()).hexdigest()}"
    _VALUES_BLOCKS[key] = terms
    return f"<{key}>"


def expand_values_blocks(sparql: str) -> str:
    if _PLACEHOLDER not in sparql:
        return sparql
    # a placeholder row of a table (<key> UNDEF ...), or a placeholder IRI
    return re.sub(rf"\(\s*<({re.escape(_PLACEHOLDER)}[0-9a-f]+)>(?:\s+UNDEF)+\s*\)|<({re.escape(_PLACEHOLDER)}[0-9a-f]+)>",
                  lambda m: _VALUES_BLOCKS[m.group(1) or m.group(2)], sparql)


def values_blocks(iris: Iterable[IRI]) -> Iterable[str]:
    """The IRIs as VALUES contents (<a> <b> ...) in blocks of BATCH_SIZE."""
    iris = sorted(set(iris))
    for start in range(0, len(iris), BATCH_SIZE):
        yield " ".join(f"<{iri}>" for iri in iris[start:start + BATCH_SIZE])
