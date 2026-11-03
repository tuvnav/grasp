"""
The normalization ladder for comparing strings (raw, case, words, spaceless, wordset, plural, inflection), fine to coarse.
"""
import re
import unicodedata
from typing import Callable, Literal, get_args

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()


SEPARATOR_RE = re.compile(r"[\s_\-/,;:()\[\]{}.]+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
PARENS_PATTERN = re.compile(r"\s*\([^)]*\)")
YEAR_PATTERN = re.compile(r"\b(1\d{3}|20\d{2})\b")


# levels of the normalization ladder, fine to coarse; written out so that type
# checkers can read the Literal (NORMALIZATION_LEVELS must have exactly these keys)
type NormalizationLevel = Literal["raw", "case", "words", "spaceless", "wordset", "plural", "inflection"]
STANDARD_LEVELS: list[NormalizationLevel] = ["raw", "case", "words", "spaceless"]
EXTENDED_LEVELS: list[NormalizationLevel] = ["wordset", "plural", "inflection"]


NORMALIZATION_LEVELS: dict[NormalizationLevel, Callable[[str], str]] = {
    "raw": lambda w: w.strip(),
    "case": lambda w: " ".join(unicodedata.normalize("NFKC", w).casefold().split()),
    "words": lambda w: " ".join(words(w)),
    "spaceless": lambda w: "".join(words(w)),
    "wordset": lambda k: " ".join(sorted(set(words(k)))),
    "plural": lambda k: " ".join(sorted({plural(w) for w in words(k)})),
    "inflection": lambda k: " ".join(sorted({lemma(w) for w in words(k)})),
}
assert list(NORMALIZATION_LEVELS) == list(get_args(NormalizationLevel.__value__)), \
    "NormalizationLevel and NORMALIZATION_LEVELS differ"


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
    WordNet lemmatizer tends to corrupt such short strings.
    Doesn't normalize adjectives because of cases like
    "planting" -> "plant" and "decayed" -> "decay".
    """
    if len(token) <= 3:
        return token
    noun = LEMMATIZER.lemmatize(token, "n")
    if noun != token or wordnet.synsets(token, wordnet.NOUN) or wordnet.synsets(token, wordnet.ADJ):
        return noun
    return LEMMATIZER.lemmatize(token, "v")


def strip_accents(text: str) -> str:
    """Decomposes accented letters (NFKD) and drops the combining marks: "é" -> "e".
    Letters of other scripts (Cyrillic, Greek) stay as they are."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").strip()


def remove_parenthesized_parts(text: str) -> str:
    return PARENS_PATTERN.sub("", text).strip()


def digits_only(text: str) -> str:
    return "".join(c for c in text.strip() if c.isdigit())


def extract_year(text: str) -> str:
    match = YEAR_PATTERN.search(text)
    return match.group(1) if match else ""


def first_name_initials(name: str) -> str:
    tokens = name.strip().split()
    if len(tokens) <= 1:
        return name.strip()

    initials = [token[0].upper() for token in tokens[:-1]]
    last_name = tokens[-1]
    return f"{' '.join(initials)} {last_name}"


def regex_replace(text: str, pattern: str, replacement: str) -> str:
    return re.compile(pattern).sub(replacement, text).strip()


# string transforms a strategy applies before the normalization ladder; written out
# for type checkers. regex_replace takes a pattern and a replacement, the others only
# the text (TRANSFORM_OPERATIONS)
type TransformOperation = Literal["strip_accents", "remove_parenthesized_parts", "regex_replace",
                                  "digits_only", "year", "initials"]
REGEX_REPLACE: TransformOperation = "regex_replace"

TRANSFORM_OPERATIONS: dict[TransformOperation, Callable[[str], str]] = {
    "strip_accents": strip_accents,                            # "é" -> "e"
    "remove_parenthesized_parts": remove_parenthesized_parts,  # "Enterprise (NCC-1701)" -> "Enterprise"
    "digits_only": digits_only,                                # "NCC-1701-D" -> "1701"
    "year": extract_year,                                      # first four-digit year, or nothing
    "initials": first_name_initials,                           # "John Ronald Smith" -> "J R Smith"
}
assert set(TRANSFORM_OPERATIONS) | {REGEX_REPLACE} == set(get_args(TransformOperation.__value__)), \
    "TransformOperation and TRANSFORM_OPERATIONS differ"


def normalization_levels(extended: bool) -> list[NormalizationLevel]:
    return STANDARD_LEVELS + (EXTENDED_LEVELS if extended else [])
