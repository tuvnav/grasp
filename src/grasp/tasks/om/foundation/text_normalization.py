"""
The normalization ladder for comparing strings (raw, case, words, spaceless, wordset, plural, inflection), fine to coarse.
"""
import re
import unicodedata
from typing import Callable

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

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


def normalization_levels(extended: bool) -> list[str]:
    return STANDARD_LEVELS + (EXTENDED_LEVELS if extended else [])
