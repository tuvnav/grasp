"""
Texts as vectors, and the cosine similarity of two lists of texts. No KG here:
the vector strategy (kg_operations) gives the texts of its two value indices
and gets back a matrix whose row i, column j is the similarity of source text i
and target text j (the positions of the texts in the two lists).

Every vectorizer gives rows of length 1 (L2-normalized), so the dot product of
two rows is their cosine, and the similarity matrix is one matrix product.
"""
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

type Vectorizer = Literal["idf_tokens", "char_3grams", "embedding"]
# a matrix with one row per text: sparse (TF-IDF) or dense (embedding). The
# type checker cannot tell the two apart on its own (scipy and scikit-learn have
# no type hints), so the functions below separate them with isinstance and cast
type TextVectors = csr_matrix | np.ndarray

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# what a model expects in front of every text (e5 models are trained with it)
EMBEDDING_TEXT_PREFIXES: dict[str, str] = {
    "intfloat/multilingual-e5-small": "query: ",
    "intfloat/multilingual-e5-base": "query: ",
}

TEXT_BATCH_SIZE = 1000

def vectorize_texts_tf_idf(
        source_texts: list[str],
        target_texts: list[str],
        vectorizer: Literal["idf_tokens", "char_3grams"],
        ) -> tuple[csr_matrix, csr_matrix]:
    """
    TF-IDF vectors of the texts of both sides, on words (idf_tokens) or on
    character 3-grams within words (char_3grams). Vocabulary and IDF are learned
    from the texts of both sides together, so that a column means the same word
    (or 3-gram) on both sides.
    """
    if vectorizer == "idf_tokens":
        # the "words" level has split the words already; \S+ also keeps one-letter words
        tf_idf = TfidfVectorizer(analyzer="word", token_pattern=r"\S+")
    else:
        tf_idf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))
    tf_idf.fit(source_texts + target_texts)
    # scikit-learn has no type hints: transform returns a csr_matrix
    return cast(csr_matrix, tf_idf.transform(source_texts)), cast(csr_matrix, tf_idf.transform(target_texts))


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str):
    """The sentence-transformers model, loaded once per process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "The embedding vectorizer needs sentence-transformers: pip install grasp[embeddings]"
        ) from error
    return SentenceTransformer(model_name, device="cpu")


# Embeddings are cached per model (text -> vector): the trial runs of a strategy
# embed the same texts again and again. In memory for the process; also on disk
# (one file per model) if this environment variable names a directory.
EMBEDDING_CACHE_DIRECTORY_VARIABLE = "GRASP_EMBEDDING_CACHE_DIR"
_embedding_cache_by_model: dict[str, dict[str, np.ndarray]] = {}


def embedding_cache_path(model_name: str) -> Path | None:
    """The cache file of a model, None without a cache directory."""
    directory = os.environ.get(EMBEDDING_CACHE_DIRECTORY_VARIABLE)
    if not directory:
        return None
    return Path(directory) / f"{model_name.replace('/', '__')}.pkl"


def load_embedding_cache(model_name: str) -> dict[str, np.ndarray]:
    """The cache of a model: from memory, else from disk, else empty."""
    if model_name not in _embedding_cache_by_model:
        path = embedding_cache_path(model_name)
        _embedding_cache_by_model[model_name] = (
            pickle.loads(path.read_bytes()) if path is not None and path.exists() else {})
    return _embedding_cache_by_model[model_name]


def save_embedding_cache(model_name: str, cache: dict[str, np.ndarray]) -> None:
    """Writes the cache of a model to disk (if a cache directory is set): to a
    temporary file first, so that an interrupted write leaves the old file intact."""
    path = embedding_cache_path(model_name)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_bytes(pickle.dumps(cache))
    temporary_path.replace(path)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Embedding vectors of the texts, one row each, L2-normalized. Only texts
    not in the cache of the model are embedded."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    cache = load_embedding_cache(model_name)
    missing_texts = [text for text in dict.fromkeys(texts) if text not in cache]  # once each, in order
    if missing_texts:
        prefix = EMBEDDING_TEXT_PREFIXES.get(model_name, "")
        model = load_embedding_model(model_name)
        vectors = model.encode(
            [prefix + text for text in missing_texts],
            normalize_embeddings=True,
            batch_size=256,
            convert_to_numpy=True,
        )
        cache.update(zip(missing_texts, vectors))
        save_embedding_cache(model_name, cache)
    return np.array([cache[text] for text in texts], dtype=np.float32)


def vectorize_texts(
        source_texts: list[str],
        target_texts: list[str],
        vectorizer: Vectorizer,
        model_name: str | None = None,
        ) -> tuple[TextVectors, TextVectors]:
    """
    The vectors of the texts of both sides: row i of the first matrix belongs to
    source_texts[i], row j of the second to target_texts[j]. model_name: only for
    the embedding vectorizer (None: DEFAULT_EMBEDDING_MODEL).
    """
    if vectorizer == "embedding":
        model_name = model_name or DEFAULT_EMBEDDING_MODEL
        return embed_texts(source_texts, model_name), embed_texts(target_texts, model_name)
    return vectorize_texts_tf_idf(source_texts, target_texts, vectorizer)


def count_text_vectors(vectors: TextVectors) -> int:
    """The number of rows (texts) of a vector matrix, sparse or dense."""
    if isinstance(vectors, np.ndarray):
        return vectors.shape[0]
    return vectors.get_shape()[0]


def build_similarity_matrix(
        source_vectors: TextVectors,
        target_vectors: TextVectors,
        ) -> np.ndarray:
    """
    The cosine of every source row with every target row: a dense n x m matrix
    (n source rows, m target rows); entry [i, j] belongs to source row i and
    target row j. Needs n * m * 4 bytes (float32): for large inputs, call it on
    blocks of source rows (source_vectors[start:end]).
    """
    if isinstance(source_vectors, np.ndarray) and isinstance(target_vectors, np.ndarray):
        similarities = source_vectors @ target_vectors.T                          # embedding: dense
    else:
        similarities = csr_matrix(source_vectors @ target_vectors.T).toarray()   # TF-IDF: sparse
    return np.asarray(similarities, dtype=np.float32)
