"""TF-IDF top token computation for SAE features.

Computes discriminative tokens for each feature without neural network inference.
Uses simple regex tokenization and TF-IDF scoring against a background corpus.
"""

import re
import math
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd


# Common English stopwords (hardcoded to avoid nltk dependency)
STOPWORDS = frozenset([
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "has", "his", "how", "its",
    "may", "new", "now", "old", "see", "way", "who", "did", "get", "got",
    "him", "let", "say", "she", "too", "use", "been", "call", "come",
    "each", "from", "have", "here", "high", "just", "know", "like", "long",
    "look", "make", "many", "more", "most", "much", "must", "name", "only",
    "over", "part", "some", "such", "take", "than", "that", "them", "then",
    "they", "this", "time", "very", "want", "well", "were", "what", "when",
    "will", "with", "word", "work", "year", "also", "back", "been", "both",
    "came", "does", "done", "down", "even", "find", "first", "give", "goes",
    "good", "great", "hand", "help", "home", "into", "keep", "last", "left",
    "life", "line", "live", "made", "main", "need", "next", "once", "open",
    "own", "play", "put", "read", "real", "right", "same", "said", "show",
    "side", "small", "still", "tell", "thing", "think", "those", "three",
    "through", "under", "using", "where", "which", "while", "world", "would",
    "about", "above", "after", "again", "being", "below", "between", "could",
    "during", "every", "found", "going", "having", "never", "other", "place",
    "point", "should", "since", "start", "state", "their", "there", "these",
    "until", "water", "people", "number", "before", "second", "little",
    "because", "against", "another", "around", "without", "within", "however",
    "already", "always", "became", "become", "called", "different", "include",
    "included", "including", "often", "several", "though", "together",
    "toward", "towards", "whether", "might", "something", "nothing",
    "everything", "anything", "someone", "everyone", "really", "actually",
    "used", "able", "rather", "given", "quite", "likely", "along",
])


def tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase, extract words of 3+ chars."""
    return re.findall(r"[a-z]{3,}", text.lower())


def compute_feature_top_tokens(
    feature_texts: list[str],
    corpus_texts: list[str],
    max_tokens: int = 15,
    min_doc_freq: int = 2,
) -> list[str]:
    """Return top discriminative tokens for a feature using TF-IDF.

    1. Tokenize all texts (lowercase, extract 3+ char words)
    2. Remove stopwords
    3. Compute TF-IDF: tf(feature_texts) * log(N / df(corpus))
    4. Return top tokens sorted by score
    """
    # Tokenize feature texts into per-document token sets
    feature_doc_tokens = []
    feature_token_counts = Counter()
    for text in feature_texts:
        tokens = [t for t in tokenize(text) if t not in STOPWORDS]
        feature_doc_tokens.append(set(tokens))
        feature_token_counts.update(tokens)

    # Count document frequency in feature texts
    feature_df = Counter()
    for token_set in feature_doc_tokens:
        for token in token_set:
            feature_df[token] += 1

    # Filter tokens by minimum document frequency in feature texts
    candidate_tokens = {
        t for t, df in feature_df.items() if df >= min_doc_freq
    }

    if not candidate_tokens:
        # Fallback: use all tokens that appear at least once
        candidate_tokens = set(feature_token_counts.keys())

    # Compute document frequency across corpus
    corpus_df = Counter()
    for text in corpus_texts:
        tokens = set(tokenize(text)) - STOPWORDS
        for token in tokens:
            corpus_df[token] += 1

    n_corpus = max(len(corpus_texts), 1)
    n_feature = max(len(feature_texts), 1)

    # Compute TF-IDF scores
    scores = {}
    for token in candidate_tokens:
        # TF: frequency in feature texts normalized by total tokens
        tf = feature_token_counts[token] / max(sum(feature_token_counts.values()), 1)

        # IDF: log(N_corpus / (1 + df_corpus)) — tokens rare in corpus score higher
        df_corpus = corpus_df.get(token, 0)
        idf = math.log((n_corpus + 1) / (1 + df_corpus))

        # Boost for tokens appearing in many feature docs
        doc_coverage = feature_df.get(token, 0) / n_feature

        scores[token] = tf * idf * (1 + doc_coverage)

    # Sort by score descending
    sorted_tokens = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
    return sorted_tokens[:max_tokens]


def compute_all_top_tokens(
    samples_df: pd.DataFrame,
    n_background: int = 500,
    max_tokens: int = 15,
    min_doc_freq: int = 2,
) -> dict[int, list[str]]:
    """Compute top tokens for all features at once.

    Args:
        samples_df: DataFrame with columns 'feature' and 'chunk_text'
        n_background: number of random background texts to sample
        max_tokens: max discriminative tokens per feature
        min_doc_freq: minimum document frequency in feature texts

    Returns:
        dict mapping feature_id -> list of top discriminative tokens
    """
    features = sorted(samples_df["feature"].unique())

    # Build background corpus: random sample from all texts
    all_texts = samples_df["chunk_text"].dropna().tolist()
    rng = np.random.RandomState(42)
    n_bg = min(n_background, len(all_texts))
    bg_indices = rng.choice(len(all_texts), size=n_bg, replace=False)
    corpus_texts = [all_texts[i] for i in bg_indices]

    result = {}
    for feat_id in features:
        feat_texts = (
            samples_df[samples_df["feature"] == feat_id]["chunk_text"]
            .dropna()
            .tolist()
        )
        if not feat_texts:
            result[feat_id] = []
            continue

        result[feat_id] = compute_feature_top_tokens(
            feature_texts=feat_texts,
            corpus_texts=corpus_texts,
            max_tokens=max_tokens,
            min_doc_freq=min_doc_freq,
        )

    return result
