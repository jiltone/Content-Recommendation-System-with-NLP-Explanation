"""
Shared utility functions used across the project.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


# ── Evaluation metrics ─────────────────────────────────────────────────────────

def precision_at_k(relevant_ids: list, recommended_ids: list, k: int = 5) -> float:
    """Fraction of top-k recommendations that are relevant."""
    top_k = recommended_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / k


def mean_cosine_similarity(query_vec: np.ndarray, candidate_vecs: np.ndarray) -> float:
    """Average cosine similarity between a query vector and a set of candidates."""
    sims = cosine_similarity(query_vec.reshape(1, -1), candidate_vecs)[0]
    return float(np.mean(sims))


def bleu_score(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU score between a reference and generated explanation."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoother = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)


# ── Display helpers ────────────────────────────────────────────────────────────

def format_recommendations(results: pd.DataFrame, explanations: list[str]) -> str:
    """Format top-k recommendations and their explanations as a markdown string."""
    lines = []
    for i, (_, row) in enumerate(results.iterrows()):
        explanation = explanations[i] if i < len(explanations) else ""
        lines.append(f"### {i + 1}. {row['title']}")
        lines.append(f"**Genres:** {row['genres']}")
        lines.append(f"**Match score:** {row['similarity_score']:.4f}")
        lines.append(f"**Why this movie:** {explanation}")
        lines.append("")
    return "\n".join(lines)


def ensure_dirs() -> None:
    """Create all required project directories if they do not exist."""
    for path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR]:
        os.makedirs(path, exist_ok=True)
