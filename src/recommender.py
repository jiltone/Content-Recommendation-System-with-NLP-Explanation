"""
Phase 5 - Recommendation Engine
Uses Sentence Transformers to encode user mood and find the most similar movies.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_NAME = "all-MiniLM-L6-v2"

_sentence_model: SentenceTransformer | None = None
_movie_embeddings: np.ndarray | None = None
_df: pd.DataFrame | None = None


def _get_model() -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is None:
        print(f"Loading sentence transformer: {MODEL_NAME}")
        _sentence_model = SentenceTransformer(MODEL_NAME)
    return _sentence_model


def load_data(
    processed_dir: str = PROCESSED_DIR,
    models_dir: str = MODELS_DIR,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load processed movie data and pre-computed embeddings (or compute them)."""
    global _movie_embeddings, _df

    df_path = os.path.join(processed_dir, "movies_processed.csv")
    emb_path = os.path.join(models_dir, "st_movie_embeddings.npy")

    _df = pd.read_csv(df_path)

    if os.path.exists(emb_path):
        print("Loading cached sentence-transformer embeddings...")
        _movie_embeddings = np.load(emb_path)
    else:
        print("Computing sentence-transformer embeddings (first run — this takes a while)...")
        model = _get_model()
        texts = _df["clean_text"].fillna("").tolist()
        _movie_embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        os.makedirs(models_dir, exist_ok=True)
        np.save(emb_path, _movie_embeddings)
        print(f"Embeddings saved to {emb_path}")

    return _df, _movie_embeddings


def recommend(
    user_input: str,
    top_k: int = 5,
    df: pd.DataFrame | None = None,
    movie_embeddings: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return top-k movie recommendations for a user mood/description."""
    global _df, _movie_embeddings

    if df is None:
        df = _df
    if movie_embeddings is None:
        movie_embeddings = _movie_embeddings

    if df is None or movie_embeddings is None:
        df, movie_embeddings = load_data()

    model = _get_model()
    user_vec = model.encode([user_input])
    scores = cosine_similarity(user_vec, movie_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[top_indices][["movieid", "title", "genres"]].copy()
    results["similarity_score"] = scores[top_indices]
    return results, scores[top_indices]


if __name__ == "__main__":
    load_data()
    results, scores = recommend("I want a fun adventure movie for the whole family")
    print(results.to_string(index=False))
