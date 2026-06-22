"""
Phase 4 - Word Embeddings
Trains a Word2Vec model on movie text and provides per-movie vector representations.
"""

import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1
WORKERS = 4
EPOCHS = 10


def load_processed(processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    path = os.path.join(processed_dir, "movies_processed.csv")
    return pd.read_csv(path)


def train_word2vec(texts: list[str], save_path: str | None = None) -> Word2Vec:
    """Train Word2Vec on a list of cleaned text strings."""
    sentences = [t.split() for t in texts]
    print("Training Word2Vec...")
    model = Word2Vec(
        sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Word2Vec model saved to {save_path}")
    return model


def get_movie_vector(text: str, model: Word2Vec) -> np.ndarray:
    """Average Word2Vec vectors for all words in text."""
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(VECTOR_SIZE)


def build_movie_matrix(df: pd.DataFrame, model: Word2Vec) -> np.ndarray:
    """Build (n_movies, vector_size) embedding matrix for all movies."""
    print("Building movie embedding matrix...")
    matrix = np.vstack([
        get_movie_vector(text, model)
        for text in tqdm(df["clean_text"].fillna(""), desc="Vectorizing")
    ])
    return matrix


def run_embeddings_pipeline(
    processed_dir: str = PROCESSED_DIR,
    models_dir: str = MODELS_DIR,
) -> tuple[pd.DataFrame, Word2Vec, np.ndarray]:
    os.makedirs(models_dir, exist_ok=True)
    df = load_processed(processed_dir)

    w2v_path = os.path.join(models_dir, "word2vec.model")
    model = train_word2vec(df["clean_text"].fillna("").tolist(), save_path=w2v_path)
    matrix = build_movie_matrix(df, model)

    np.save(os.path.join(models_dir, "w2v_movie_embeddings.npy"), matrix)
    print("Embeddings saved.")
    return df, model, matrix


if __name__ == "__main__":
    run_embeddings_pipeline()
