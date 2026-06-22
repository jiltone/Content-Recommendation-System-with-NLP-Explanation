"""
Phase 3 - NLP Preprocessing
Supports TMDB 5000 dataset (tmdb_5000_movies.csv + tmdb_5000_credits.csv).
Falls back to MovieLens (movies.csv + tags.csv) if TMDB files are not found.
"""

import re
import os
import ast
import pandas as pd
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def _parse_names(raw: str) -> str:
    """Extract 'name' fields from a Python-literal list-of-dicts string (TMDB format)."""
    try:
        items = ast.literal_eval(str(raw))
        return " ".join(item["name"] for item in items if "name" in item)
    except Exception:
        return ""


# ── TMDB loader ────────────────────────────────────────────────────────────────

def load_tmdb(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load and merge tmdb_5000_movies.csv + tmdb_5000_credits.csv."""
    movies_path = os.path.join(raw_dir, "tmdb_5000_movies.csv")
    credits_path = os.path.join(raw_dir, "tmdb_5000_credits.csv")

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # credits may have column 'movie_id' or 'id'
    id_col = "movie_id" if "movie_id" in credits.columns else "id"
    credits = credits.rename(columns={id_col: "id"})

    df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    # Parse Python-literal columns
    df["genres_text"] = df["genres"].apply(_parse_names)
    df["keywords_text"] = df["keywords"].apply(_parse_names)
    df["cast_text"] = df["cast"].apply(
        lambda x: " ".join(
            item["name"] for item in ast.literal_eval(str(x))[:5]
            if "name" in item
        ) if pd.notna(x) else ""
    )

    df["overview"] = df["overview"].fillna("")
    df["combined_text"] = (
        df["genres_text"] + " "
        + df["keywords_text"] + " "
        + df["overview"] + " "
        + df["cast_text"]
    )

    # Normalise column names to match the rest of the pipeline
    df = df.rename(columns={"id": "movieid"})
    return df[["movieid", "title", "genres_text", "combined_text"]].rename(
        columns={"genres_text": "genres"}
    )


# ── MovieLens loader (fallback) ────────────────────────────────────────────────

def load_movielens(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    movies = pd.read_csv(os.path.join(raw_dir, "movies.csv"))
    movies.columns = [c.strip().lower() for c in movies.columns]

    tags = pd.read_csv(os.path.join(raw_dir, "tags.csv"))
    tags.columns = [c.strip().lower() for c in tags.columns]

    tags_agg = (
        tags.groupby("movieid")["tag"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index()
        .rename(columns={"tag": "tags_text"})
    )
    df = movies.merge(tags_agg, on="movieid", how="left")
    df["tags_text"] = df["tags_text"].fillna("")
    df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
    df["combined_text"] = df["genres_clean"] + " " + df["tags_text"]
    return df[["movieid", "title", "genres", "combined_text"]]


# ── Pipeline ───────────────────────────────────────────────────────────────────

def preprocess_pipeline(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Detect dataset type, clean text, and save processed CSV."""
    os.makedirs(processed_dir, exist_ok=True)

    tmdb_movies = os.path.join(raw_dir, "tmdb_5000_movies.csv")
    if os.path.exists(tmdb_movies):
        print("Detected TMDB 5000 dataset.")
        df = load_tmdb(raw_dir)
    else:
        print("TMDB not found — falling back to MovieLens dataset.")
        df = load_movielens(raw_dir)

    print(f"Loaded {len(df)} movies. Cleaning text...")
    tqdm.pandas(desc="Cleaning")
    df["clean_text"] = df["combined_text"].progress_apply(clean_text)

    out_path = os.path.join(processed_dir, "movies_processed.csv")
    df[["movieid", "title", "genres", "clean_text"]].to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


if __name__ == "__main__":
    preprocess_pipeline()
