"""
Phase 6 - LLM Explanation Generator
Uses google/flan-t5-base with chain-of-thought prompting to explain why a movie
matches the user's mood.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = "google/flan-t5-base"

_tokenizer: T5Tokenizer | None = None
_llm_model: T5ForConditionalGeneration | None = None


def _load_model() -> tuple[T5Tokenizer, T5ForConditionalGeneration]:
    global _tokenizer, _llm_model
    if _tokenizer is None or _llm_model is None:
        print(f"Loading LLM: {MODEL_NAME}")
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        _llm_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        _llm_model.eval()
    return _tokenizer, _llm_model


def _build_prompt(user_mood: str, movie_title: str, movie_genres: str) -> str:
    return (
        f'A user is looking for: "{user_mood}"\n\n'
        f"Recommended movie: {movie_title}\n"
        f"Genres: {movie_genres}\n\n"
        f"Step 1: Think about what the user wants.\n"
        f"Step 2: Think about what this movie offers.\n"
        f"Step 3: Explain in 2 sentences why this movie matches what the user wants.\n\n"
        f"Explanation:"
    )


def generate_explanation(
    user_mood: str,
    movie_title: str,
    movie_genres: str,
    max_new_tokens: int = 120,
) -> str:
    """Generate a natural-language explanation for a movie recommendation."""
    tokenizer, model = _load_model()
    prompt = _build_prompt(user_mood, movie_title, movie_genres)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    explanation = generate_explanation(
        user_mood="a thrilling sci-fi adventure",
        movie_title="Interstellar",
        movie_genres="Adventure|Drama|Sci-Fi",
    )
    print(explanation)
