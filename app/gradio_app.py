"""
Phase 7 - Gradio Web App (Enhanced UI)
Run:  python app/gradio_app.py
Then open: http://localhost:7860
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from src.recommender import load_data, recommend
from src.explainer import generate_explanation

print("Initialising recommender system...")
df, movie_embeddings = load_data()
print("Ready.")

MEDAL = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

GENRE_COLORS = {
    "Action": "#e74c3c", "Adventure": "#e67e22", "Animation": "#f1c40f",
    "Comedy": "#2ecc71", "Crime": "#8e44ad", "Documentary": "#16a085",
    "Drama": "#2980b9", "Fantasy": "#9b59b6", "Horror": "#c0392b",
    "Mystery": "#1abc9c", "Romance": "#e91e63", "Science Fiction": "#00bcd4",
    "Sci-Fi": "#00bcd4", "Thriller": "#ff5722", "War": "#795548",
    "Western": "#a0522d", "Family": "#4caf50", "History": "#607d8b",
    "Music": "#ff4081", "Biography": "#795548",
}


def genre_badges(genres_str: str) -> str:
    genres = [g.strip() for g in genres_str.replace("|", ",").split(",") if g.strip()]
    badges = []
    for g in genres[:5]:
        color = GENRE_COLORS.get(g, "#607d8b")
        badges.append(
            f'<span style="background:{color};color:#fff;padding:3px 10px;'
            f'border-radius:12px;font-size:12px;font-weight:600;margin:2px;'
            f'display:inline-block;">{g}</span>'
        )
    return "".join(badges)


def score_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#2ecc71" if pct >= 40 else "#f39c12" if pct >= 25 else "#e74c3c"
    return (
        f'<div style="margin:6px 0 10px 0;">'
        f'<span style="font-size:12px;color:#aaa;">Match score: <b style="color:{color}">{pct}%</b></span>'
        f'<div style="background:#2a2a3a;border-radius:6px;height:6px;margin-top:4px;">'
        f'<div style="background:{color};width:{min(pct*2,100)}%;height:6px;border-radius:6px;"></div>'
        f'</div></div>'
    )


def build_card(rank: int, title: str, genres: str, score: float, explanation: str) -> str:
    medal = MEDAL[rank] if rank < len(MEDAL) else f"{rank+1}."
    return f"""
<div style="background:linear-gradient(135deg,#1e1e2e,#252540);border:1px solid #3a3a5c;
border-radius:16px;padding:20px 24px;margin:12px 0;box-shadow:0 4px 20px rgba(0,0,0,0.4);">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
    <span style="font-size:28px;">{medal}</span>
    <h3 style="margin:0;color:#e8e8ff;font-size:18px;font-weight:700;">{title}</h3>
  </div>
  <div style="margin-bottom:6px;">{genre_badges(genres)}</div>
  {score_bar(score)}
  <div style="background:rgba(255,255,255,0.04);border-left:3px solid #7c6af7;
  padding:10px 14px;border-radius:0 8px 8px 0;margin-top:4px;">
    <span style="font-size:13px;color:#b0b0d0;font-style:italic;">💡 {explanation}</span>
  </div>
</div>"""


def get_recommendations(user_input: str, top_k: int = 5) -> str:
    if not user_input.strip():
        return "<p style='color:#e74c3c;text-align:center;'>Please describe what you feel like watching.</p>"

    results, scores = recommend(user_input, top_k=int(top_k), df=df, movie_embeddings=movie_embeddings)

    cards = []
    for i, (_, row) in enumerate(results.iterrows()):
        explanation = generate_explanation(user_input, row["title"], row["genres"])
        cards.append(build_card(i, row["title"], row["genres"], float(scores[i]), explanation))

    header = (
        f'<div style="text-align:center;margin-bottom:16px;">'
        f'<span style="color:#7c6af7;font-size:14px;font-weight:600;">Found {len(cards)} recommendations for:</span><br>'
        f'<span style="color:#e8e8ff;font-size:16px;font-style:italic;">"{user_input}"</span>'
        f'</div>'
    )
    return header + "".join(cards)


CSS = """
/* Hide Gradio footer/watermark */
footer { display: none !important; }
.built-with { display: none !important; }
#component-0 { min-height: 100vh; }

/* Page background */
body, .gradio-container {
    background: #0f0f1a !important;
    font-family: 'Segoe UI', sans-serif !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 32px 0 16px 0;
}

/* Input box */
textarea, input[type=text] {
    background: #1a1a2e !important;
    color: #e8e8ff !important;
    border: 1px solid #3a3a5c !important;
    border-radius: 12px !important;
    font-size: 15px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.25) !important;
}

/* Labels */
label span, .label-wrap span {
    color: #9898c0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg, #7c6af7, #a855f7) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 12px 0 !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.88 !important; }

/* Slider */
input[type=range] { accent-color: #7c6af7 !important; }

/* Examples */
.examples-header { color: #9898c0 !important; font-size: 12px !important; }
.example-btn {
    background: #1e1e2e !important;
    border: 1px solid #3a3a5c !important;
    border-radius: 8px !important;
    color: #b0b0d0 !important;
    font-size: 13px !important;
}
.example-btn:hover { border-color: #7c6af7 !important; color: #e8e8ff !important; }

/* Output panel */
.output-html { background: transparent !important; border: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f1a; }
::-webkit-scrollbar-thumb { background: #3a3a5c; border-radius: 3px; }
"""

EXAMPLES = [
    ["a funny animated movie for the whole family"],
    ["dark psychological thriller with a twist ending"],
    ["romantic comedy set in New York"],
    ["epic space adventure with stunning visuals"],
    ["inspiring true story about overcoming adversity"],
    ["action-packed superhero movie"],
    ["slow-burn mystery with great acting"],
]

with gr.Blocks(title="CineAI — Movie Recommender", css=CSS, theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div class="app-header">
      <div style="font-size:48px;">🎬</div>
      <h1 style="color:#e8e8ff;font-size:32px;font-weight:800;margin:8px 0 4px 0;letter-spacing:-0.5px;">
        CineAI
      </h1>
      <p style="color:#7c6af7;font-size:15px;margin:0;font-weight:500;">
        AI-Powered Movie Recommendations with NLP Explanations
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=300):
            gr.HTML('<div style="height:8px;"></div>')
            user_input = gr.Textbox(
                label="Describe your mood or what you want to watch",
                placeholder="e.g. a dark psychological thriller with unexpected twists...",
                lines=4,
                show_label=True,
            )
            top_k = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Number of recommendations",
            )
            submit_btn = gr.Button("Find Movies ✨", variant="primary", size="lg")

            gr.HTML('<div style="margin-top:20px;color:#9898c0;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Try an example</div>')
            gr.Examples(
                examples=EXAMPLES,
                inputs=user_input,
                label="",
            )

        with gr.Column(scale=3, min_width=400):
            output = gr.HTML(
                value='<div style="text-align:center;padding:60px 20px;color:#3a3a5c;">'
                      '<div style="font-size:48px;">🍿</div>'
                      '<p style="margin-top:12px;font-size:15px;">Your recommendations will appear here</p>'
                      '</div>'
            )

    submit_btn.click(fn=get_recommendations, inputs=[user_input, top_k], outputs=output)
    user_input.submit(fn=get_recommendations, inputs=[user_input, top_k], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
