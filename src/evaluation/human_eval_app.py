"""
Human evaluation rating tool (README §7.3): presents responses from all
compared systems in randomized order with system identity hidden, including
unlabeled human reference responses as an upper reference point. Rates on a
5-point scale across 4 dimensions (relevance, language understanding,
contextual coherence, naturalness/appropriateness) and writes ratings to
logs/human_eval/.

Reads the most recent output of `python -m src.evaluation.compare`.

Run: streamlit run src/evaluation/human_eval_app.py
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = ROOT / "logs" / "comparisons"
HUMAN_EVAL_DIR = ROOT / "logs" / "human_eval"

DIMENSIONS = [
    ("relevance", "Relevance -- does the response address what the user asked?"),
    ("language_understanding", "Language understanding -- was the mixed-language input correctly interpreted?"),
    ("contextual_coherence", "Contextual coherence -- does it follow from prior turns?"),
    ("naturalness", "Naturalness -- would a Nigerian speaker plausibly write this?"),
]


def _latest_comparison_file() -> Path | None:
    if not COMPARISON_DIR.exists():
        return None
    files = sorted(COMPARISON_DIR.glob("*.json"))
    return files[-1] if files else None


def _build_rating_items(comparison: dict, seed: int) -> list[dict]:
    references = comparison["references"]
    prompts = comparison["prompts"]
    generations_by_system = comparison["generations_by_system"]

    items = []
    rng = random.Random(seed)
    for i, (prompt, ref) in enumerate(zip(prompts, references)):
        candidates = [("reference", ref)]
        for system, gens in generations_by_system.items():
            candidates.append((system, gens[i]))
        rng.shuffle(candidates)
        for label, text in candidates:
            items.append({"item_index": i, "prompt": prompt, "true_label": label, "text": text})
    rng.shuffle(items)
    return items


st.set_page_config(page_title="Naija-Switch Human Eval", page_icon="📋")
st.title("📋 Naija-Switch Human Evaluation")

comparison_path = _latest_comparison_file()
if comparison_path is None:
    st.warning("No comparison run found. Run `python -m src.evaluation.compare --systems mock` first.")
    st.stop()

st.caption(f"Rating items from: `{comparison_path.name}`")

if "rater_id" not in st.session_state:
    st.session_state.rater_id = None

if st.session_state.rater_id is None:
    rater_id = st.text_input("Rater ID (any name/handle -- not shared publicly)")
    if st.button("Start") and rater_id:
        st.session_state.rater_id = rater_id
        st.rerun()
    st.stop()

if "items" not in st.session_state:
    with comparison_path.open(encoding="utf-8") as f:
        comparison = json.load(f)
    st.session_state.items = _build_rating_items(comparison, seed=hash(st.session_state.rater_id) & 0xFFFF)
    st.session_state.idx = 0

items = st.session_state.items
idx = st.session_state.idx

if idx >= len(items):
    st.success(f"All {len(items)} items rated. Thank you!")
    st.stop()

item = items[idx]
st.progress(idx / len(items))
st.markdown(f"**User said:** {item['prompt']}")
st.markdown(f"**Response ({idx + 1}/{len(items)}):**")
st.info(item["text"])

with st.form(key=f"rating_form_{idx}"):
    scores = {}
    for key, question in DIMENSIONS:
        scores[key] = st.slider(question, min_value=1, max_value=5, value=3, key=f"{key}_{idx}")
    submitted = st.form_submit_button("Submit rating")

if submitted:
    HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HUMAN_EVAL_DIR / f"{st.session_state.rater_id}.jsonl"
    record = {
        "rater_id": st.session_state.rater_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparison_file": comparison_path.name,
        "item_index": item["item_index"],
        "true_label": item["true_label"],  # system identity, hidden from the UI but logged for analysis
        "prompt": item["prompt"],
        "response": item["text"],
        "scores": scores,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    st.session_state.idx += 1
    st.rerun()
