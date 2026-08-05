# Naija-Switch (engineering build)

Implementation of the research brief in [`README (1).md`](<README (1).md>): a
code-switch-aware chatbot for English-Nigerian Pidgin conversation. This
file documents the engineering build that lives in this repo -- setup, how
to run each pipeline stage, and exactly what runs today on a laptop versus
what needs a GPU / real contributors / paid API access to reach the
research design's full scale.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat on cmd
pip install -r requirements.txt
cp .env.example .env            # then fill in HF_TOKEN to get live chat responses
```

`bitsandbytes` (QLoRA) is Linux-only in `requirements.txt` -- on Windows/macOS
without a supported GPU, `train_lora.py` automatically falls back to full
precision (see "Scope of this build" below).

## Run it

```bash
# 1. Data pipeline: load sources -> filter -> privacy-strip -> CMI-score
python -m src.data_pipeline.pipeline
python -m src.data_pipeline.split          # dialogue-level 80/10/10 split

# 2. Annotation + agreement
python -m src.annotation.annotate_cli --annotator-id you   # interactive
python -m src.annotation.agreement --demo                  # Fleiss' kappa / Krippendorff's alpha demo

# 3. Model selection + prompt design
python -m src.modeling.baseline_probe       # comparison table (needs HF_TOKEN for the live-probe column)
python -m src.modeling.prompt_design        # prints the frozen system prompt

# 4. LoRA/QLoRA fine-tuning
python -m src.modeling.train_lora --smoke-test               # proves the training code path works, CPU, seconds
python -m src.modeling.train_lora --config configs/training_config.yaml   # real run, needs a GPU

# 5. Evaluation harness
python -m src.evaluation.automatic_metrics   # BLEU / BERTScore / chrF demo
python -m src.evaluation.compare --systems mock    # comparative run over data/splits/test.jsonl
streamlit run src/evaluation/human_eval_app.py     # rate the outputs of the comparison run above

# 6. Chat app
streamlit run src/app/ui.py
```

## Repository layout

Matches the research brief's §9 structure: `data/` (raw/authored/processed/splits),
`src/data_pipeline`, `src/annotation`, `src/modeling`, `src/evaluation`,
`src/app`, `configs/`, `logs/`.

## Scope of this build

This is a genuine end-to-end implementation of every component the research
brief describes -- not a stub. Concretely, today, with zero paid
infrastructure:

- The full data pipeline runs on a **29-dialogue hand-authored seed corpus**
  (`data/authored/dialogues.jsonl`, ~116 turns across the 4 in-scope
  domains) -- not the target 3,000-5,000 turns, which needs a pool of real
  Nigerian code-switching contributors this environment doesn't have.
- `src/data_pipeline/sources/{naijasenti,afrisenti,flores200}.py` are real
  loaders against the actual public Hugging Face datasets named in the
  research brief. They degrade gracefully (log a warning, return no data)
  if a dataset has been renamed/gated/deprecated on the Hub since the
  brief was written, or if there's no network access -- this was observed
  and handled during this build (NaijaSenti/AfriSenti's original loading
  scripts are no longer supported by current `datasets`, and FLORES-200 is
  now gated). The pipeline still completes successfully on the authored
  source alone.
- `train_lora.py` is a real `transformers` + `peft` LoRA/QLoRA script. Its
  `--smoke-test` flag swaps in a tiny public model so the full training
  loop (LoRA injection, tokenization, loss masking, the overfitting
  verbatim-reproduction check) is verified correct on CPU in seconds.
  Running it for real against Llama 3 / Qwen2.5 / Gemma 2 needs a CUDA GPU
  and the multi-GB base model weights.
- The chat app's default backend is a deterministic `MockBackend` so the
  UI, dialogue manager (with context truncation), and logging are
  verifiably working with **zero configuration**. Add `HF_TOKEN` to `.env`
  to switch to live generation via the Hugging Face Inference API
  (`INFERENCE_BACKEND=hf`, or leave `INFERENCE_BACKEND` unset to
  auto-select once a token is present); switch to `INFERENCE_BACKEND=local`
  once you have a GPU and/or a trained LoRA adapter.
- `src/evaluation/compare.py` runs the same harness across any registered
  `Backend`s (`mock` works out of the box; `hf`/`local` once configured).
  General-purpose commercial chatbot baselines (README §7.4 point 3) are
  not wired to any live API in this build -- no such keys were provided --
  but any Backend-shaped wrapper around one can be registered in
  `_resolve_backend()` and compared identically.
- Inter-annotator agreement (`agreement.py --demo`) and the bootstrap /
  Wilcoxon significance tests (`stats.py`) run against real math
  (`statsmodels`/`krippendorff`/`scipy`), demonstrated on a synthetic
  double-annotation and synthetic ratings respectively, since there's no
  pool of real human annotators/raters in this environment. Swap in real
  annotator files / real rating logs and the same functions apply
  unchanged.

In short: every deliverable in the research brief has real, runnable code
behind it, exercised end-to-end on a small seed dataset. Scaling to the
brief's full target (3-5k authored+curated turns, a real GPU fine-tune,
live commercial baselines, a real human-rater study) is an infrastructure
and staffing question, not a missing-code question.
