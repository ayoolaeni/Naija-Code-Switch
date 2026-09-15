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

## Running the chat app with Docker (recommended for handing this to someone else)

The chat app (`src/app/ui.py`) is packaged as its own Docker image -- this
is the easiest way for someone without Python installed to run it,
including the actual fine-tuned model (see `HOW_TO_RUN.txt` for a
plain-language version of these same steps written for a non-technical
recipient of a zipped copy of this project).

**This image bakes in `torch`/`transformers`/`peft` and the trained LoRA
adapter under `checkpoints/naija-switch-lora/`**, and defaults to
`INFERENCE_BACKEND=local` -- no `.env` or `HF_TOKEN` needed to see real,
fine-tuned responses out of the box. Trade-off: the image is a few GB
(CPU-only `torch` build, since a client machine isn't assumed to have a
GPU set up for Docker) and generation runs on CPU -- correct, but tens of
seconds to a couple of minutes per reply, not instant. If `checkpoints/
naija-switch-lora/` doesn't exist when building (e.g. you haven't
fine-tuned yet -- see "Fine-tuning on a free cloud GPU" above), the build
will fail at the `COPY checkpoints/naija-switch-lora` step; fine-tune
first, or edit the Dockerfile to drop that line and default to `mock`/`hf`
instead.

**One-time setup (whoever is running it):**
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   and make sure it's running.
2. From the project folder:
   ```bash
   docker compose up -d --build
   ```
   First build downloads the base image, `torch`, `transformers`, etc., and
   on first actual use also downloads the ~3GB base model from Hugging
   Face (cached in a named Docker volume afterward, so this only happens
   once) -- expect this to take a while on a normal connection.
3. Open http://localhost:8501 in a browser.

**To use the hosted API backend instead** (fast, but does *not* use your
fine-tuned adapter -- only the unfine-tuned base model): copy
`.env.example` to `.env`, set `HF_TOKEN` (a free token from
https://huggingface.co/settings/tokens) and `INFERENCE_BACKEND=hf`, then
re-run `docker compose up -d --build`. `docker compose` reads `.env`
automatically; the token is never baked into the image or committed to
git.

**Day to day, after the first build:**
```bash
docker compose up -d      # start
docker compose down       # stop
docker compose logs -f    # view logs
```

## Run it (without Docker, for development)

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
# No local GPU? See "Fine-tuning on a free cloud GPU" below for ready-to-run
# Colab/Kaggle notebooks that do this same real run on free hardware.

# 5. Evaluation harness
python -m src.evaluation.automatic_metrics   # BLEU / BERTScore / chrF demo
python -m src.evaluation.compare --systems mock    # comparative run over data/splits/test.jsonl
streamlit run src/evaluation/human_eval_app.py     # rate the outputs of the comparison run above

# 6. Chat app
streamlit run src/app/ui.py
```

## Fine-tuning on a free cloud GPU (Colab / Kaggle)

`train_lora.py --config configs/training_config.yaml` needs a CUDA GPU to
run against the real base model (Qwen2.5-1.5B-Instruct) -- most laptops
don't have one. Two ready-to-run notebooks do the exact same real run on
free hosted GPUs, no local setup beyond a browser:

- `notebooks/finetune_colab.ipynb` -- upload to
  [colab.research.google.com](https://colab.research.google.com),
  `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`.
  Colab's free GPU quota is a tighter rolling daily-ish limit.
- `notebooks/finetune_kaggle.ipynb` -- upload to
  [kaggle.com](https://www.kaggle.com) (`Code -> New Notebook -> File ->
  Import Notebook`), verify your account by phone once
  (`Settings -> Phone Verification` -- required for GPU/internet access,
  no payment involved), then in the notebook's Settings panel set
  **Accelerator -> GPU T4 x2** and **Internet -> On**, then `Run -> Run
  All`. Kaggle's free quota is weekly (~30 GPU-hours), which tends to be
  more forgiving than Colab's.

Both notebooks: clone this repo fresh, install dependencies, regenerate
`data/processed/`+`data/splits/` from the committed `data/authored/*.jsonl`
sources (those derived files are gitignored -- a fresh clone never has
them), run the CPU smoke test, then run the real fine-tune, and finally
get the trained adapter off the cloud machine (Colab: auto-downloads a zip
via the browser; Kaggle: zips it into `/kaggle/working/` where you download
it from the file browser panel or, more reliably, from the **Output** tab
after `Save Version -> Save & Run All`).

**Known gotchas already fixed in both notebooks** (worth knowing if you
hit a variant of them):
- Both platforms' base images can ship an old `torchao` that newer `peft`
  versions hard-error on when injecting LoRA layers, even though this
  project never uses `torchao` at all -- both notebooks `pip uninstall`
  it right after installing requirements.
- `train_lora.py`'s post-training overfitting-reproduction check used to
  build inputs on CPU and pass them straight to a GPU model, crashing with
  a device-mismatch error on any real GPU run -- fixed in
  `_check_verbatim_reproduction()`.
- `save_strategy="epoch"` with no cap kept a full checkpoint (adapter
  weights **and** optimizer/scheduler state) per epoch on disk --
  `save_total_limit=1` now caps that, and both notebooks' zip step copies
  only the top-level adapter files (skipping any `checkpoint-*/`
  subfolder) so the downloaded zip is a few MB, not 100+ MB of training
  state you don't need for inference.

## Running the chat app against a real fine-tuned adapter

Once you have a trained adapter (from a local run or one of the notebooks
above), extract just its top-level files (`adapter_config.json`,
`adapter_model.safetensors`, tokenizer files -- not any `checkpoint-*/`
subfolder) into `checkpoints/naija-switch-lora/` (this path is gitignored,
so it's local-only; nothing here gets pushed to GitHub). Then set, in
`.env`:

```
INFERENCE_BACKEND=local
LOCAL_BASE_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
LOCAL_LORA_ADAPTER_DIR=checkpoints/naija-switch-lora
```

Run `streamlit run src/app/ui.py` and the sidebar should read **"Active:
local_transformers"**. Two gotchas that cost real debugging time getting
this working, in case they resurface:

- **Always launch via this project's own venv, not a bare `streamlit`
  command.** `streamlit` on your system `PATH` can silently resolve to a
  *different*, unrelated Python install (e.g. a global one with
  `streamlit`/`torch`/`transformers` but no `peft`), which either crashes
  oddly or behaves inconsistently with what's documented here. Use
  `.venv\Scripts\streamlit.exe run src\app\ui.py` (PowerShell) /
  `.venv/Scripts/streamlit.exe run src/app/ui.py` (Git Bash), or activate
  the venv first.
- **No GPU means slow, not broken.** CPU generation for a 1.5B model
  routinely takes tens of seconds to a couple of minutes per reply. A long
  spinner is expected; it is not evidence something is wrong.

(Under the hood: `src/app/inference_engine.py` loads `.env` via an
explicit path derived from `Path(__file__)`, not python-dotenv's default
"walk up from the caller's file" search -- that default search breaks
under `streamlit run` specifically, because Streamlit executes the app
script through its own `exec` mechanism rather than a normal import, so
the default heuristic silently finds no `.env` and loads nothing, with no
error. If you ever see the sidebar say `mock` despite a correctly-set
`.env`, this is the first thing to suspect.)

## Repository layout

Matches the research brief's §9 structure: `data/` (raw/authored/processed/splits),
`src/data_pipeline`, `src/annotation`, `src/modeling`, `src/evaluation`,
`src/app`, `configs/`, `logs/`. `notebooks/` holds the Colab/Kaggle
fine-tuning notebooks described above (not part of the brief's original
structure, added for free-GPU access).

## Scope of this build

This is a genuine end-to-end implementation of every component the research
brief describes -- not a stub. Concretely, today, with zero paid
infrastructure:

- The full data pipeline runs on a **797-dialogue authored corpus** (3,188
  turns across the 4 in-scope domains: greetings, qa, customer_support,
  everyday_chat), split across two files under `data/authored/`:
  - `dialogues.jsonl` -- the original **29-dialogue hand-authored seed**
    (~116 turns, contributor ids `c1`-`c4`), written by hand
    (`scripts/author_seed_dialogues.py`).
  - `dialogues_synthetic.jsonl` -- a **768-dialogue script-generated batch**
    (~3,072 turns, contributor ids `synth_c1`-`synth_c4`, kept in a
    deliberately distinct id namespace for traceable provenance), produced
    by `scripts/generate_synthetic_dialogues.py` from ~192 distinct
    hand-written 4-turn scenario skeletons (48 per domain) each rendered
    under 4 orthographic "contributor spelling profiles" (README §6.3:
    `una`/`unu`, `sabi`/`savvy`, `dey`/`de`, `abeg`/`abeg o`, `wan`/`wan
    na`/`wanna`, `no wahala`/`no wahala at all`, trailing `-o`/`sha`/`sef`
    tags) rather than normalized/duplicated text.
  - **Numerically**, 3,188 turns now sits inside the research brief's
    target range of 3,000-5,000 conversational turn pairs (§3/§6.3) for the
    first time. Read that plainly, though: the brief's target implicitly
    assumes turns collected/authored by "habitual Nigerian code-switchers"
    (§6.3) -- a pool of real, distinct human contributors. ~97% of this
    corpus's turns are instead one script's templated output authored by
    an AI assistant, not organically collected from many real people, so
    it should not be read as satisfying the brief's diversity intent even
    though it now clears the numeric target. Scaling the **real-contributor**
    portion of the dataset remains a staffing question, not a missing-code
    one.
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
  verbatim-reproduction check) is verified correct on CPU in seconds. A
  real run against Qwen2.5-1.5B-Instruct needs a CUDA GPU and the
  multi-GB base model weights -- this has since actually been done (not
  just proven runnable) on a free Kaggle T4 GPU via
  `notebooks/finetune_kaggle.ipynb`, producing a real trained LoRA
  adapter loadable by the chat app's `local` backend (see "Running the
  chat app against a real fine-tuned adapter" above). That adapter itself
  isn't committed to this repo (`checkpoints/` is gitignored, since it's a
  local build artifact, not source) -- re-run the notebook to reproduce
  it.
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
behind it, exercised end-to-end on a dataset that now clears the brief's
numeric 3,000-5,000 turn target (3,188 turns) -- though, as noted above,
only ~3% of those turns are hand-authored by a person; the rest is one
script's templated output, not the brief's intended pool of real habitual
Nigerian code-switchers. Scaling to a real-contributor-authored dataset at
this size, a real GPU fine-tune, live commercial baselines, and a real
human-rater study remain an infrastructure and staffing question, not a
missing-code question.
