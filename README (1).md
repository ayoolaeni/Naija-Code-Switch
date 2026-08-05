# Naija-Switch: A Code-Switch-Aware Chatbot for English–Nigerian Pidgin Conversations

> Design and development of a conversational agent that understands and responds naturally to English–Nigerian Pidgin code-switched input, adapted from an open-weight LLM using parameter-efficient fine-tuning.

This README is written for an AI coding assistant (e.g., Claude Code) to use as the primary spec when scaffolding and building this application. It summarizes the academic project (an MIT research project at the University of Lagos) into an actionable engineering brief. Where the underlying research document is more authoritative than a specific engineering choice below, prefer fidelity to the research design over convenience.

---

## 1. Project Summary

**Problem:** General-purpose chatbots are trained overwhelmingly on monolingual, standard-English text. Nigerian users routinely mix English and Nigerian Pidgin ("Naija") within the same message or even the same clause — e.g. *"Abeg help me check my account balance"* or *"I don tire, make we continue tomorrow"*. Because Pidgin is not simply "broken English" but a distinct language with its own grammar (aspect markers like `don`, hortatives like `make we`, discourse particles like `abeg`), general LLMs misinterpret intent, lose conversational context, and produce unnatural responses on this input.

**Goal:** Build and evaluate a chatbot that treats English–Pidgin code-switching as normal input rather than noise, by adapting an open-weight LLM on a purpose-built code-switched conversational dataset, and compare it against unadapted/general-purpose baselines.

**This is a research + engineering deliverable.** The AI assistant should build:
1. A **dataset construction toolkit** (collection, cleaning, annotation, quality control).
2. A **model adaptation pipeline** (baseline probing, prompt design, LoRA/QLoRA fine-tuning).
3. An **evaluation harness** (automatic metrics, code-switching-specific metrics, human evaluation tooling).
4. A **working chat application** (UI + dialogue manager + inference engine + logging).

---

## 2. Objectives (map directly to deliverables)

| # | Objective | Deliverable |
|---|---|---|
| 1 | Determine the best LLM adaptation strategy for low-resource code-switched dialogue | A short comparison script/report: prompting vs in-context learning vs full fine-tuning vs PEFT (LoRA/QLoRA) |
| 2 | Build a chatbot that handles English–Pidgin mixed input | Annotated dataset + adapted model + serving pipeline |
| 3 | Evaluate the chatbot on code-switched conversations | Automatic metrics + code-switching metrics + human eval pipeline |
| 4 | Compare against general-purpose chatbots | Comparative evaluation harness on a held-out test set |

**Decision already made by the research design (do not re-litigate unless data disagrees):** parameter-efficient fine-tuning (**LoRA**, with **QLoRA** as a memory-constrained fallback) on top of a well-designed system prompt, applied to one of: **Llama 3**, **Qwen2.5**, or **Gemma 2** (final choice determined empirically — see §5).

---

## 3. Scope & Constraints

- **Languages:** English and Nigerian Pidgin only. No Yoruba/Igbo/Hausa, no other West African Pidgin varieties.
- **Domain:** General everyday conversation — greetings, basic Q&A, light customer support, casual chat. **Not** for medical, legal, or financial advice — the app should include a disclaimer and should not be tuned to give confident advice in these domains.
- **Modality:** Text only. No speech-to-text or text-to-speech.
- **Models:** Only open-weight models may be fine-tuned (Llama 3 / Qwen2.5 / Gemma 2 families). Closed/commercial models (e.g. via API) are used **only** as comparison baselines, never fine-tuned.
- **Compute:** Assume a single consumer/cloud GPU (e.g. one T4/A10/A100 notebook instance) with limited VRAM. Favor 4-bit quantized training (QLoRA) as the default-safe path; prefer models in the 2B–8B parameter range.
- **Data:** No public English–Pidgin conversational dataset exists. Expect the dataset to be small (target: **3,000–5,000 conversational turn pairs**), assembled from a mix of adapted existing corpora and manually authored dialogues. Build tooling accordingly — do not assume large-scale data availability.

---

## 4. System Architecture

Four components, per the research design (Section 3.7):

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌─────────────┐
│  Chat UI (web)   │────▶│  Dialogue Manager │────▶│  Inference Engine  │────▶│   Logging   │
│  (Streamlit/     │◀────│  (history, system │◀────│  (adapted LLM +    │     │  (inputs,   │
│   Gradio/simple  │     │   prompt, context  │     │   LoRA adapters,   │     │  outputs,   │
│   React+FastAPI) │     │   truncation)      │     │   local or hosted) │     │  gen params)│
└─────────────────┘     └──────────────────┘     └───────────────────┘     └─────────────┘
```

- **UI:** Minimal — a chat window with message history. The contribution of this project is the language handling, not UI polish. Streamlit or Gradio is sufficient for the prototype; a small React frontend + FastAPI backend is an acceptable alternative if the assistant is asked for something more production-like.
- **Dialogue Manager:** Maintains conversation history, prepends the system prompt, truncates history when approaching the model's context window. This is the component responsible for fixing "failure mode 3" (loss of context in multi-turn Pidgin exchanges) — do not let this be an afterthought.
- **Inference Engine:** Loads the base model + trained LoRA adapters (via Hugging Face PEFT). Should support both local inference and a hosted/remote runtime.
- **Logging:** Every exchange (input, output, generation parameters, timestamp) is logged to support the evaluation pipeline in §7 and to make results traceable.
- **No translation pipeline.** User input goes directly to the adapted model as written — do not build a "translate to English → process → translate back" architecture. This was explicitly rejected in the research design because English and Pidgin share most of their lexicon, making language segmentation itself unreliable.

---

## 5. Model Selection & Adaptation Pipeline

### 5.1 Candidates
- Llama 3 (Meta)
- Qwen2.5 (Alibaba)
- Gemma 2 (Google)

### 5.2 Selection process (build this as a script/notebook)
For each candidate, measure:
1. **Baseline competence** — zero-shot and few-shot performance on a held-out validation sample of code-switched dialogue (before any fine-tuning).
2. **Parameter scale / memory footprint** — must fit the available compute budget with LoRA/QLoRA.
3. **Tokenization behavior on Pidgin** — measure token fragmentation on a fixed Pidgin text sample (heavy fragmentation = poor representation = costlier adaptation).
4. **Ecosystem support** — availability of stable Hugging Face `transformers` / `peft` / `bitsandbytes` support.
5. **License** — must permit research use, modification, and reporting of results.

Output a small comparison table/report; pick the model with the best combination of baseline competence and feasibility.

### 5.3 Prompt design
Build a system prompt that:
- Establishes the assistant's role and tone.
- Instructs the model to **respond in the same mixture of English and Pidgin the user used** (don't default to pure English).
- Includes 2–4 worked few-shot examples of natural code-switched exchanges.

Iterate prompt variants on the validation set only; freeze the winning prompt before touching the test set.

### 5.4 Fine-tuning
- **Method:** LoRA (`peft` library), targeting attention query/value projections initially.
- **Fallback for memory constraints:** QLoRA — 4-bit quantized base model + 16-bit adapters (`bitsandbytes`).
- **Training data format:** instruction–response pairs that **preserve preceding conversational turns** as context (not single-turn pairs) — this is essential to teaching context retention.
- **Starting hyperparameters** (tune on validation, never on test):

| Parameter | Initial value |
|---|---|
| LoRA rank (r) | 8–32 |
| LoRA alpha | ~2× rank |
| Target modules | attention Q/V projections (expand only if justified) |
| Learning rate | 1e-4 to 2e-4 |
| Batch size | small, with gradient accumulation |
| Epochs | 3–5, with early stopping on validation loss |
| Precision | 4-bit base + 16-bit adapters (QLoRA) |
| Seed | fixed and logged |

- **Overfitting guardrails** (small-dataset risk): monitor validation loss for early stopping; prefer conservative LoRA rank; check generated test outputs for verbatim reproduction of training examples and flag/report any found.

---

## 6. Dataset Construction Toolkit

No public English–Pidgin **conversational, code-switched** dataset exists. Build tooling to assemble one from a mix of sources — expect the manually authored portion to be the core of the dataset, not a supplement.

### 6.1 Candidate sources (build a loader/adapter per usable source)

| Source | Usable for | Notes / Caveats |
|---|---|---|
| NaijaSynCor / Naija treebank (Caron et al., 2019) | Conversational structure, authentic turn-taking | Transcribed **speech** — needs conversion to written chat register |
| NaijaSenti (Muhammad et al., 2022) | Authentic code-mixed tokens/orthography | Isolated tweets, not dialogue — needs reconstruction into exchanges |
| AfriSenti (Muhammad et al., 2023) | Additional Pidgin lexical coverage | Single utterances, no conversational context |
| PidginUNMT corpus (Ogueji & Ahia, 2019) | Pidgin lexical/grammatical grounding | Mostly monolingual Pidgin, news register, little code-switching |
| FLORES-200 (NLLB Team, 2022) | Reference parallel English–Pidgin sentences | Translated, not naturally occurring dialogue — reference only |
| Public social media | Contemporary, naturally code-switched chat register | **Only use sources whose published ToS explicitly permit research use.** Skip otherwise. |
| Manually authored dialogues | Core training data — direct control over domain/switch density | Mitigate author bias with multiple contributors |

**Explicitly excluded / not usable for this purpose:**
- **NaijaVoices** (Emezue et al., 2025) — speech dataset for Igbo/Hausa/Yoruba, **not Pidgin**, not text. Do not use.
- FLORES-200 — reference/parallel sentences only, not dialogue; use only as a reference set, not training data.

### 6.2 Pipeline steps to implement
1. Retrieve and license-tag each source.
2. Filter for genuine English–Pidgin mixing (token-level language ID pass + manual verification of a sample — automatic ID is unreliable for this pair since lexicons overlap heavily).
3. Convert transcribed/spoken material (Naija treebank) into written chat register; strip prosodic/transcription artifacts.
4. Author additional dialogues to cover gaps in domain/switch-pattern coverage (see authoring guidelines below).
5. Annotate every turn (see §6.3).
6. Run QA / inter-annotator agreement (see §6.4).
7. Partition into train/validation/test (see §6.5).

### 6.3 Authoring guidelines (build a lightweight annotation/authoring tool around these rules)
- Contributors should be habitual Nigerian code-switchers, not just the developer.
- Minimum 4 turns per dialogue (teaches multi-turn context, not isolated Q&A).
- Allow switch points to fall anywhere naturally, including mid-clause — do not constrain to sentence boundaries.
- **Do not normalize spelling** across contributors — orthographic variability is a real feature of written Pidgin and should be preserved in the data.
- Target: 3,000–5,000 conversational turn pairs across the four in-scope domains (greetings, Q&A, light customer support, everyday chat).

### 6.4 Annotation scheme
Per turn, tag:
- Conversational domain
- Speaker role
- Token-level language tags: `English` / `Pidgin` / `Language-independent` (named entities, numerals, punctuation)

Compute and report:
- **Code-Mixing Index (CMI)** (Gambäck & Das, 2016) — proportion of non-dominant-language tokens per utterance.
- Corpus-level **Multilingual Index** and **Integration Index** (Guzmán et al., 2017), for comparability with other code-switching corpora.

QA:
- ≥10% of the dataset double-annotated.
- **Fleiss' kappa** for token-level language tag agreement.
- **Krippendorff's alpha** for ordinal quality judgments (human eval scores).
- Log agreement scores honestly, even if low — a low score is itself a finding.

### 6.5 Partitioning
- ~80/10/10 train/validation/test split.
- Split at the **whole-dialogue** level, never at the individual-turn level (prevents leakage).
- Distribute authored vs. retrieved material proportionally across all three partitions.
- Test set held out entirely until final evaluation; never used for hyperparameter tuning.

---

## 7. Evaluation Harness

Build a reusable evaluation script that runs identically across all compared systems (unadapted baseline, adapted model, general-purpose chatbot baselines) on the identical held-out test set.

### 7.1 Automatic metrics
- **BLEU** (`sacrebleu`) — n-gram overlap vs. reference response.
- **BERTScore** (`bert-score`) — contextual embedding similarity, credits valid paraphrase.
- **chrF** — character n-gram based, more robust to Pidgin's orthographic variability than word-level metrics.

### 7.2 Code-switching-specific metrics
- **Switching-density comparison:** compute CMI over generated responses vs. CMI over human reference responses for the same inputs. A system that under-mixes relative to references is defaulting to English — flag this explicitly.
- **Language-appropriateness rate:** proportion of responses judged (by human raters) as using an appropriate language mixture for the given input.

### 7.3 Human evaluation
Build a simple rating tool (e.g., a Streamlit/Gradio app or a spreadsheet-based workflow) presenting:
- Responses from all systems **in randomized order, system identity hidden**.
- Include human reference responses among rated items, unlabeled, as an upper reference point.

Rate on a 5-point scale across 4 dimensions:

| Dimension | Question | Maps to |
|---|---|---|
| Relevance | Does the response address what the user asked? | RQ3 |
| Language understanding | Was the mixed-language input correctly interpreted? | Failure mode 1 (misreading Pidgin grammar) |
| Contextual coherence | Does it follow from prior turns? | Failure mode 3 (context loss) |
| Naturalness / appropriateness | Would a Nigerian speaker plausibly write this? | RQ4 |

Compute inter-rater agreement with **Krippendorff's alpha**.

### 7.4 Comparative evaluation
Evaluate at minimum these conditions on the identical test set:
1. Selected model, unadapted (baseline).
2. Selected model, after LoRA/QLoRA adaptation.
3. ≥2 general-purpose chatbots accessed via public interface (baseline only — never fine-tuned).

Record access date and model version for the general-purpose baselines (they change over time without notice).

### 7.5 Statistical analysis
- **Bootstrap resampling** (Koehn, 2004) for automatic metric significance testing.
- **Non-parametric test** (e.g., Wilcoxon signed-rank / Mann-Whitney) for ordinal human ratings.
- Report non-significant differences plainly as such — do not overstate small-sample results.

---

## 8. Suggested Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.10+ |
| Model/training | PyTorch, Hugging Face `transformers`, `peft`, `bitsandbytes` |
| Compute | Cloud GPU notebook (Colab/Kaggle/cloud VM) or local GPU |
| Evaluation | `sacrebleu`, `bert-score`, custom CMI implementation |
| Statistics | `scipy`, `statsmodels` |
| Interface | Streamlit or Gradio (fastest path); FastAPI + React optional for a more production-like build |
| Data handling | `pandas`, `datasets` (Hugging Face) |
| Version control | Git |

---

## 9. Suggested Repository Structure

```
naija-switch/
├── README.md
├── data/
│   ├── raw/                  # retrieved corpora, license-tagged
│   ├── authored/             # manually authored dialogues
│   ├── processed/            # cleaned, annotated, CMI-scored
│   └── splits/               # train/val/test (dialogue-level split)
├── src/
│   ├── data_pipeline/        # collection, cleaning, filtering, CMI scoring
│   ├── annotation/           # annotation tool + agreement calculators
│   ├── modeling/
│   │   ├── baseline_probe.py # zero/few-shot probing across candidate models
│   │   ├── prompt_design.py
│   │   ├── train_lora.py     # LoRA/QLoRA fine-tuning
│   │   └── inference.py
│   ├── evaluation/
│   │   ├── automatic_metrics.py   # BLEU, BERTScore, chrF
│   │   ├── cs_metrics.py          # CMI, switching-density, appropriateness
│   │   ├── human_eval_app.py      # rating interface
│   │   └── stats.py               # bootstrap, significance tests
│   └── app/
│       ├── dialogue_manager.py
│       ├── inference_engine.py
│       ├── logging.py
│       └── ui.py              # Streamlit/Gradio chat app
├── configs/
│   └── training_config.yaml
├── notebooks/                 # exploratory / model-selection notebooks
├── logs/                      # conversation logs for evaluation traceability
└── requirements.txt
```

---

## 10. Ethical & Practical Guardrails (build these into the code, not just as policy)

- **Data provenance:** only use sources whose published terms permit research use. If unclear, exclude the source.
- **Privacy:** strip usernames, handles, phone numbers, emails, and other direct identifiers from any retrieved text before storing it.
- **Consent:** dialogue contributors and human-eval judges must be informed of study purpose and usage; participation is voluntary; no identifying info reported, ratings aggregated only.
- **Scope disclaimer in the app:** the chatbot must not present itself as a source of medical, legal, or financial advice.
- **Representation:** the dataset likely skews toward whatever region/register the contributors and sources come from. The app/report should say so plainly rather than implying broad regional coverage of Nigerian Pidgin.

---

## 11. Build Order (recommended sequence for the AI assistant)

1. Scaffold the repo structure above.
2. Build the data pipeline: source loaders → filtering/language-ID pass → CMI scoring utility.
3. Build the annotation tool and agreement calculators; author an initial small batch of dialogues to validate the schema end-to-end.
4. Build `baseline_probe.py` to zero/few-shot test the three candidate model families and produce the selection table (§5.2).
5. Implement prompt design + few-shot templates; validate on the validation split.
6. Implement the LoRA/QLoRA training script and run adaptation once the dataset reaches a usable size.
7. Build the evaluation harness (automatic + CS-specific metrics; stats module).
8. Build the human-eval rating app.
9. Build the chat application (dialogue manager, inference engine, logging, UI) wired to the adapted model.
10. Run the full comparative evaluation (unadapted vs. adapted vs. general-purpose baselines) and generate the results report.

---

## 12. Key Terms (for quick reference)

- **Code-switching:** alternating use of two+ languages within one conversation/utterance.
- **Code-mixing / intra-sentential switching:** language alternation within a clause.
- **Naija / Nigerian Pidgin:** English-lexified contact language with its own distinct grammar, ~100M speakers.
- **CMI (Code-Mixing Index):** proportion of tokens not in an utterance's dominant language; quantifies switching density.
- **LoRA:** Low-Rank Adaptation — parameter-efficient fine-tuning via small trainable low-rank matrices, base weights frozen.
- **QLoRA:** LoRA on top of a 4-bit quantized base model, for memory-constrained training.

---


