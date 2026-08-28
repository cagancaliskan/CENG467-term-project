# CODEMAP — what lives where

A file-and-folder guide to this repository, so a new reader can find any piece of
the pipeline without opening every file. For *how to run* things, see
[`README.md`](README.md); this document explains *what each file is for*.

The project distils a large teacher LLM (GPT-4o-mini / Claude Haiku 4.5) into a
small student (mT5-small + LoRA) for Turkish abstractive news summarization, and
evaluates it against three baselines on MLSUM-TR and TR-News.

---

## 1. Top-level layout

| Path | What it is for |
| --- | --- |
| `README.md` | Project overview, headline results, setup, end-to-end run commands. |
| `CODEMAP.md` | **This file** — file-by-file / folder-by-folder guide. |
| `PREREGISTRATION.md` | Primary outcome, equivalence margin, seed plan and judge-exclusion rule, fixed before the analysis. |
| `LICENSE` / `CITATION.cff` | MIT licence and citation metadata. |
| `requirements.txt` | Pinned Python dependencies (Colab T4 compatible). |
| `.env.example` | Template for `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. |
| `configs/` | YAML configs for data, training presets, and budget. |
| `src/` | All Python source (data, teachers, student, eval, utils). |
| `scripts/` | Bash entry points wrapping each pipeline stage. |
| `notebooks/` | Three Colab notebooks (week1, train, eval). |
| `demo/` | Gradio web demo. |
| `data/` | Datasets and synthetic caches (mostly git-ignored). |
| `outputs/` | Predictions, checkpoints, logs, and committed result tables. |
| `report/` | IEEE LaTeX paper (`main_ieee.tex`), figures, compiled PDF. |

---

## 2. Pipeline at a glance (data flow)

```
scripts/01_download_data.sh   → src/data/load_mlsum.py , load_trnews.py
        └─ data/raw/*.jsonl  {id, article, reference, topic}

scripts/02_generate_teacher.sh → src/teachers/generate{,_batch}.py
        └─ data/synthetic/<provider>/<variant>/<id>.json   (teacher summaries)
        └─ src/data/prepare_synthetic.py → training pairs

scripts/03_train_student.sh    → src/student/train.py  (mT5-small + LoRA)
        └─ outputs/checkpoints/<run>/   (LoRA adapters)

(inference)                    → src/student/infer.py , infer_teacher.py
        └─ outputs/predictions/<system>.jsonl  {id, prediction, reference, article}

scripts/04_evaluate.sh         → src/eval/run_eval.py
        ├─ rouge_tr.py , bertscore_eval.py , error_analysis.py
        └─ outputs/results/*.json   (metric tables)

(analysis)                     → src/eval/bias_analysis.py        (ethics / §7)
                               → src/eval/qual_*.py               (LLM-as-judge)
                               → src/eval/make_figures.py         (report figures)
        └─ report/figures/*.png  +  outputs/results/*

report/main.tex  ── compiles with report/figures/ ──→  report/main.pdf
```

---

## 3. `src/` — Python source

### `src/data/` — dataset loading and prep
| File | Contains |
| --- | --- |
| `load_mlsum.py` | MLSUM-TR loader with HuggingFace + mirror fallbacks; normalizes to `{id, article, reference, topic, url, date, split, source}`; SHA-1 dedup + min-length filter. |
| `load_trnews.py` | TR-News loader for out-of-domain evaluation; same normalized schema. |
| `make_pilot.py` | Samples a small pilot subset (e.g. `pilot_100`) for quick smoke tests. |
| `prepare_synthetic.py` | Joins per-article teacher caches into student training pairs (synthetic summary ↔ article). |

### `src/teachers/` — teacher LLM generation
| File | Contains |
| --- | --- |
| `prompts.py` | The two Turkish prompt variants (`concise`, `detailed`) used for every teacher call. Documented verbatim for reproducibility. |
| `base.py` | Shared teacher-client interface/base class. |
| `openai_teacher.py` | GPT-4o-mini client wrapper. |
| `anthropic_teacher.py` | Claude Haiku 4.5 client wrapper. |
| `generate.py` | Real-time teacher generation with per-article JSON caching (safe to interrupt/resume). |
| `generate_batch.py` | Anthropic Batch-API generation (~50% cheaper); writes the same cache format as `generate.py`. |

### `src/student/` — student model
| File | Contains |
| --- | --- |
| `train.py` | Fine-tunes `google/mt5-small` with LoRA adapters (fp32, 3 epochs, eff. batch 16). Entry point for B2 and all S-* students. |
| `peft_utils.py` | LoRA config/attach helpers (rank, target `q`/`v` projections). |
| `infer.py` | Batched generation for the student (or any seq2seq); **strips `<extra_id_*>` sentinels**; writes `outputs/predictions/<system>.jsonl`. |
| `infer_teacher.py` | Materializes teacher predictions from the cache into the same prediction-JSONL schema for scoring. |

### `src/eval/` — metrics and analysis
| File | Contains |
| --- | --- |
| `run_eval.py` | Main scoring driver. Takes `--pred label=path` and emits the aggregated metric table (ROUGE + BERTScore + error flags). |
| `rouge_tr.py` | Turkish-aware ROUGE: `standard` (Unicode lowercase) and `stem` (5-char prefix) variants. |
| `bertscore_eval.py` | BERTScore F1 using `xlm-roberta-large`. |
| `error_analysis.py` | Per-prediction heuristic flags: 4-gram repetition, hallucinated numbers (`halluc#`), source extractive overlap, length ratio, morphology flag. |
| **`bias_analysis.py`** | **Ethics/bias probes (report §7), post-hoc, no retraining**: gender representation (gendered-term + name lexicon vs. source/references), foreign/English token leakage (q/w/x signal), and representational-harm error taxonomy from the judge notes. Writes `outputs/results/bias_results.json` and `report/figures/fig6–8`. Self-test: `python -m src.eval.bias_analysis --self-test`. |
| `qual_llm_judge.py` | Runs the LLM-as-judge (Claude Opus) over a stratified sample, scoring five binary axes. |
| `qual_aggregate.py` | Rubric definition + aggregation of judge labels into per-system pass rates. |
| `qual_merge.py` | Merges externally produced judge labels into the canonical label file. |
| `qual_export.py` | Exports the stratified sample for (re)annotation. |
| `make_figures.py` | Generates the main report figures (`fig1`–`fig5`). |

### `src/utils/` — shared helpers
| File | Contains |
| --- | --- |
| `io.py` | `read_jsonl` / `write_jsonl`, run-config dump helpers. |
| `logging.py` | Project logger factory. |
| `seed.py` | `set_seed` for reproducible runs. |

---

## 4. `scripts/` — bash entry points
| File | Wraps |
| --- | --- |
| `01_download_data.sh` | MLSUM-TR + TR-News download/normalization. |
| `02_generate_teacher.sh` | Teacher generation: `… <provider> <variant> <n>`. |
| `03_train_student.sh` | Student training: `--teacher … --size … --lora-rank …`. |
| `04_evaluate.sh` | Full evaluation across the six systems. |
| `run_all_ablations.sh` | All four ablations (size, LoRA rank, prompt, teacher). |
| `check_cache.py` | Reports teacher-cache coverage / missing ids. |
| `inspect_outputs.py` | Quick peek at prediction JSONL contents. |

---

## 5. `configs/` — YAML configuration
| File | Contains |
| --- | --- |
| `data_config.yaml` | Split caps (20k/2k/2k), filtering thresholds, truncation length. |
| `teacher_prompts.yaml` | Prompt-variant selection and teacher decoding settings. |
| `budget.yaml` | Line-by-line cost breakdown. |
| `train/student_default.yaml` | Default student preset (rank 8, 10k synthetic). |
| `train/baseline_human.yaml` | B2 human-supervised preset. |
| `train/ablation_size.yaml` | Dataset-size sweep (1k/5k/10k). |
| `train/ablation_lora.yaml` | LoRA-rank sweep (4/8/16/32). |
| `train/ablation_prompt.yaml` | Concise vs. detailed prompt. |

---

## 6. `notebooks/` and `demo/`
| File | Contains |
| --- | --- |
| `notebooks/colab_week1.ipynb` | Drive mount, install, data download + pilot. |
| `notebooks/colab_train.ipynb` | Student training on Colab T4. |
| `notebooks/colab_eval.ipynb` | Evaluation + figure generation. |
| `demo/app.py` | Gradio app showing six side-by-side summaries; loads LoRA student checkpoints + teacher API clients. |

---

## 7. `data/` and `outputs/` — what is committed vs. generated

Most large artifacts are git-ignored (regenerable). What you see in a fresh clone:

| Path | Committed? | Contents |
| --- | --- | --- |
| `data/raw/` | ignored | Normalized MLSUM-TR / TR-News JSONL (regenerate via script 01). |
| `data/processed/` | ignored | Tokenized training sets. |
| `data/synthetic/` | **committed** | Teacher summary caches (`openai/`, `anthropic/`) — the distillation supervision. |
| `outputs/predictions/` | ignored | Per-system generated summaries `{id, prediction, reference, article}`. |
| `outputs/checkpoints/` | ignored | Trained LoRA adapters. |
| `outputs/logs/` | ignored | Training/eval logs. |
| `outputs/results/` | **committed** | Metric + analysis tables: `qual_labels_filled.csv`, `qual_summary.{json,md}`, `opus_judgments.csv`, and **`bias_results.json`** (ethics analysis). |

---

## 8. `report/` — the paper
| Path | Contains |
| --- | --- |
| `main_ieee.tex` | IEEE LaTeX source (camera-ready). |
| `main_ieee.pdf` | Compiled camera-ready. |
| `figures/fig1–fig5` | Main-results, scaling, LoRA, OOD, and quality-vs-faithfulness figures. |
| `figures/fig6_bias_gender.png` | §7 gender-representation figure. |
| `figures/fig7_bias_leakage.png` | §7 foreign-token-leakage figure. |
| `figures/fig8_bias_repharm.png` | §7 representational-harm figure. |
| `docs/development/` | Planning and process notes. `docs/archive/` holds superseded drafts. |

---

## 9. Find it fast

| I want to… | Go to |
| --- | --- |
| Change the teacher prompts | `src/teachers/prompts.py` |
| See/!change ROUGE or stemming | `src/eval/rouge_tr.py` |
| Add a faithfulness/error flag | `src/eval/error_analysis.py` |
| Reproduce the bias/ethics numbers | `src/eval/bias_analysis.py` → `outputs/results/bias_results.json` |
| Reproduce the headline metric table | `src/eval/run_eval.py` (see README "Reproducing the report numbers") |
| Change training hyper-params | `configs/train/*.yaml` → consumed by `src/student/train.py` |
| Understand the prediction file schema | `src/student/infer.py` (output record format) |
| Edit the paper | `report/main.tex` (+ `report/figures/`) |

---

## 10. Command-line entry points

Every module runs with `--help`:

```bash
python -m src.data.load_mlsum --help
python -m src.teachers.generate_batch --help
python -m src.student.train --help
python -m src.student.infer --help
python -m src.eval.run_eval --help
python -m src.eval.bias_analysis --help      # ethics / bias analysis (no GPU, no API key)
```
