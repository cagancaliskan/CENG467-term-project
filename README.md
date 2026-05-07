# Synthetic Data Distillation for Turkish Abstractive News Summarization

Term project — **CENG 467 Natural Language Understanding and Generation**, Izmir Institute of Technology, Spring 2026. Instructor: Prof. Dr. Aytuğ Onan.

## What this project does

We prompt large teacher LLMs (GPT-4o-mini and Claude) to generate abstractive summaries of Turkish news articles, then fine-tune a small student model (mT5-small, ~300M parameters) on those synthetic summaries using LoRA adapters. We compare the student against three baselines: zero-shot mT5-small, mT5-small fine-tuned on human reference summaries, and the teacher LLMs themselves.

## Repository structure

```
.
├── PROJECT_PLAN.md             Roadmap, experiment matrix, timeline
├── README.md                   This file
├── requirements.txt            Pinned Python dependencies (Colab T4)
├── .env.example                Template for API keys
├── configs/                    YAML configs for data + training
│   └── train/                  Training presets (default, ablations)
├── src/
│   ├── data/                   MLSUM-TR + TR-News loaders, synthetic prep
│   ├── teachers/               Prompts + OpenAI/Anthropic clients + runner
│   ├── student/                mT5-small + LoRA train and infer scripts
│   ├── eval/                   ROUGE + BERTScore + qualitative analysis
│   └── utils/                  Shared helpers (io, logging, seed)
├── scripts/                    Bash entry points for each pipeline stage
├── notebooks/                  Colab notebooks wrapping the CLI scripts
├── data/                       (gitignored, except synthetic/)
│   ├── raw/                    Downloaded MLSUM-TR / TR-News splits
│   ├── processed/              Tokenized HF datasets
│   └── synthetic/              Teacher-generated summaries (committed)
├── outputs/                    (gitignored, except results/)
│   ├── checkpoints/            LoRA adapters
│   ├── predictions/            Per-system test set generations
│   ├── logs/                   Training logs + run_config.json files
│   └── results/                Final metric tables (committed)
└── report/                     LNCS LaTeX source + figures
```

## Setup

```bash
git clone <repo-url>
cd "CENG467 Term Project"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```

For Colab, open `notebooks/colab_train.ipynb` — it handles installs and mounts Drive.

## End-to-end pipeline

```bash
# 1. Download MLSUM-TR + TR-News
bash scripts/01_download_data.sh

# 2. Generate teacher summaries (cache automatic — safe to interrupt)
bash scripts/02_generate_teacher.sh openai concise 10000
bash scripts/02_generate_teacher.sh anthropic concise 10000

# 3. Train student on synthetic data
bash scripts/03_train_student.sh --teacher openai --size 10000 --lora-rank 8

# 4. Evaluate everything
bash scripts/04_evaluate.sh

# Optional: run all ablations sequentially
bash scripts/run_all_ablations.sh
```

Every stage is also runnable as `python -m src.<module> --help` for finer control.

## Reproducing the report numbers

```bash
# Re-runs evaluation only (uses cached predictions in outputs/predictions/)
python -m src.eval.run_eval --systems all --metrics rouge bertscore --out outputs/results/main.json
```

## Cost estimate

Generating both 14k-call teacher pipelines (train + test + OOD) costs **~$7 in total** with Claude 3 Haiku, GPT-4o-mini, real-time API, and 3000-char article truncation. Caching makes reruns free. See `configs/budget.yaml` and `PROJECT_PLAN.md` §4 for the line items.

## Citation

If you use this code, cite the underlying datasets:

- Scialom, T. et al. *MLSUM: The Multilingual Summarization Corpus*, EMNLP 2020.
- Baykara, B., Güngör, T. *Abstractive text summarization and new large-scale datasets for agglutinative languages: Turkish and Hungarian*, Language Resources and Evaluation, 2022.

## License

Code released under MIT. Datasets follow their original licenses.
