You are helping me build a university term project for CENG 467 Natural Language Understanding and Generation at Izmir Institute of Technology (IYTE), Spring 2026, taught by Prof. Dr. Aytug ONAN.

PROJECT: Synthetic Data Distillation from Large Language Models for Turkish Abstractive News Summarization

WHAT THE PROJECT DOES: We prompt a large teacher LLM (GPT-4o-mini or Claude) to generate abstractive summaries of Turkish news articles, then fine-tune a small student model (mT5-small, ~300M params) on those synthetic summaries using LoRA adapters. We compare against three baselines: (1) zero-shot mT5-small, (2) mT5-small fine-tuned on human reference summaries, (3) the teacher LLM itself.

DATASETS: MLSUM-TR (Scialom et al., 2020, ~250K Turkish news pairs) as primary. TR-News (Baykara & Gungor, 2022, ~280K Turkish news articles) as secondary out-of-domain evaluation.

EVALUATION: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore (multilingual encoder). Ablations over synthetic dataset size (1k, 5k, 10k), teacher prompt variants (concise vs. detailed), and LoRA rank (4, 8, 16, 32).

COMPUTE: Google Colab free tier (T4 GPU). Everything must fit in 15GB VRAM. Use LoRA adapters and careful batch-size/gradient-accumulation tuning.

DELIVERABLES:
- Fully reproducible GitHub repository with clear structure, all scripts, requirements.txt, and a README
- 6-8 page academic report in LNCS format
- Live demo

GRADING PRIORITIES (out of 100):
- Model/Method Implementation: 20 pts (biggest category, take this seriously)
- Baseline Comparison: 15 pts
- Evaluation Methodology: 15 pts
- Ablation/Sensitivity Analysis: 10 pts
- Generation Quality Analysis: 10 pts
- Presentation/Demo: 10 pts
- Problem Formulation: 10 pts
- Ethics/Bias Analysis: 5 pts
- Reproducibility/Code Quality: 5 pts

REQUIRED EXPERIMENTAL COMPONENTS (do not skip any):
- At least two baseline comparisons
- Standard NLP evaluation metrics
- Ablation or sensitivity study
- Qualitative error analysis of failure cases (hallucinations, omissions, repetition, morphological errors)
- Ethical considerations (bias, hallucination risks, misuse)

RULES FOR YOU:
- Always produce complete, runnable code. Never leave placeholder comments like "add logic here".
- All code must be Colab-compatible (T4 GPU, 15GB VRAM).
- When writing scripts, include proper argument parsing so experiments are reproducible from command line.
- Document all LLM prompts and API configurations since this is a course requirement.
- Use HuggingFace Transformers, PEFT, and datasets libraries as the core stack.
- When I ask for the report, write in LNCS format with proper academic citations.
- Keep Turkish language considerations in mind throughout (agglutinative morphology, ROUGE underestimation, tokenization challenges).
- If something requires me to run it myself (training jobs, API calls, manual annotation), tell me clearly what to do and in what order.
