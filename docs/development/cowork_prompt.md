# CENG 467 Term Project: Full Context for Claude Cowork

## WHO I AM
I am a student at Izmir Institute of Technology (IYTE), taking CENG 467 Natural Language Understanding and Generation (Spring 2026, Prof. Dr. Aytug ONAN). I am working in a group of up to three members. All premade project topics from the course were already taken, so we proposed a custom topic which needs instructor approval.

## PROJECT TITLE
Synthetic Data Distillation from Large Language Models for Turkish Abstractive News Summarization

## PROBLEM DESCRIPTION
Abstractive summarization in Turkish lags behind English due to limited high-quality training data and the morphological complexity of the language. Large proprietary LLMs (GPT-4, Claude, Gemini) produce fluent Turkish summaries but are expensive, closed-source, and impractical to deploy at scale. Small open models such as mT5-small and mBART are deployable but perform poorly zero-shot on Turkish. This project investigates whether a small student model can inherit the summarization capability of a large teacher LLM through synthetic data distillation, using teacher-generated summaries as training signal instead of human references. The goal is to produce a compact, deployable Turkish summarizer that approaches teacher quality at a fraction of the inference cost.

## PROPOSED APPROACH
We will construct a synthetic training corpus by prompting a large teacher LLM (e.g., GPT-4o-mini or Claude) to generate abstractive summaries for Turkish news articles from the TR-News or MLSUM-TR dataset. A small student model (mT5-small, approximately 300M parameters) will then be fine-tuned on these teacher-generated (article, summary) pairs using standard sequence-to-sequence objectives. We will additionally explore (i) the effect of teacher prompt design on student quality, and (ii) the effect of synthetic-data scale (1k, 5k, 10k examples) on final performance. Training will be performed on Google Colab using LoRA adapters for memory efficiency.

## BASELINE METHODS
1. Zero-shot student: mT5-small applied directly without fine-tuning.
2. Reference-fine-tuned student: the same mT5-small fine-tuned on human-written reference summaries from MLSUM-TR, serving as the upper bound for supervised small-model performance.
3. Zero-shot teacher: the large LLM applied directly, serving as the performance ceiling.

Comparing against all three lets us isolate the value added by distillation specifically.

## DATASETS
- MLSUM-TR (Scialom et al., 2020): approximately 250K Turkish news article and summary pairs from Internet Haber. We will use a subset of around 10K articles for synthetic data generation and the standard test split (around 11K) for evaluation.
- TR-News (Baykara & Gungor, 2022): approximately 280K Turkish news articles with summaries, used as a secondary out-of-domain evaluation set to test generalization.

## EVALUATION STRATEGY
- Automatic metrics: ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore using a multilingual encoder (better suited to Turkish morphology than n-gram overlap alone).
- Ablation study: student performance as a function of (a) synthetic dataset size, (b) teacher prompt variants (concise vs. detailed), and (c) LoRA rank.
- Error analysis: manual inspection of around 50 generated summaries, categorizing failures into hallucination, omission, repetition, and morphological errors (incorrect agglutinative suffixes).
- Ethical considerations: discussion of hallucination risks, teacher-model bias inheritance, and licensing implications of training on LLM-generated data.

## EXPECTED CHALLENGES
- Teacher hallucinations propagating to the student: if the teacher fabricates facts, the student will learn to fabricate them. We will mitigate this with light filtering based on ROUGE overlap with the source article.
- Turkish morphological evaluation: ROUGE is known to underestimate quality in morphologically rich languages, so we include BERTScore to compensate.
- Compute constraints: fine-tuning on Colab free tier requires careful batch-size and gradient-accumulation tuning; LoRA adapters keep memory tractable.
- API cost for synthetic data generation: will be controlled by capping the synthetic corpus size at 10K examples and using a cost-efficient teacher model.

## COURSE REQUIREMENTS (DO NOT SKIP ANY OF THESE)

### Deliverables
- 6-8 page academic-style report in LNCS format
- Fully reproducible GitHub repository
- Short live demonstration of the system

### GitHub Repository Must Include
- Clear project structure
- Training and evaluation scripts
- Dataset preparation instructions
- Dependency file (requirements.txt or environment.yml)
- README explaining how results can be reproduced
- Since we use LLM APIs, all prompts and configuration settings must be documented

### Report Structure (mandatory sections)
1. Introduction
2. Related Work
3. Methodology
4. Experimental Setup
5. Results
6. Error Analysis
7. Discussion
8. Conclusion
- Proper citation of datasets, libraries, and research papers is mandatory.

### Required Experimental Components
- Baseline Comparison: at least two baseline models or prompting strategies
- Evaluation Metrics: BLEU, ROUGE, METEOR, BERTScore, Exact Match, F1-score, or perplexity (as appropriate)
- Ablation or Sensitivity Study: how architectural components, prompts, or hyperparameters affect model performance
- Error Analysis: qualitative analysis of failure cases (common mistakes, hallucinations, linguistic errors)
- Ethical Considerations: bias, hallucination risks, misuse scenarios

### Grading Rubric (100 points total)
- Problem Formulation and Task Understanding: 10 pts
- Baseline or Prompting Comparison: 15 pts
- Model / Method Implementation: 20 pts (BIGGEST CATEGORY)
- Evaluation Methodology: 15 pts
- Ablation or Prompt Sensitivity Analysis: 10 pts
- Generation Quality Analysis: 10 pts
- Ethical Considerations and Bias Analysis: 5 pts
- Reproducibility and Code Quality: 5 pts
- Presentation and Demonstration: 10 pts
- Bonus (+5 pts): additional advanced model, reproducing results from a recent paper, or additional experiments

## TECHNICAL DECISIONS AND CONSTRAINTS

### What We Chose and Why
- Student model: mT5-small (~300M params) because it fits Colab free tier T4 GPU
- Fine-tuning method: LoRA adapters for memory efficiency
- Teacher model: GPT-4o-mini or Claude API (cost-efficient, good Turkish output)
- Primary dataset: MLSUM-TR (well-established, large Turkish summarization dataset)
- Secondary dataset: TR-News (out-of-domain generalization test)
- Synthetic data sizes to test: 1k, 5k, 10k examples
- Teacher prompt variants to test: concise vs. detailed instruction prompts
- LoRA rank to ablate over: multiple values (e.g., 4, 8, 16, 32)

### Compute Environment
- Google Colab free tier (T4 GPU, 15GB VRAM)
- Must be careful with batch sizes and gradient accumulation
- LoRA keeps memory tractable

### Things That Must Be Done By Humans (Not Claude)
- Actually running training jobs on Colab
- Downloading datasets and managing HuggingFace tokens
- Running LLM API calls (requires API keys)
- Manual error analysis of 50 generated summaries
- Live demo presentation

### Things Claude Can Help With
- All code (data preprocessing, synthetic data generation scripts, training scripts, evaluation scripts)
- Literature review and related work
- Report writing in LNCS format
- Experimental design and ablation planning
- Error analysis framework and categorization
- GitHub repo structure and README
- Visualization of results (tables, charts)
- Debugging

## PROJECT TIMELINE
- Week 7: Group formation + topic selection + proposal submission (DONE)
- Week 8-10: Literature review and initial development
- Week 11: Technical checkpoint (working baselines + initial results)
- Week 12-14: Method development, experiments, and paper writing
- Week 15: Final submission (LNCS paper + system)
- Week 15-16: Project presentations and demo

## WHAT I NEED YOU TO DO NOW
Start by setting up the complete project structure as a GitHub repository. Then work through each component systematically:

1. First, create the full repo structure (folders, README, requirements.txt, .gitignore)
2. Write the data preparation scripts (downloading MLSUM-TR, preprocessing)
3. Write the synthetic data generation script (teacher LLM prompting pipeline)
4. Write the training script (mT5-small fine-tuning with LoRA on synthetic data)
5. Write the baseline scripts (zero-shot student, reference-fine-tuned student, zero-shot teacher)
6. Write the evaluation pipeline (ROUGE, BERTScore computation)
7. Write the ablation experiment runner
8. Create result visualization scripts

Do each step completely before moving to the next. Ask me if you need any clarification, but do not skip any detail listed above.
