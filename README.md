# Multiclass Misinformation Detection

This repository contains experiments on **3-class misinformation detection** (`false`, `true`, `unknown`) using multiple datasets from the [**ComplexDataLab/Misinfo_Datasets**](https://huggingface.co/datasets/ComplexDataLab/Misinfo_Datasets) collection. The project compares **lexical**, **embedding-based**, and **transformer-based** models under **claim-only** and **claim+evidence** settings, and evaluates **multilingual and cross-lingual generalization**.

## Objectives
- Compare three modeling families for multiclass misinformation detection.
- Analyze the effect of **evidence concatenation** versus **claim-only** inputs.
- Evaluate **monolingual and zero-shot cross-lingual transfer** on X-FACT.
- Identify dataset artifacts via **keyword correlation analysis** (EQA subset).

## Datasets
- **FEVER (EN)** - claim-only verification
- **Climate-FEVER (EN)** - claim + human-verified evidence
- **X-FACT (25 languages)** - claim-only and claim+evidence, multilingual evaluation

## Models
- **TF–IDF + Complement Naive Bayes (C-NB)** - strong lexical baseline
- **FastText embeddings + MLP** - lightweight semantic model
- **Fine-tuned XLM-RoBERTa (XLM-R)** - multilingual transformer

## Key Observations
- Claim-only verification remains challenging across datasets.
- Curated evidence substantially improves performance, while noisy evidence yields smaller gains.
- Lexical models degrade under cross-lingual transfer, whereas XLM-R retains useful performance.
- Keyword correlation analysis indicates that some datasets contain lexical shortcuts, partially explaining strong baseline results.

## Notebooks
- `Fever.ipynb` — FEVER claim-only experiments
- `climate_fever_claim_evidence.ipynb` - Climate-FEVER claim+evidence experiments
- `x_fact_claim_only.ipynb` - X-FACT claim-only (all languages)
- `x_fact_claim_evidence.ipynb` - X-FACT claim+evidence (all languages)
- `x_fact_crosslingual.ipynb` - monolingual (EN/ES/PL) and zero-shot EN→ES/PL evaluation
- `keyword_analysis.ipynb` - keyword correlation analysis inspired by  
  [*A Guide to Misinformation Detection Data and Evaluation*](https://arxiv.org/pdf/2411.05060v2)

## Evaluation Metrics
- **Macro-F1** (primary metric for class imbalance)
- **Macro/Micro AUPRC**
- **Expected Calibration Error (ECE)**
- Confusion matrices and focused error analysis

## Execution
Each notebook is self-contained and designed to run in **Google Colab**. 
