# FlareSearch — A Tool for Aligning Manuscripts with Suitable Journals

**Author:** Chin Hong An  
**Supervisor:** Ts Dr Ku Chin Soon  
**Institution:** Universiti Tunku Abdul Rahman (UTAR), Faculty of Information and Communication Technology  
**Degree:** Bachelor of Computer Science (Honours)  

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flaresearch.streamlit.app)

---

## Overview

FlareSearch is a journal recommendation system that helps researchers find suitable journals for their manuscripts. It compares three retrieval models — **TF-IDF**, **Okapi BM25**, and **Sentence-BERT** — with hyperparameters tuned using Bayesian optimisation (Optuna).

The best-performing model (**TF-IDF with Title+Scope+Category**, HitRate@1 = **88.84%**) is deployed as the backend of the FlareSearch web application.

### Key Results

| Model | HitRate@1 | MAP@1 | Execution Time |
|-------|-----------|-------|----------------|
| **TF-IDF** | **88.84%** | **88.84%** | **4.08s** |
| BM25 | 83.47% | 83.47% | 268.47s |
| SBERT | 68.53% | 68.53% | 125.13s |

*All results use the Title+Scope+Category input strategy on the held-out test set (4,356 queries).*

---

## Dataset

- **Source:** [Directory of Open Access Journals (DOAJ)](https://doaj.org/)
- **Size:** 21,782 journals across 327 categories
- **Split:** 80/20 stratified (17,426 train / 4,356 test), random_state=42

---

## Project Structure

```
.
├── app.py                          # Streamlit web application (TF-IDF engine)
├── utils.py                        # Shared utilities (nltk_clean, metrics)
├── preprocess.py                   # Data cleaning, normalisation, train/test split
├── tune_tfidf.py                   # Optuna tuning for TF-IDF (50 trials, 5-fold CV)
├── tune_bm25.py                    # Optuna tuning for BM25 (30 trials, 3-fold CV)
├── tune_sbert.py                   # Optuna tuning for SBERT (100 trials, 5-fold CV)
├── run_evaluation.py               # Final evaluation: 3 models × 3 strategies
├── eval_equal_distribution.py      # Balanced evaluation (20 train, 5 test per category)
├── eval_per_category.py            # Per-category HitRate@1 breakdown
├── requirements.txt                # Deployment dependencies (app only)
├── packages.txt                    # System dependencies for Streamlit Cloud
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── journalcsv__doaj.csv            # Raw DOAJ dataset
├── doaj_normalized.csv             # Preprocessed full dataset
├── doaj_train.csv                  # Training set (17,426 journals)
├── doaj_test.csv                   # Test set (4,356 journals)
└── results/
    ├── tfidf_best_params.json      # Best TF-IDF hyperparameters
    ├── bm25_best_params.json       # Best BM25 hyperparameters
    ├── sbert_best_params.json      # Best SBERT hyperparameters
    ├── final_metrics.json          # Full evaluation results
    └── comparison_table.csv        # Formatted comparison table
```

---

## Setup & Installation

### Requirements

- Python 3.10+
- pip

### Local Development (Full Pipeline)

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install ALL dependencies (for tuning + evaluation + app)
pip install pandas numpy scikit-learn nltk sentence-transformers streamlit matplotlib scipy rank-bm25 optuna

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### App Only (Streamlit)

```bash
pip install pandas numpy scikit-learn nltk streamlit
```

---

## How to Run

### 1. Data Preprocessing

```bash
python preprocess.py --infile journalcsv__doaj.csv --out_prefix doaj
# Outputs: doaj_train.csv, doaj_test.csv, doaj_normalized.csv
```

### 2. Hyperparameter Tuning

```bash
python tune_tfidf.py    # ~9 minutes, 50 trials
python tune_bm25.py     # ~3.1 hours, 30 trials
python tune_sbert.py    # ~17 minutes, 100 trials (GPU recommended)
```

### 3. Final Evaluation

```bash
python run_evaluation.py
# Outputs: results/final_metrics.json, results/comparison_table.csv
```

### 4. Additional Experiments

```bash
python eval_equal_distribution.py   # Balanced category evaluation
python eval_per_category.py         # Per-category breakdown
```

### 5. Launch Web Application

```bash
streamlit run app.py
```

---

## Best Hyperparameters (Optuna-Tuned)

| Model | Parameter | Value |
|-------|-----------|-------|
| TF-IDF | ngram_max | 3 |
| | max_features | 3,000 |
| | sublinear_tf | True |
| | min_df | 5 |
| BM25 | k1 | 2.2247 |
| | b | 0.9990 |
| SBERT | model | all-MiniLM-L6-v2 |
| | w_title | 0.1509 |
| | w_scope | 0.1246 |
| | w_cat | 0.7244 |

---

## Evaluation Metrics

- **HitRate@K** — Proportion of queries where a correct-category journal appears in the top-K
- **MAP@K** — Mean Average Precision at K (rewards higher-ranked correct results)
- **NDCG@K** — Normalised Discounted Cumulative Gain at K

K values: {1, 3, 5, 10}

---

## Deployment

The web application is deployed on **Streamlit Community Cloud** and accessible at:

🔗 **[flaresearch.streamlit.app](https://flaresearch.streamlit.app)**

Only 7 files are required for deployment (see `.gitignore` for excluded files).

---

## License

© 2026 Chin Hong An. All rights reserved.  
Submitted in partial fulfillment of the degree of Bachelor of Computer Science (Honours) at UTAR.