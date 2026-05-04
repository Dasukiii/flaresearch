# tune_tfidf.py
"""
Optuna-based hyperparameter tuning for TF-IDF retrieval.
Uses stratified 5-fold CV on the training set to find optimal params.

Search space:
  - ngram_range: (1, max) where max in {1, 2, 3}
  - max_features: 3000 to 30000
  - sublinear_tf: True/False
  - min_df: 1 to 5

Usage:
  python tune_tfidf.py --train_csv doaj_train.csv --n_trials 50
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from utils import mean_hit_rate_at_k

# Suppress Optuna info logs (show only warnings+)
optuna.logging.set_verbosity(optuna.logging.WARNING)

K_EVAL = 1  # Optimize for HitRate@1 (accuracy)


def evaluate_tfidf_fold(train_texts, train_cats, val_texts, val_cats,
                        ngram_max, max_features, sublinear_tf, min_df):
    """Build TF-IDF on train fold, score val queries, return HitRate@1."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        dtype=np.float32
    )

    # Fit on training docs
    train_matrix = vectorizer.fit_transform(train_texts)
    val_matrix = vectorizer.transform(val_texts)

    # Compute cosine similarity: val queries vs train docs
    # Process in batches to save memory
    batch_size = 200
    all_ranked_cats = []
    train_cats_arr = np.array(train_cats)

    for start in range(0, len(val_texts), batch_size):
        end = min(start + batch_size, len(val_texts))
        sims = cosine_similarity(val_matrix[start:end], train_matrix)  # (batch, n_train)
        # For each query, get top-K indices
        top_indices = np.argsort(sims, axis=1)[:, ::-1][:, :K_EVAL]
        for row_topk in top_indices:
            ranked_cats = train_cats_arr[row_topk].tolist()
            all_ranked_cats.append(ranked_cats)

    val_true_cats = list(val_cats)
    return mean_hit_rate_at_k(all_ranked_cats, val_true_cats, K_EVAL)


def objective(trial, texts, cats, n_folds=5):
    """Optuna objective: mean HitRate@1 across stratified K-fold CV."""
    ngram_max = trial.suggest_int("ngram_max", 1, 3)
    max_features = trial.suggest_int("max_features", 3000, 30000, step=1000)
    sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])
    min_df = trial.suggest_int("min_df", 1, 5)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, cats)):
        train_texts = [texts[i] for i in train_idx]
        train_cats = [cats[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_cats = [cats[i] for i in val_idx]

        score = evaluate_tfidf_fold(
            train_texts, train_cats, val_texts, val_cats,
            ngram_max, max_features, sublinear_tf, min_df
        )
        fold_scores.append(score)

    mean_score = np.mean(fold_scores)
    return mean_score


def main():
    parser = argparse.ArgumentParser(description="Tune TF-IDF hyperparameters with Optuna")
    parser.add_argument("--train_csv", default="doaj_train.csv", help="Training CSV path")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--out_dir", default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(args.train_csv, dtype=str, low_memory=False).fillna("")
    texts = train_df["text"].tolist()
    cats = train_df["__primary_cat__"].tolist()

    # Filter out empty texts/cats
    valid = [(t, c) for t, c in zip(texts, cats) if t.strip() and c.strip()]
    texts = [v[0] for v in valid]
    cats = [v[1] for v in valid]
    print(f"  {len(texts)} valid samples, {len(set(cats))} categories")

    # Run Optuna study
    print(f"\nStarting Optuna TF-IDF tuning ({args.n_trials} trials, {args.n_folds}-fold CV)...")
    start_time = time.time()

    study = optuna.create_study(
        direction="maximize",
        study_name="tfidf_tuning",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Custom callback to print progress
    def print_progress(study, trial):
        print(f"  Trial {trial.number:3d} | HitRate@1 = {trial.value:.4f} | "
              f"ngram_max={trial.params['ngram_max']}, "
              f"max_features={trial.params['max_features']}, "
              f"sublinear_tf={trial.params['sublinear_tf']}, "
              f"min_df={trial.params['min_df']}")

    study.optimize(
        lambda trial: objective(trial, texts, cats, args.n_folds),
        n_trials=args.n_trials,
        callbacks=[print_progress]
    )

    elapsed = time.time() - start_time

    # Results
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best TF-IDF HitRate@1: {best.value:.4f}")
    print(f"Best params: {best.params}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Save results
    result = {
        "model": "tfidf",
        "best_hitrate_at_1": float(best.value),
        "best_params": best.params,
        "n_trials": args.n_trials,
        "n_folds": args.n_folds,
        "elapsed_seconds": round(elapsed, 1),
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ]
    }
    out_path = os.path.join(args.out_dir, "tfidf_best_params.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")

    # Save Optuna visualizations
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(args.out_dir, "tfidf_optimization_history.png"))
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(args.out_dir, "tfidf_param_importance.png"))
        print("Saved Optuna plots")
    except Exception as e:
        print(f"Could not save Optuna plots (install plotly/kaleido for plots): {e}")


if __name__ == "__main__":
    main()
