# tune_bm25.py
"""
Optuna-based hyperparameter tuning for BM25 retrieval.
Uses stratified 3-fold CV on the training set (3-fold for speed).

Search space:
  - k1: 0.5 to 3.0 (continuous)
  - b:  0.0 to 1.0 (continuous)

Usage:
  python tune_bm25.py --train_csv doaj_train.csv --n_trials 30
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import optuna
from rank_bm25 import BM25Okapi
from sklearn.model_selection import StratifiedKFold
from utils import mean_hit_rate_at_k

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

K_EVAL = 1  # Optimize for HitRate@1


def tokenize_texts(texts):
    """Simple whitespace tokenization (text is already cleaned by nltk_clean)."""
    return [t.split() for t in texts]


def evaluate_bm25_fold(train_tokens, train_cats, val_tokens, val_cats, k1, b):
    """Build BM25 on train fold, score val queries, return HitRate@1."""
    bm25 = BM25Okapi(train_tokens, k1=k1, b=b)
    train_cats_arr = np.array(train_cats)

    all_ranked_cats = []
    for q_tokens in val_tokens:
        scores = bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:K_EVAL]
        ranked_cats = train_cats_arr[top_idx].tolist()
        all_ranked_cats.append(ranked_cats)

    return mean_hit_rate_at_k(all_ranked_cats, list(val_cats), K_EVAL)


def objective(trial, all_tokens, cats, n_folds=3):
    """Optuna objective: mean HitRate@1 across stratified K-fold CV."""
    k1 = trial.suggest_float("k1", 0.5, 3.0)
    b = trial.suggest_float("b", 0.0, 1.0)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []

    # Use integer indices for splitting
    indices = np.arange(len(all_tokens))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, cats)):
        train_tokens = [all_tokens[i] for i in train_idx]
        train_cats = [cats[i] for i in train_idx]
        val_tokens = [all_tokens[i] for i in val_idx]
        val_cats = [cats[i] for i in val_idx]

        score = evaluate_bm25_fold(train_tokens, train_cats, val_tokens, val_cats, k1, b)
        fold_scores.append(score)

    return np.mean(fold_scores)


def main():
    parser = argparse.ArgumentParser(description="Tune BM25 hyperparameters with Optuna")
    parser.add_argument("--train_csv", default="doaj_train.csv", help="Training CSV path")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--n_folds", type=int, default=3, help="Number of CV folds")
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

    # Pre-tokenize all texts (done once, reused across trials)
    print("Pre-tokenizing texts...")
    all_tokens = tokenize_texts(texts)
    print(f"  Done. Avg tokens per doc: {np.mean([len(t) for t in all_tokens]):.1f}")

    # Run Optuna study
    print(f"\nStarting Optuna BM25 tuning ({args.n_trials} trials, {args.n_folds}-fold CV)...")
    print("NOTE: BM25 is CPU-bound and slow. This may take several hours.\n")
    start_time = time.time()

    study = optuna.create_study(
        direction="maximize",
        study_name="bm25_tuning",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    def print_progress(study, trial):
        elapsed = time.time() - start_time
        print(f"  Trial {trial.number:3d} | HitRate@1 = {trial.value:.4f} | "
              f"k1={trial.params['k1']:.3f}, b={trial.params['b']:.3f} | "
              f"elapsed: {elapsed/60:.1f} min")

    study.optimize(
        lambda trial: objective(trial, all_tokens, cats, args.n_folds),
        n_trials=args.n_trials,
        callbacks=[print_progress]
    )

    elapsed = time.time() - start_time

    # Results
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best BM25 HitRate@1: {best.value:.4f}")
    print(f"Best params: k1={best.params['k1']:.4f}, b={best.params['b']:.4f}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Save results
    result = {
        "model": "bm25",
        "best_hitrate_at_1": float(best.value),
        "best_params": {k: round(v, 6) for k, v in best.params.items()},
        "n_trials": args.n_trials,
        "n_folds": args.n_folds,
        "elapsed_seconds": round(elapsed, 1),
        "all_trials": [
            {"number": t.number, "value": t.value, "params": {k: round(v, 6) for k, v in t.params.items()}}
            for t in study.trials
        ]
    }
    out_path = os.path.join(args.out_dir, "bm25_best_params.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")

    # Save Optuna visualizations
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(args.out_dir, "bm25_optimization_history.png"))
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(args.out_dir, "bm25_param_importance.png"))
        fig3 = plot_contour(study, params=["k1", "b"])
        fig3.write_image(os.path.join(args.out_dir, "bm25_contour_k1_b.png"))
        print("Saved Optuna plots")
    except Exception as e:
        print(f"Could not save Optuna plots (install plotly/kaleido for plots): {e}")


if __name__ == "__main__":
    main()
