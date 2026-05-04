# tune_sbert.py
"""
Optuna-based hyperparameter tuning for SBERT weight combinations.
Uses stratified 5-fold CV on the training set.

Search space:
  - w_title: 0.1 to 0.9 (continuous)
  - w_scope: 0.1 to 0.9 (continuous, constrained so w_title + w_scope <= 1.0)
  - w_cat = 1.0 - w_title - w_scope (derived, must be >= 0)
  - model: all-MiniLM-L6-v2 or all-mpnet-base-v2

Embeddings are precomputed once per model variant.

Usage:
  python tune_sbert.py --train_csv doaj_train.csv --n_trials 100
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import pickle
import optuna
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from utils import mean_hit_rate_at_k

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

K_EVAL = 1  # Optimize for HitRate@1

MODEL_NAMES = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]


def compute_embeddings(model_name, titles, scopes, batch_size=64, cache_dir="results"):
    """Compute or load cached embeddings for a model variant."""
    safe_name = model_name.replace("/", "_").replace("-", "_")
    title_cache = os.path.join(cache_dir, f"tune_{safe_name}_title_emb.npy")
    scope_cache = os.path.join(cache_dir, f"tune_{safe_name}_scope_emb.npy")

    if os.path.exists(title_cache) and os.path.exists(scope_cache):
        print(f"  Loading cached embeddings for {model_name}...")
        title_emb = np.load(title_cache)
        scope_emb = np.load(scope_cache)
        return title_emb, scope_emb

    print(f"  Computing embeddings for {model_name} (this takes ~10-15 min with GPU)...")
    model = SentenceTransformer(model_name)

    title_emb = model.encode(titles, batch_size=batch_size, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True)
    scope_emb = model.encode(scopes, batch_size=batch_size, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(title_cache, title_emb)
    np.save(scope_cache, scope_emb)
    print(f"  Cached embeddings to {cache_dir}")
    return title_emb, scope_emb


def compute_category_centroids(scope_emb, cats):
    """Compute mean scope embedding per category (centroid)."""
    unique_cats = sorted(set(cats))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    centroids = np.zeros((len(unique_cats), scope_emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_cats), dtype=int)

    for i, cat in enumerate(cats):
        idx = cat_to_idx[cat]
        centroids[idx] += scope_emb[i]
        counts[idx] += 1

    # Average
    for i in range(len(unique_cats)):
        if counts[i] > 0:
            centroids[i] /= counts[i]

    return centroids, unique_cats, cat_to_idx


def evaluate_sbert_fold(train_title_emb, train_scope_emb, train_cats,
                        val_title_emb, val_scope_emb, val_cats,
                        w_title, w_scope, w_cat):
    """Score val queries against train docs using weighted SBERT similarity."""
    # Compute category centroids from training fold
    centroids, unique_cats_list, cat_to_idx = compute_category_centroids(
        train_scope_emb, train_cats
    )

    train_cats_arr = np.array(train_cats)
    all_ranked_cats = []

    # Process in batches
    batch_size = 200
    for start in range(0, len(val_cats), batch_size):
        end = min(start + batch_size, len(val_cats))

        # Title similarity
        sim_title = cosine_similarity(val_title_emb[start:end], train_title_emb)
        # Scope similarity
        sim_scope = cosine_similarity(val_scope_emb[start:end], train_scope_emb)

        # Category centroid similarity (query scope vs all category centroids)
        if w_cat > 0 and len(centroids) > 0:
            sim_q_to_centroids = cosine_similarity(val_scope_emb[start:end], centroids)  # (batch, n_cats)
            # Map each training doc to its category centroid sim
            train_cat_indices = np.array([cat_to_idx.get(c, 0) for c in train_cats])
            sim_cat = sim_q_to_centroids[:, train_cat_indices]  # (batch, n_train)
        else:
            sim_cat = np.zeros_like(sim_title)

        # Combined score
        combined = w_title * sim_title + w_scope * sim_scope + w_cat * sim_cat

        # Get top-K
        top_indices = np.argsort(combined, axis=1)[:, ::-1][:, :K_EVAL]
        for row_topk in top_indices:
            all_ranked_cats.append(train_cats_arr[row_topk].tolist())

    return mean_hit_rate_at_k(all_ranked_cats, list(val_cats), K_EVAL)


def objective(trial, embeddings_cache, cats, n_folds=5):
    """Optuna objective: mean HitRate@1 across stratified K-fold CV."""
    model_name = trial.suggest_categorical("model", MODEL_NAMES)
    w_title = trial.suggest_float("w_title", 0.1, 0.8)
    w_scope = trial.suggest_float("w_scope", 0.1, 0.8)

    # Constrain: w_title + w_scope <= 1.0, w_cat >= 0
    w_cat = 1.0 - w_title - w_scope
    if w_cat < 0:
        return 0.0  # Invalid combination, penalize

    title_emb, scope_emb = embeddings_cache[model_name]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    indices = np.arange(len(cats))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, cats)):
        train_title = title_emb[train_idx]
        train_scope = scope_emb[train_idx]
        train_cats = [cats[i] for i in train_idx]
        val_title = title_emb[val_idx]
        val_scope = scope_emb[val_idx]
        val_cats = [cats[i] for i in val_idx]

        score = evaluate_sbert_fold(
            train_title, train_scope, train_cats,
            val_title, val_scope, val_cats,
            w_title, w_scope, w_cat
        )
        fold_scores.append(score)

    return np.mean(fold_scores)


def main():
    parser = argparse.ArgumentParser(description="Tune SBERT weights with Optuna")
    parser.add_argument("--train_csv", default="doaj_train.csv", help="Training CSV path")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--out_dir", default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(args.train_csv, dtype=str, low_memory=False).fillna("")
    titles = train_df["title"].tolist()
    # Scope = subjects + keywords (the 'text' field minus the title, or use text_raw)
    scopes = train_df["text_raw"].tolist()
    cats = train_df["__primary_cat__"].tolist()

    # Filter out empty
    valid_mask = [bool(t.strip() and s.strip() and c.strip()) for t, s, c in zip(titles, scopes, cats)]
    titles = [t for t, v in zip(titles, valid_mask) if v]
    scopes = [s for s, v in zip(scopes, valid_mask) if v]
    cats = [c for c, v in zip(cats, valid_mask) if v]
    print(f"  {len(titles)} valid samples, {len(set(cats))} categories")

    # Precompute embeddings for both model variants
    embeddings_cache = {}
    for model_name in MODEL_NAMES:
        print(f"\nPreparing embeddings for {model_name}...")
        title_emb, scope_emb = compute_embeddings(
            model_name, titles, scopes, batch_size=args.batch_size, cache_dir=args.out_dir
        )
        embeddings_cache[model_name] = (title_emb, scope_emb)

    # Run Optuna study
    print(f"\nStarting Optuna SBERT tuning ({args.n_trials} trials, {args.n_folds}-fold CV)...")
    start_time = time.time()

    study = optuna.create_study(
        direction="maximize",
        study_name="sbert_tuning",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    def print_progress(study, trial):
        w_cat = 1.0 - trial.params['w_title'] - trial.params['w_scope']
        print(f"  Trial {trial.number:3d} | HitRate@1 = {trial.value:.4f} | "
              f"model={trial.params['model']}, "
              f"w=({trial.params['w_title']:.2f}, {trial.params['w_scope']:.2f}, {w_cat:.2f})")

    study.optimize(
        lambda trial: objective(trial, embeddings_cache, cats, args.n_folds),
        n_trials=args.n_trials,
        callbacks=[print_progress]
    )

    elapsed = time.time() - start_time

    # Results
    best = study.best_trial
    w_cat_best = 1.0 - best.params['w_title'] - best.params['w_scope']
    print(f"\n{'='*60}")
    print(f"Best SBERT HitRate@1: {best.value:.4f}")
    print(f"Best params: model={best.params['model']}, "
          f"w_title={best.params['w_title']:.4f}, "
          f"w_scope={best.params['w_scope']:.4f}, "
          f"w_cat={w_cat_best:.4f}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Save results
    result = {
        "model": "sbert",
        "best_hitrate_at_1": float(best.value),
        "best_params": {
            "model_name": best.params["model"],
            "w_title": round(best.params["w_title"], 6),
            "w_scope": round(best.params["w_scope"], 6),
            "w_cat": round(w_cat_best, 6),
        },
        "n_trials": args.n_trials,
        "n_folds": args.n_folds,
        "elapsed_seconds": round(elapsed, 1),
        "all_trials": [
            {
                "number": t.number, "value": t.value,
                "params": {
                    "model_name": t.params.get("model", ""),
                    "w_title": round(t.params.get("w_title", 0), 6),
                    "w_scope": round(t.params.get("w_scope", 0), 6),
                    "w_cat": round(1.0 - t.params.get("w_title", 0) - t.params.get("w_scope", 0), 6),
                }
            }
            for t in study.trials
        ]
    }
    out_path = os.path.join(args.out_dir, "sbert_best_params.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")

    # Save Optuna visualizations
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(args.out_dir, "sbert_optimization_history.png"))
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(args.out_dir, "sbert_param_importance.png"))
        print("Saved Optuna plots")
    except Exception as e:
        print(f"Could not save Optuna plots (install plotly/kaleido for plots): {e}")


if __name__ == "__main__":
    main()
