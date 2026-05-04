# eval_equal_distribution.py
"""
Equal distribution experiment (B!SON Eval Setup 2).
Samples the same number of journals per category to remove popularity bias.

Usage:
  python eval_equal_distribution.py --train_csv doaj_train.csv --test_csv doaj_test.csv
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from utils import mean_hit_rate_at_k, map_at_k, ndcg_at_k

K_LIST = [1, 3, 5, 10]
RANDOM_STATE = 42


def load_best_params(results_dir="results"):
    """Load best hyperparameters from tuning JSONs."""
    params = {}
    for model in ["tfidf", "bm25", "sbert"]:
        path = os.path.join(results_dir, f"{model}_best_params.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            params[model] = data["best_params"]
        else:
            if model == "tfidf":
                params[model] = {"ngram_max": 2, "max_features": 10000, "sublinear_tf": True, "min_df": 2}
            elif model == "bm25":
                params[model] = {"k1": 1.5, "b": 0.75}
            elif model == "sbert":
                params[model] = {"model_name": "all-MiniLM-L6-v2", "w_title": 0.6, "w_scope": 0.3, "w_cat": 0.1}
    return params


def create_equal_distribution(train_df, test_df, n_train_per_cat, n_test_per_cat):
    """Create balanced train/test sets with equal journals per category."""
    train_cats = train_df["__primary_cat__"].value_counts()
    test_cats = test_df["__primary_cat__"].value_counts()

    # Find categories with enough members in both train and test
    qualifying = []
    for cat in train_cats.index:
        if cat in test_cats.index:
            if train_cats[cat] >= n_train_per_cat and test_cats[cat] >= n_test_per_cat:
                qualifying.append(cat)

    print(f"  Categories with >= {n_train_per_cat} train & >= {n_test_per_cat} test: "
          f"{len(qualifying)} / {len(train_cats)}")

    # Sample equally from each qualifying category
    rng = np.random.RandomState(RANDOM_STATE)
    balanced_train_indices = []
    balanced_test_indices = []

    for cat in sorted(qualifying):
        train_idx = train_df[train_df["__primary_cat__"] == cat].index.tolist()
        test_idx = test_df[test_df["__primary_cat__"] == cat].index.tolist()

        sampled_train = rng.choice(train_idx, size=n_train_per_cat, replace=False)
        sampled_test = rng.choice(test_idx, size=n_test_per_cat, replace=False)

        balanced_train_indices.extend(sampled_train)
        balanced_test_indices.extend(sampled_test)

    balanced_train = train_df.loc[balanced_train_indices].reset_index(drop=True)
    balanced_test = test_df.loc[balanced_test_indices].reset_index(drop=True)

    print(f"  Balanced train: {len(balanced_train)} journals ({len(qualifying)} cats × {n_train_per_cat})")
    print(f"  Balanced test:  {len(balanced_test)} journals ({len(qualifying)} cats × {n_test_per_cat})")

    return balanced_train, balanced_test, qualifying


def build_query_text(row, use_title=True, use_scope=True, use_cat=False):
    """Build query text based on input strategy."""
    parts = []
    if use_title:
        parts.append(str(row.get("title_clean", row.get("title", ""))))
    if use_scope:
        parts.append(str(row.get("subjects_clean", "")))
        parts.append(str(row.get("keywords_clean", "")))
    if use_cat:
        cat = str(row.get("__primary_cat__", ""))
        if cat.strip():
            parts.append(cat)
    return " ".join(p for p in parts if p.strip())


def evaluate_ranked_results(ranked_cats_per_query, true_cats, k_list):
    """Compute all metrics."""
    metrics = {}
    for k in k_list:
        metrics[f"HitRate@{k}"] = mean_hit_rate_at_k(ranked_cats_per_query, true_cats, k)
        metrics[f"MAP@{k}"] = map_at_k(ranked_cats_per_query, true_cats, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(ranked_cats_per_query, true_cats, k)
    return metrics


def run_tfidf(train_df, test_df, params, k_list):
    train_texts = [build_query_text(row, use_cat=True) for _, row in train_df.iterrows()]
    test_texts = [build_query_text(row, use_cat=True) for _, row in test_df.iterrows()]
    train_cats = train_df["__primary_cat__"].tolist()
    test_cats = test_df["__primary_cat__"].tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, params.get("ngram_max", 2)),
        max_features=params.get("max_features", 10000),
        sublinear_tf=params.get("sublinear_tf", True),
        min_df=params.get("min_df", 2),
        dtype=np.float32
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)

    max_k = max(k_list)
    train_cats_arr = np.array(train_cats)
    all_ranked_cats = []

    batch_size = 200
    for start in range(0, len(test_texts), batch_size):
        end = min(start + batch_size, len(test_texts))
        sims = cosine_similarity(test_matrix[start:end], train_matrix)
        top_indices = np.argsort(sims, axis=1)[:, ::-1][:, :max_k]
        for row_topk in top_indices:
            all_ranked_cats.append(train_cats_arr[row_topk].tolist())

    return evaluate_ranked_results(all_ranked_cats, test_cats, k_list)


def run_bm25(train_df, test_df, params, k_list):
    train_texts = [build_query_text(row, use_cat=True) for _, row in train_df.iterrows()]
    test_texts = [build_query_text(row, use_cat=True) for _, row in test_df.iterrows()]
    train_cats = train_df["__primary_cat__"].tolist()
    test_cats = test_df["__primary_cat__"].tolist()

    train_tokens = [t.split() for t in train_texts]
    test_tokens = [t.split() for t in test_texts]
    bm25 = BM25Okapi(train_tokens, k1=params.get("k1", 1.5), b=params.get("b", 0.75))

    max_k = max(k_list)
    train_cats_arr = np.array(train_cats)
    all_ranked_cats = []

    for q_tokens in test_tokens:
        scores = bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:max_k]
        all_ranked_cats.append(train_cats_arr[top_idx].tolist())

    return evaluate_ranked_results(all_ranked_cats, test_cats, k_list)


def run_sbert(train_df, test_df, params, k_list):
    model_name = params.get("model_name", "all-MiniLM-L6-v2")
    w_title = params.get("w_title", 0.6)
    w_scope = params.get("w_scope", 0.3)
    w_cat = params.get("w_cat", 0.1)

    train_titles = train_df["title"].tolist()
    test_titles = test_df["title"].tolist()
    train_scopes = train_df["text_raw"].tolist()
    test_scopes = test_df["text_raw"].tolist()
    train_cats = train_df["__primary_cat__"].tolist()
    test_cats = test_df["__primary_cat__"].tolist()

    model = SentenceTransformer(model_name)
    print(f"    Encoding embeddings for {model_name}...")
    train_title_emb = model.encode(train_titles, batch_size=64, show_progress_bar=True,
                                    convert_to_numpy=True, normalize_embeddings=True)
    train_scope_emb = model.encode(train_scopes, batch_size=64, show_progress_bar=True,
                                    convert_to_numpy=True, normalize_embeddings=True)
    test_title_emb = model.encode(test_titles, batch_size=64, show_progress_bar=True,
                                   convert_to_numpy=True, normalize_embeddings=True)
    test_scope_emb = model.encode(test_scopes, batch_size=64, show_progress_bar=True,
                                   convert_to_numpy=True, normalize_embeddings=True)

    # Category centroids
    unique_cats = sorted(set(train_cats))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    centroids = np.zeros((len(unique_cats), train_scope_emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_cats), dtype=int)
    for i, cat in enumerate(train_cats):
        centroids[cat_to_idx[cat]] += train_scope_emb[i]
        counts[cat_to_idx[cat]] += 1
    for i in range(len(unique_cats)):
        if counts[i] > 0:
            centroids[i] /= counts[i]

    max_k = max(k_list)
    train_cats_arr = np.array(train_cats)
    all_ranked_cats = []
    batch_size = 200

    for start in range(0, len(test_cats), batch_size):
        end = min(start + batch_size, len(test_cats))
        combined = (w_title * cosine_similarity(test_title_emb[start:end], train_title_emb) +
                    w_scope * cosine_similarity(test_scope_emb[start:end], train_scope_emb))
        if w_cat > 0:
            sim_centroids = cosine_similarity(test_scope_emb[start:end], centroids)
            train_cat_indices = np.array([cat_to_idx.get(c, 0) for c in train_cats])
            combined += w_cat * sim_centroids[:, train_cat_indices]

        top_indices = np.argsort(combined, axis=1)[:, ::-1][:, :max_k]
        for row_topk in top_indices:
            all_ranked_cats.append(train_cats_arr[row_topk].tolist())

    return evaluate_ranked_results(all_ranked_cats, test_cats, k_list)


def main():
    parser = argparse.ArgumentParser(description="Equal distribution evaluation")
    parser.add_argument("--train_csv", default="doaj_train.csv")
    parser.add_argument("--test_csv", default="doaj_test.csv")
    parser.add_argument("--n_train_per_cat", type=int, default=20)
    parser.add_argument("--n_test_per_cat", type=int, default=5)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("EQUAL DISTRIBUTION EXPERIMENT (B!SON Eval Setup 2)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(args.train_csv, dtype=str, low_memory=False).fillna("")
    test_df = pd.read_csv(args.test_csv, dtype=str, low_memory=False).fillna("")
    train_df = train_df[train_df["__primary_cat__"].str.strip() != ""].reset_index(drop=True)
    test_df = test_df[test_df["__primary_cat__"].str.strip() != ""].reset_index(drop=True)

    # Create balanced sets
    print("\nCreating balanced dataset...")
    bal_train, bal_test, qualifying_cats = create_equal_distribution(
        train_df, test_df, args.n_train_per_cat, args.n_test_per_cat
    )

    # Load best params
    print("\nLoading best hyperparameters...")
    best_params = load_best_params(args.results_dir)
    for model, params in best_params.items():
        print(f"  {model}: {params}")

    # Run all 3 models on balanced set (Title+Scope+Cat strategy only)
    all_results = {}
    execution_times = {}
    models = {
        "TF-IDF": (run_tfidf, best_params.get("tfidf", {})),
        "BM25": (run_bm25, best_params.get("bm25", {})),
        "SBERT": (run_sbert, best_params.get("sbert", {})),
    }

    for model_name, (run_fn, params) in models.items():
        print(f"\n{'─'*50}")
        print(f"Running: {model_name} (Title+Scope+Cat)")
        start = time.time()
        metrics = run_fn(bal_train, bal_test, params, K_LIST)
        elapsed = time.time() - start
        all_results[model_name] = metrics
        execution_times[model_name] = round(elapsed, 2)
        print(f"  HitRate@1={metrics['HitRate@1']:.4f}, MAP@10={metrics['MAP@10']:.4f}, "
              f"NDCG@10={metrics['NDCG@10']:.4f} ({elapsed:.1f}s)")

    # Save results
    output = {
        "experiment": "equal_distribution",
        "n_train_per_cat": args.n_train_per_cat,
        "n_test_per_cat": args.n_test_per_cat,
        "n_qualifying_categories": len(qualifying_cats),
        "n_train_total": len(bal_train),
        "n_test_total": len(bal_test),
        "strategy": "Title+Scope+Cat",
        "best_params": best_params,
        "results": all_results,
        "execution_times": execution_times,
    }
    json_path = os.path.join(args.results_dir, "equal_distribution_metrics.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {json_path}")

    # Comparison plot: Random vs Equal distribution
    print("\nGenerating comparison plot...")

    # Load random distribution results for comparison
    random_path = os.path.join(args.results_dir, "final_metrics.json")
    random_results = {}
    if os.path.exists(random_path):
        with open(random_path) as f:
            random_data = json.load(f)
        for model_name in ["TF-IDF", "BM25", "SBERT"]:
            key = f"{model_name} | Title+Scope+Cat"
            if key in random_data["results"]:
                random_results[model_name] = random_data["results"][key]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    model_names = ["TF-IDF", "BM25", "SBERT"]
    colors = {"TF-IDF": "#e74c3c", "BM25": "#3498db", "SBERT": "#2ecc71"}

    for ax_idx, k in enumerate([1, 10]):
        ax = axes[ax_idx]
        metric_key = f"HitRate@{k}"
        x = np.arange(len(model_names))
        width = 0.35

        random_vals = [random_results.get(m, {}).get(metric_key, 0) for m in model_names]
        equal_vals = [all_results.get(m, {}).get(metric_key, 0) for m in model_names]

        bars1 = ax.bar(x - width/2, random_vals, width, label="Random Distribution",
                       color=[colors[m] for m in model_names], alpha=0.7)
        bars2 = ax.bar(x + width/2, equal_vals, width, label="Equal Distribution",
                       color=[colors[m] for m in model_names], alpha=1.0,
                       edgecolor='black', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric_key, fontsize=11)
        ax.set_title(f"{metric_key}: Random vs Equal Distribution", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f"Equal Distribution Experiment ({len(qualifying_cats)} categories, "
                 f"{args.n_train_per_cat} train / {args.n_test_per_cat} test per cat)",
                 fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, "equal_distribution_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON: Random vs Equal Distribution (Title+Scope+Cat)")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Metric':<12} {'Random':>10} {'Equal':>10} {'Diff':>10}")
    print("-" * 52)
    for m in model_names:
        for k in [1, 5, 10]:
            mk = f"HitRate@{k}"
            r_val = random_results.get(m, {}).get(mk, 0)
            e_val = all_results.get(m, {}).get(mk, 0)
            diff = e_val - r_val
            print(f"{m:<10} {mk:<12} {r_val:>10.4f} {e_val:>10.4f} {diff:>+10.4f}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
