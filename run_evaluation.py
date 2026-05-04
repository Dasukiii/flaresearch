# run_evaluation.py
"""
Final controlled experiment: evaluates all 3 models x 3 input strategies
on the held-out test set using best-tuned hyperparameters.

Produces:
  - results/final_metrics.json
  - results/comparison_table.csv
  - results/hitrate_comparison.png
  - results/execution_time.png

Usage:
  python run_evaluation.py --train_csv doaj_train.csv --test_csv doaj_test.csv
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

# Input strategy definitions
INPUT_STRATEGIES = {
    "Title": {"use_title": True, "use_scope": False, "use_cat": False},
    "Title+Scope": {"use_title": True, "use_scope": True, "use_cat": False},
    "Title+Scope+Cat": {"use_title": True, "use_scope": True, "use_cat": True},
}


def load_best_params(results_dir="results"):
    """Load best hyperparameters from tuning JSONs."""
    params = {}
    for model in ["tfidf", "bm25", "sbert"]:
        path = os.path.join(results_dir, f"{model}_best_params.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            params[model] = data["best_params"]
            print(f"  Loaded {model} best params: {params[model]}")
        else:
            print(f"  WARNING: {path} not found, using defaults for {model}")
            if model == "tfidf":
                params[model] = {"ngram_max": 2, "max_features": 10000, "sublinear_tf": True, "min_df": 2}
            elif model == "bm25":
                params[model] = {"k1": 1.5, "b": 0.75}
            elif model == "sbert":
                params[model] = {"model_name": "all-MiniLM-L6-v2", "w_title": 0.6, "w_scope": 0.3, "w_cat": 0.1}
    return params


def build_query_text(row, strategy):
    """Build query text based on input strategy."""
    parts = []
    if strategy["use_title"]:
        parts.append(str(row.get("title_clean", row.get("title", ""))))
    if strategy["use_scope"]:
        parts.append(str(row.get("subjects_clean", "")))
        parts.append(str(row.get("keywords_clean", "")))
    if strategy["use_cat"]:
        cat = str(row.get("__primary_cat__", ""))
        if cat.strip():
            parts.append(cat)
    return " ".join(p for p in parts if p.strip())


def evaluate_ranked_results(ranked_cats_per_query, true_cats, k_list):
    """Compute all metrics for a set of ranked results."""
    metrics = {}
    for k in k_list:
        metrics[f"HitRate@{k}"] = mean_hit_rate_at_k(ranked_cats_per_query, true_cats, k)
        metrics[f"MAP@{k}"] = map_at_k(ranked_cats_per_query, true_cats, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(ranked_cats_per_query, true_cats, k)
    return metrics


# ======================== MODEL RUNNERS ========================

def run_tfidf(train_df, test_df, strategy_name, strategy, params, k_list):
    """Run TF-IDF retrieval for a given input strategy."""
    # Build corpus texts
    train_texts = [build_query_text(row, strategy) for _, row in train_df.iterrows()]
    test_texts = [build_query_text(row, strategy) for _, row in test_df.iterrows()]
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

    # Batch processing
    batch_size = 200
    for start in range(0, len(test_texts), batch_size):
        end = min(start + batch_size, len(test_texts))
        sims = cosine_similarity(test_matrix[start:end], train_matrix)
        top_indices = np.argsort(sims, axis=1)[:, ::-1][:, :max_k]
        for row_topk in top_indices:
            all_ranked_cats.append(train_cats_arr[row_topk].tolist())

    return evaluate_ranked_results(all_ranked_cats, test_cats, k_list)


def run_bm25(train_df, test_df, strategy_name, strategy, params, k_list):
    """Run BM25 retrieval for a given input strategy."""
    train_texts = [build_query_text(row, strategy) for _, row in train_df.iterrows()]
    test_texts = [build_query_text(row, strategy) for _, row in test_df.iterrows()]
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


def run_sbert(train_df, test_df, strategy_name, strategy, params, k_list, cache_dir="results"):
    """Run SBERT retrieval for a given input strategy."""
    model_name = params.get("model_name", "all-MiniLM-L6-v2")
    w_title = params.get("w_title", 0.6)
    w_scope = params.get("w_scope", 0.3)
    w_cat = params.get("w_cat", 0.1)

    # Prepare texts
    train_titles = train_df["title"].tolist()
    test_titles = test_df["title"].tolist()
    train_scopes = train_df["text_raw"].tolist()
    test_scopes = test_df["text_raw"].tolist()
    train_cats = train_df["__primary_cat__"].tolist()
    test_cats = test_df["__primary_cat__"].tolist()

    # Load or compute embeddings
    model = SentenceTransformer(model_name)
    print(f"    Encoding train titles ({len(train_titles)} docs)...")
    train_title_emb = model.encode(train_titles, batch_size=64, show_progress_bar=True,
                                    convert_to_numpy=True, normalize_embeddings=True)
    print(f"    Encoding train scopes...")
    train_scope_emb = model.encode(train_scopes, batch_size=64, show_progress_bar=True,
                                    convert_to_numpy=True, normalize_embeddings=True)
    print(f"    Encoding test titles ({len(test_titles)} docs)...")
    test_title_emb = model.encode(test_titles, batch_size=64, show_progress_bar=True,
                                   convert_to_numpy=True, normalize_embeddings=True)
    print(f"    Encoding test scopes...")
    test_scope_emb = model.encode(test_scopes, batch_size=64, show_progress_bar=True,
                                   convert_to_numpy=True, normalize_embeddings=True)

    max_k = max(k_list)
    train_cats_arr = np.array(train_cats)

    # Category centroids (from train scopes)
    unique_cats = sorted(set(train_cats))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    centroids = np.zeros((len(unique_cats), train_scope_emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_cats), dtype=int)
    for i, cat in enumerate(train_cats):
        idx = cat_to_idx[cat]
        centroids[idx] += train_scope_emb[i]
        counts[idx] += 1
    for i in range(len(unique_cats)):
        if counts[i] > 0:
            centroids[i] /= counts[i]

    all_ranked_cats = []
    batch_size = 200

    for start in range(0, len(test_cats), batch_size):
        end = min(start + batch_size, len(test_cats))

        # Compute similarity components based on strategy
        combined = np.zeros((end - start, len(train_cats)), dtype=np.float32)

        if strategy["use_title"]:
            sim_title = cosine_similarity(test_title_emb[start:end], train_title_emb)
            combined += w_title * sim_title

        if strategy["use_scope"]:
            sim_scope = cosine_similarity(test_scope_emb[start:end], train_scope_emb)
            combined += w_scope * sim_scope

        if strategy["use_cat"] and w_cat > 0:
            sim_q_to_centroids = cosine_similarity(test_scope_emb[start:end], centroids)
            train_cat_indices = np.array([cat_to_idx.get(c, 0) for c in train_cats])
            sim_cat = sim_q_to_centroids[:, train_cat_indices]
            combined += w_cat * sim_cat

        top_indices = np.argsort(combined, axis=1)[:, ::-1][:, :max_k]
        for row_topk in top_indices:
            all_ranked_cats.append(train_cats_arr[row_topk].tolist())

    return evaluate_ranked_results(all_ranked_cats, test_cats, k_list)


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(description="Run final controlled evaluation")
    parser.add_argument("--train_csv", default="doaj_train.csv")
    parser.add_argument("--test_csv", default="doaj_test.csv")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("FYP2 FINAL EVALUATION — Multi-Model Comparison")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(args.train_csv, dtype=str, low_memory=False).fillna("")
    test_df = pd.read_csv(args.test_csv, dtype=str, low_memory=False).fillna("")

    # Filter to valid rows
    train_df = train_df[train_df["text"].str.strip() != ""].reset_index(drop=True)
    test_df = test_df[test_df["text"].str.strip() != ""].reset_index(drop=True)
    train_df = train_df[train_df["__primary_cat__"].str.strip() != ""].reset_index(drop=True)
    test_df = test_df[test_df["__primary_cat__"].str.strip() != ""].reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Load best params
    print("\nLoading best hyperparameters...")
    best_params = load_best_params(args.results_dir)

    # Run all experiments
    all_results = {}
    execution_times = {}
    models = {
        "TF-IDF": (run_tfidf, best_params.get("tfidf", {})),
        "BM25": (run_bm25, best_params.get("bm25", {})),
        "SBERT": (run_sbert, best_params.get("sbert", {})),
    }

    for model_name, (run_fn, params) in models.items():
        for strategy_name, strategy in INPUT_STRATEGIES.items():
            key = f"{model_name} | {strategy_name}"
            print(f"\n{'─'*50}")
            print(f"Running: {key}")
            print(f"  Params: {params}")

            start = time.time()
            if model_name == "SBERT":
                metrics = run_fn(train_df, test_df, strategy_name, strategy, params, K_LIST,
                                 cache_dir=args.results_dir)
            else:
                metrics = run_fn(train_df, test_df, strategy_name, strategy, params, K_LIST)
            elapsed = time.time() - start

            all_results[key] = metrics
            execution_times[key] = round(elapsed, 2)
            print(f"  HitRate@1={metrics['HitRate@1']:.4f}, "
                  f"MAP@10={metrics['MAP@10']:.4f}, "
                  f"NDCG@10={metrics['NDCG@10']:.4f} "
                  f"({elapsed:.1f}s)")

    # Save comprehensive results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    # 1. JSON
    output = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "k_list": K_LIST,
        "best_params": best_params,
        "results": all_results,
        "execution_times": execution_times,
    }
    json_path = os.path.join(args.results_dir, "final_metrics.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {json_path}")

    # 2. CSV table
    rows = []
    for key, metrics in all_results.items():
        row = {"Experiment": key}
        row.update(metrics)
        row["Time (s)"] = execution_times[key]
        rows.append(row)
    table_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.results_dir, "comparison_table.csv")
    table_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # Print table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(table_df.to_string(index=False, float_format="%.4f"))

    # 3. HitRate@K comparison plot
    fig, axes = plt.subplots(1, len(K_LIST), figsize=(5 * len(K_LIST), 6))
    colors = {"TF-IDF": "#e74c3c", "BM25": "#3498db", "SBERT": "#2ecc71"}
    strategy_names = list(INPUT_STRATEGIES.keys())

    for ax_idx, k in enumerate(K_LIST):
        ax = axes[ax_idx] if len(K_LIST) > 1 else axes
        metric_key = f"HitRate@{k}"
        x = np.arange(len(strategy_names))
        width = 0.25
        offsets = [-width, 0, width]

        for m_idx, model_name in enumerate(["TF-IDF", "BM25", "SBERT"]):
            vals = []
            for s_name in strategy_names:
                key = f"{model_name} | {s_name}"
                vals.append(all_results.get(key, {}).get(metric_key, 0))
            ax.bar(x + offsets[m_idx], vals, width, label=model_name,
                   color=colors[model_name], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric_key)
        ax.set_title(f"{metric_key}")
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Model Comparison: HitRate@K by Input Strategy", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, "hitrate_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved {plot_path}")

    # 4. Execution time bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = list(execution_times.keys())
    times = list(execution_times.values())
    bar_colors = [colors.get(k.split(" | ")[0], "#888") for k in keys]
    ax.barh(keys, times, color=bar_colors, alpha=0.85)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Execution Time per Experiment")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    time_plot_path = os.path.join(args.results_dir, "execution_time.png")
    plt.savefig(time_plot_path, dpi=150)
    plt.close()
    print(f"  Saved {time_plot_path}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
