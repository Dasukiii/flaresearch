# eval_per_category.py
"""
Per-category performance analysis.
Breaks down HitRate@1 by category to identify which categories are
easiest/hardest for each model.

Usage:
  python eval_per_category.py --train_csv doaj_train.csv --test_csv doaj_test.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

K_EVAL = 1  # Focus on HitRate@1


def load_best_params(results_dir="results"):
    """Load best hyperparameters from tuning JSONs."""
    params = {}
    for model in ["tfidf", "bm25", "sbert"]:
        path = os.path.join(results_dir, f"{model}_best_params.json")
        if os.path.exists(path):
            with open(path) as f:
                params[model] = json.load(f)["best_params"]
        else:
            if model == "tfidf":
                params[model] = {"ngram_max": 2, "max_features": 10000, "sublinear_tf": True, "min_df": 2}
            elif model == "bm25":
                params[model] = {"k1": 1.5, "b": 0.75}
            elif model == "sbert":
                params[model] = {"model_name": "all-MiniLM-L6-v2", "w_title": 0.6, "w_scope": 0.3, "w_cat": 0.1}
    return params


def build_query_text(row):
    """Build Title+Scope+Cat query text."""
    parts = [
        str(row.get("title_clean", "")),
        str(row.get("subjects_clean", "")),
        str(row.get("keywords_clean", "")),
        str(row.get("__primary_cat__", "")),
    ]
    return " ".join(p for p in parts if p.strip())


def get_per_query_hits(ranked_cats_per_query, true_cats):
    """Return list of 1/0 for each query indicating HitRate@1."""
    return [1 if rc[0] == tc else 0 for rc, tc in zip(ranked_cats_per_query, true_cats)]


def run_tfidf_per_query(train_df, test_df, params):
    train_texts = [build_query_text(row) for _, row in train_df.iterrows()]
    test_texts = [build_query_text(row) for _, row in test_df.iterrows()]
    train_cats = train_df["__primary_cat__"].tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, params.get("ngram_max", 2)),
        max_features=params.get("max_features", 10000),
        sublinear_tf=params.get("sublinear_tf", True),
        min_df=params.get("min_df", 2),
        dtype=np.float32
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)

    train_cats_arr = np.array(train_cats)
    all_ranked_cats = []
    batch_size = 200
    for start in range(0, len(test_texts), batch_size):
        end = min(start + batch_size, len(test_texts))
        sims = cosine_similarity(test_matrix[start:end], train_matrix)
        top_indices = np.argsort(sims, axis=1)[:, ::-1][:, :K_EVAL]
        for row_topk in top_indices:
            all_ranked_cats.append(train_cats_arr[row_topk].tolist())
    return all_ranked_cats


def run_bm25_per_query(train_df, test_df, params):
    train_texts = [build_query_text(row) for _, row in train_df.iterrows()]
    test_texts = [build_query_text(row) for _, row in test_df.iterrows()]
    train_cats = train_df["__primary_cat__"].tolist()

    train_tokens = [t.split() for t in train_texts]
    test_tokens = [t.split() for t in test_texts]
    bm25 = BM25Okapi(train_tokens, k1=params.get("k1", 1.5), b=params.get("b", 0.75))

    train_cats_arr = np.array(train_cats)
    all_ranked_cats = []
    for q_tokens in test_tokens:
        scores = bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:K_EVAL]
        all_ranked_cats.append(train_cats_arr[top_idx].tolist())
    return all_ranked_cats


def run_sbert_per_query(train_df, test_df, params):
    model_name = params.get("model_name", "all-MiniLM-L6-v2")
    w_title = params.get("w_title", 0.6)
    w_scope = params.get("w_scope", 0.3)
    w_cat = params.get("w_cat", 0.1)

    model = SentenceTransformer(model_name)
    print(f"    Encoding embeddings...")
    train_title_emb = model.encode(train_df["title"].tolist(), batch_size=64,
                                    show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    train_scope_emb = model.encode(train_df["text_raw"].tolist(), batch_size=64,
                                    show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    test_title_emb = model.encode(test_df["title"].tolist(), batch_size=64,
                                   show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    test_scope_emb = model.encode(test_df["text_raw"].tolist(), batch_size=64,
                                   show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    train_cats = train_df["__primary_cat__"].tolist()
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

    train_cats_arr = np.array(train_cats)
    all_ranked_cats = []
    batch_size = 200

    for start in range(0, len(test_df), batch_size):
        end = min(start + batch_size, len(test_df))
        combined = (w_title * cosine_similarity(test_title_emb[start:end], train_title_emb) +
                    w_scope * cosine_similarity(test_scope_emb[start:end], train_scope_emb))
        if w_cat > 0:
            sim_centroids = cosine_similarity(test_scope_emb[start:end], centroids)
            train_cat_indices = np.array([cat_to_idx.get(c, 0) for c in train_cats])
            combined += w_cat * sim_centroids[:, train_cat_indices]

        top_indices = np.argsort(combined, axis=1)[:, ::-1][:, :K_EVAL]
        for row_topk in top_indices:
            all_ranked_cats.append(train_cats_arr[row_topk].tolist())

    return all_ranked_cats


def main():
    parser = argparse.ArgumentParser(description="Per-category performance analysis")
    parser.add_argument("--train_csv", default="doaj_train.csv")
    parser.add_argument("--test_csv", default="doaj_test.csv")
    parser.add_argument("--min_test_samples", type=int, default=5,
                        help="Minimum test samples for a category to be included in analysis")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("PER-CATEGORY PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(args.train_csv, dtype=str, low_memory=False).fillna("")
    test_df = pd.read_csv(args.test_csv, dtype=str, low_memory=False).fillna("")
    train_df = train_df[train_df["__primary_cat__"].str.strip() != ""].reset_index(drop=True)
    test_df = test_df[test_df["__primary_cat__"].str.strip() != ""].reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    test_cats = test_df["__primary_cat__"].tolist()

    # Load best params
    best_params = load_best_params(args.results_dir)

    # Run all 3 models (Title+Scope+Cat)
    print("\n--- Running TF-IDF ---")
    tfidf_ranked = run_tfidf_per_query(train_df, test_df, best_params["tfidf"])
    print("--- Running BM25 ---")
    bm25_ranked = run_bm25_per_query(train_df, test_df, best_params["bm25"])
    print("--- Running SBERT ---")
    sbert_ranked = run_sbert_per_query(train_df, test_df, best_params["sbert"])

    # Get per-query hits
    tfidf_hits = get_per_query_hits(tfidf_ranked, test_cats)
    bm25_hits = get_per_query_hits(bm25_ranked, test_cats)
    sbert_hits = get_per_query_hits(sbert_ranked, test_cats)

    # Build per-category dataframe
    analysis_df = pd.DataFrame({
        "category": test_cats,
        "tfidf_hit": tfidf_hits,
        "bm25_hit": bm25_hits,
        "sbert_hit": sbert_hits,
    })

    cat_stats = analysis_df.groupby("category").agg(
        n_test=("category", "count"),
        tfidf_hitrate=("tfidf_hit", "mean"),
        bm25_hitrate=("bm25_hit", "mean"),
        sbert_hitrate=("sbert_hit", "mean"),
    ).reset_index()

    # Filter to categories with enough test samples
    cat_stats = cat_stats[cat_stats["n_test"] >= args.min_test_samples].sort_values(
        "tfidf_hitrate", ascending=False
    ).reset_index(drop=True)

    # Add training set size per category
    train_cat_counts = train_df["__primary_cat__"].value_counts().to_dict()
    cat_stats["n_train"] = cat_stats["category"].map(train_cat_counts).fillna(0).astype(int)

    # Add best model per category
    def best_model(row):
        scores = {"TF-IDF": row["tfidf_hitrate"], "BM25": row["bm25_hitrate"], "SBERT": row["sbert_hitrate"]}
        return max(scores, key=scores.get)
    cat_stats["best_model"] = cat_stats.apply(best_model, axis=1)

    print(f"\n{len(cat_stats)} categories with >= {args.min_test_samples} test samples\n")

    # Top 10 easiest categories (highest TF-IDF HitRate)
    print("TOP 10 EASIEST CATEGORIES (by TF-IDF HitRate@1)")
    print("-" * 80)
    top10 = cat_stats.head(10)
    for _, row in top10.iterrows():
        print(f"  {row['category'][:45]:<45} n={row['n_test']:>3} "
              f"TF-IDF={row['tfidf_hitrate']:.3f}  BM25={row['bm25_hitrate']:.3f}  "
              f"SBERT={row['sbert_hitrate']:.3f}")

    # Bottom 10 hardest categories
    print("\nBOTTOM 10 HARDEST CATEGORIES (by TF-IDF HitRate@1)")
    print("-" * 80)
    bottom10 = cat_stats.tail(10).iloc[::-1]
    for _, row in bottom10.iterrows():
        print(f"  {row['category'][:45]:<45} n={row['n_test']:>3} "
              f"TF-IDF={row['tfidf_hitrate']:.3f}  BM25={row['bm25_hitrate']:.3f}  "
              f"SBERT={row['sbert_hitrate']:.3f}")

    # Categories where SBERT beats TF-IDF
    sbert_wins = cat_stats[cat_stats["sbert_hitrate"] > cat_stats["tfidf_hitrate"]]
    print(f"\nCategories where SBERT > TF-IDF: {len(sbert_wins)} / {len(cat_stats)}")

    # Best model distribution
    best_model_counts = cat_stats["best_model"].value_counts()
    print(f"\nBest model per category:")
    for model, count in best_model_counts.items():
        print(f"  {model}: {count} categories ({count/len(cat_stats)*100:.1f}%)")

    # Save CSV
    csv_path = os.path.join(args.results_dir, "per_category_analysis.csv")
    cat_stats.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Save JSON summary
    summary = {
        "n_categories_analyzed": len(cat_stats),
        "min_test_samples": args.min_test_samples,
        "best_model_distribution": best_model_counts.to_dict(),
        "categories_where_sbert_beats_tfidf": len(sbert_wins),
        "mean_hitrate_across_categories": {
            "TF-IDF": float(cat_stats["tfidf_hitrate"].mean()),
            "BM25": float(cat_stats["bm25_hitrate"].mean()),
            "SBERT": float(cat_stats["sbert_hitrate"].mean()),
        },
        "top10_easiest": top10[["category", "n_test", "tfidf_hitrate", "bm25_hitrate", "sbert_hitrate"]].to_dict(orient="records"),
        "bottom10_hardest": bottom10[["category", "n_test", "tfidf_hitrate", "bm25_hitrate", "sbert_hitrate"]].to_dict(orient="records"),
    }
    json_path = os.path.join(args.results_dir, "per_category_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {json_path}")

    # ============ VISUALIZATION ============
    # 1. Top/Bottom categories horizontal bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top 15 easiest
    n_show = min(15, len(cat_stats))
    top_cats = cat_stats.head(n_show)
    y_pos = np.arange(n_show)
    bar_height = 0.25

    ax1.barh(y_pos - bar_height, top_cats["tfidf_hitrate"], bar_height, label="TF-IDF", color="#e74c3c", alpha=0.85)
    ax1.barh(y_pos, top_cats["bm25_hitrate"], bar_height, label="BM25", color="#3498db", alpha=0.85)
    ax1.barh(y_pos + bar_height, top_cats["sbert_hitrate"], bar_height, label="SBERT", color="#2ecc71", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([c[:35] for c in top_cats["category"]], fontsize=8)
    ax1.set_xlabel("HitRate@1")
    ax1.set_title("Top 15 Easiest Categories", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1.05)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Bottom 15 hardest
    bottom_cats = cat_stats.tail(n_show).iloc[::-1]
    y_pos2 = np.arange(len(bottom_cats))
    ax2.barh(y_pos2 - bar_height, bottom_cats["tfidf_hitrate"], bar_height, label="TF-IDF", color="#e74c3c", alpha=0.85)
    ax2.barh(y_pos2, bottom_cats["bm25_hitrate"], bar_height, label="BM25", color="#3498db", alpha=0.85)
    ax2.barh(y_pos2 + bar_height, bottom_cats["sbert_hitrate"], bar_height, label="SBERT", color="#2ecc71", alpha=0.85)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([c[:35] for c in bottom_cats["category"]], fontsize=8)
    ax2.set_xlabel("HitRate@1")
    ax2.set_title("Bottom 15 Hardest Categories", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    plt.suptitle("Per-Category HitRate@1 Analysis (Title+Scope+Cat)", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, "per_category_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {plot_path}")

    # 2. Scatter: category size vs HitRate
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (model, col) in zip(axes, [("TF-IDF", "tfidf_hitrate"),
                                         ("BM25", "bm25_hitrate"),
                                         ("SBERT", "sbert_hitrate")]):
        colors_map = {"TF-IDF": "#e74c3c", "BM25": "#3498db", "SBERT": "#2ecc71"}
        ax.scatter(cat_stats["n_train"], cat_stats[col], alpha=0.5, s=30, color=colors_map[model])
        ax.set_xlabel("Training Set Size (n_train)")
        ax.set_ylabel("HitRate@1")
        ax.set_title(f"{model}: Category Size vs Accuracy", fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)
        # Add trend line
        z = np.polyfit(cat_stats["n_train"], cat_stats[col], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(cat_stats["n_train"].min(), cat_stats["n_train"].max(), 100)
        ax.plot(x_trend, p(x_trend), "--", color="black", alpha=0.5, linewidth=1)

    plt.suptitle("Effect of Category Training Size on HitRate@1", fontsize=13)
    plt.tight_layout()
    scatter_path = os.path.join(args.results_dir, "category_size_vs_accuracy.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {scatter_path}")

    print(f"\n{'='*60}")
    print("PER-CATEGORY ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
