# evaluate.py
"""
Evaluation script for DOAJ-style journal artifacts.

Produces:
  - eval_metrics.json        (summary metrics per method and K)
  - recall_at_k.png          (grouped bar chart of HitRate@K for the 4 methods)

Usage examples:
  python evaluate.py --art_dir artifacts --test_csv doaj_norm_test.csv --out_dir artifacts
  python evaluate.py --art_dir artifacts --out_dir artifacts
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
K_LIST = [1, 3, 5, 10]
TITLE_WEIGHT = 0.6     # weight for title in Title+Scope combos
SCOPE_WEIGHT = 0.3     # weight for scope
CAT_WEIGHT = 0.1       # weight for category centroid boost (only for the +Category method)
MIN_TEST_SAMPLES = 10  # if test set smaller than this, warn

# ---------- HELPERS ----------
def load_artifacts(art_dir="artifacts"):
    docs_path = os.path.join(art_dir, "docs.csv")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"docs.csv not found in {art_dir}")
    docs = pd.read_csv(docs_path, dtype=str, low_memory=False).fillna("")

    def try_load(name):
        p = os.path.join(art_dir, name)
        return np.load(p) if os.path.exists(p) else None

    doc_emb = try_load("doc_embeddings.npy")
    title_emb = try_load("doc_title_embeddings.npy")
    scope_emb = try_load("doc_scope_embeddings.npy")
    cat_emb = try_load("category_embeddings.npy")

    categories = None
    cats_path = os.path.join(art_dir, "categories.pkl")
    if os.path.exists(cats_path):
        with open(cats_path, "rb") as f:
            categories = pickle.load(f)

    return docs, doc_emb, title_emb, scope_emb, cat_emb, categories

def normalize_text_for_match(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def map_rows_to_docs(df_rows, docs_df):
    """
    Try to map each row in df_rows to an index in docs_df.
    Matching strategy: exact normalized title -> docs['title'] OR exact normalized text -> docs['text'].
    Returns: list of indices (or None) with same length as df_rows.
    """
    docs_title_map = {normalize_text_for_match(t): i for i,t in enumerate(docs_df.get('title', pd.Series([""]*len(docs_df))).astype(str).tolist())}
    docs_text_map = {normalize_text_for_match(t): i for i,t in enumerate(docs_df.get('text', pd.Series([""]*len(docs_df))).astype(str).tolist())}

    mapped = []
    for _, row in df_rows.iterrows():
        cand = ""
        if 'title' in row:
            cand = normalize_text_for_match(row['title'])
        if cand and cand in docs_title_map:
            mapped.append(docs_title_map[cand])
            continue
        cand2 = ""
        if 'text' in row:
            cand2 = normalize_text_for_match(row['text'])
        if cand2 and cand2 in docs_text_map:
            mapped.append(docs_text_map[cand2])
            continue
        # fallback to a few possible keys
        found = False
        for k in ['title_clean','paper_title','paper_title_clean']:
            if k in row:
                cand3 = normalize_text_for_match(row[k])
                if cand3 in docs_title_map:
                    mapped.append(docs_title_map[cand3])
                    found = True
                    break
        if not found:
            mapped.append(None)
    return mapped

def primary_token_from_field(val):
    """Return primary token (coarse) from category-like field."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    for sep in (':', ';', ',', '|', '/', '-'):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if parts:
                return parts[0]
    return s

# ---------- METRIC COMPUTATION ----------
def evaluate_retrieval_methods(
    test_indices, docs, title_emb, scope_emb, cat_emb, doc_primary_cat_idx, categories,
    k_list=K_LIST,
    title_w=TITLE_WEIGHT, scope_w=SCOPE_WEIGHT, cat_w=CAT_WEIGHT
):
    """
    For each test index, compute rankings under 4 methods and compute HitRate@K and Precision@K.
    - test_indices: list of integer indices into docs (queries)
    - title_emb, scope_emb: numpy arrays (N x D)
    - cat_emb: C x D or None
    - doc_primary_cat_idx: numpy array length N of ints (category index) or -1
    """
    N = title_emb.shape[0]
    # prepare containers
    methods = ["title", "title_scope", "title_scope_cat", "title_scope_cat_filter"]
    results = {m: {K: 0 for K in k_list} for m in methods}
    counts = {m: 0 for m in methods}  # number of valid queries considered per method (should be equal)
    total_queries = 0

    for q_idx in test_indices:
        # ensure q_idx valid
        q_idx = int(q_idx)
        if q_idx < 0 or q_idx >= N:
            continue
        total_queries += 1

        # Build query embeddings (use the doc's own embeddings as proxy queries)
        q_title = title_emb[q_idx:q_idx+1]   # 1xd
        q_scope = scope_emb[q_idx:q_idx+1]   # 1xd

        # candidate indices: all except the query itself
        all_cand = np.arange(N)
        cand_mask = np.ones(N, dtype=bool)
        cand_mask[q_idx] = False
        cand_idx = all_cand[cand_mask]

        # compute pairwise sims vectorized
        sims_title_all = cosine_similarity(q_title, title_emb[cand_idx])[0]   # len(cand_idx)
        sims_scope_all = cosine_similarity(q_scope, scope_emb[cand_idx])[0]

        # candidate primary category indices
        cand_cat_idx = doc_primary_cat_idx[cand_idx]  # -1 for missing

        # For category-boosted method we will compute, for each candidate, sim between query scope and the candidate's category centroid.
        if cat_emb is not None and categories is not None:
            # compute sim between q_scope and all category centroids
            sim_q_to_cats = cosine_similarity(q_scope, cat_emb)[0]  # length C
            # map to candidates by indexing the centroid sim by candidate's cat index (or 0 if -1)
            cand_cat_sim = np.array([sim_q_to_cats[c] if c >= 0 and c < len(sim_q_to_cats) else 0.0 for c in cand_cat_idx])
        else:
            cand_cat_sim = np.zeros_like(sims_title_all)

        # compute combined raw scores
        combined_title_scope = title_w * sims_title_all + scope_w * sims_scope_all
        combined_title_scope_cat = title_w * sims_title_all + scope_w * sims_scope_all + cat_w * cand_cat_sim

        # For each method, build ranked candidate lists and check hits
        # Define a hit: candidate has same primary category as the query
        q_cat = doc_primary_cat_idx[q_idx]  # may be -1
        # If query category missing, we cannot evaluate "same category" hit; skip such queries
        if q_cat is None or int(q_cat) < 0:
            # we skip queries with no primary category for category-based hit evaluation.
            # However we can still consider pure semantic retrieval? for simplicity skip.
            continue

        # For each method, produce top-K candidate categories and test if any candidate shares q_cat
        # 1) Title
        order_title = np.argsort(sims_title_all)[::-1]  # indices into cand_idx
        cand_order_title = cand_idx[order_title]
        cand_cat_order_title = cand_cat_idx[order_title]

        # 2) Title+Scope
        order_ts = np.argsort(combined_title_scope)[::-1]
        cand_order_ts = cand_idx[order_ts]
        cand_cat_order_ts = cand_cat_idx[order_ts]

        # 3) Title+Scope+Category
        order_tsc = np.argsort(combined_title_scope_cat)[::-1]
        cand_order_tsc = cand_idx[order_tsc]
        cand_cat_order_tsc = cand_cat_idx[order_tsc]

        # 4) Title+Scope+Category (filter) -> restrict candidate set to same primary category as query
        same_cat_mask = (doc_primary_cat_idx == q_cat)
        same_cat_mask[q_idx] = False  # exclude the query itself
        cand_idx_samecat = np.where(same_cat_mask)[0]
        if cand_idx_samecat.size > 0:
            sims_title_same = cosine_similarity(q_title, title_emb[cand_idx_samecat])[0]
            sims_scope_same = cosine_similarity(q_scope, scope_emb[cand_idx_samecat])[0]
            if cat_emb is not None and categories is not None:
                sim_q_to_cats_local = cosine_similarity(q_scope, cat_emb)[0]
                cand_cat_idx_same = doc_primary_cat_idx[cand_idx_samecat]
                cand_cat_sim_same = np.array([sim_q_to_cats_local[c] if c>=0 and c < len(sim_q_to_cats_local) else 0.0 for c in cand_cat_idx_same])
            else:
                cand_cat_sim_same = np.zeros_like(sims_title_same)
            combined_same = title_w * sims_title_same + scope_w * sims_scope_same + cat_w * cand_cat_sim_same
            order_same = np.argsort(combined_same)[::-1]
            cand_order_same = cand_idx_samecat[order_same]
            cand_cat_order_same = doc_primary_cat_idx[cand_order_same]
        else:
            cand_order_same = np.array([], dtype=int)
            cand_cat_order_same = np.array([], dtype=int)

        # now evaluate HitRate@K and Precision@K per method
        # helper to count if any candidate in top-K has same category
        def evaluate_hits(top_order_indices, top_cats, K):
            if len(top_cats) == 0:
                return 0  # no candidates -> no hit
            topk = top_cats[:K]
            # hit if any equals q_cat
            return 1 if (np.array(topk) == q_cat).any() else 0

        # Title
        for K in K_LIST:
            hit = evaluate_hits(cand_order_title, cand_cat_order_title, K)
            results["title"][K] += hit
        counts["title"] += 1

        # Title+Scope
        for K in K_LIST:
            hit = evaluate_hits(cand_order_ts, cand_cat_order_ts, K)
            results["title_scope"][K] += hit
        counts["title_scope"] += 1

        # Title+Scope+Cat
        for K in K_LIST:
            hit = evaluate_hits(cand_order_tsc, cand_cat_order_tsc, K)
            results["title_scope_cat"][K] += hit
        counts["title_scope_cat"] += 1

        # Title+Scope+Cat (filter)
        for K in K_LIST:
            hit = evaluate_hits(cand_order_same, cand_cat_order_same, K)
            results["title_scope_cat_filter"][K] += hit
        counts["title_scope_cat_filter"] += 1

    # finalize by dividing by counts to get HitRate (Recall@K-like)
    metrics = {}
    for m in results:
        n = counts.get(m, 0)
        if n == 0:
            # no queries successfully evaluated for this method
            metrics[m] = {K: None for K in K_LIST}
            continue
        metrics[m] = {K: float(results[m][K]) / float(n) for K in K_LIST}
    # also include top1 accuracy = HitRate@1
    summary = {m: {"accuracy_top1": metrics[m][1] if metrics[m][1] is not None else 0.0, "recall_at_k": metrics[m]} for m in metrics}
    return summary, total_queries

# ---------- MAIN ----------
def main(art_dir="artifacts", test_csv=None, out_dir="artifacts"):
    docs, doc_emb, title_emb, scope_emb, cat_emb, categories = load_artifacts(art_dir)

    # Basic checks
    if title_emb is None or scope_emb is None:
        raise FileNotFoundError("Title and scope embeddings required (doc_title_embeddings.npy and doc_scope_embeddings.npy). Run embed_index.py first.")

    N = title_emb.shape[0]
    # Build primary category index per doc (map category string -> index in categories list)
    if categories is not None:
        cat_to_idx = {c: i for i, c in enumerate(categories)}
    else:
        cat_to_idx = {}

    # Derive each doc's primary category index from '__primary_cat__' or 'category' or '' -> -1
    def doc_primary_cat_idx_array(docs_df, cat_to_idx_map):
        arr = []
        for _, row in docs_df.iterrows():
            val = ""
            if '__primary_cat__' in docs_df.columns and str(row.get('__primary_cat__', "")).strip():
                val = str(row.get('__primary_cat__', "")).strip()
            elif 'category' in docs_df.columns and str(row.get('category', "")).strip():
                val = str(row.get('category', "")).strip()
            else:
                val = ""
            if val == "":
                arr.append(-1)
            else:
                # coarse: take primary token
                p = primary_token_from_field(val)
                if p in cat_to_idx_map:
                    arr.append(cat_to_idx_map[p])
                else:
                    # unknown fine-grained string -> try to coarse-match
                    found = None
                    for kname in cat_to_idx_map.keys():
                        if p.lower() == kname.lower() or p.lower() in kname.lower() or kname.lower() in p.lower():
                            found = cat_to_idx_map[kname]
                            break
                    arr.append(found if found is not None else -1)
        return np.array(arr, dtype=int)

    doc_cat_idx = doc_primary_cat_idx_array(docs, cat_to_idx)

    # Determine test indices
    test_indices = None
    if test_csv and os.path.exists(test_csv):
        try:
            test_df = pd.read_csv(test_csv, dtype=str, low_memory=False).fillna("")
            mapped = map_rows_to_docs(test_df, docs)
            mapped_indices = [m for m in mapped if m is not None]
            print(f"Mapped {len(mapped_indices)}/{len(mapped)} test rows to docs.csv")
            if len(mapped_indices) > 0:
                test_indices = mapped_indices
        except Exception as e:
            print("Failed to map test_csv to docs:", e)
            test_indices = None

    if test_indices is None:
        # fallback to internal split (deterministic)
        y_labels = docs.get('__primary_cat__', None)
        if y_labels is None or y_labels.astype(str).str.strip().eq("").all():
            # no categories: just random split
            idx = np.arange(N)
            train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
        else:
            # stratify when possible
            strat_col = docs['__primary_cat__'].astype(str).replace("", np.nan)
            try:
                train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, random_state=42, stratify=strat_col.fillna("NA"))
            except Exception:
                train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, random_state=42)
        test_indices = test_idx.tolist()
        print(f"No external test CSV mapped — using internal split. Test size = {len(test_indices)}")

    if len(test_indices) < MIN_TEST_SAMPLES:
        print(f"Warning: only {len(test_indices)} test queries available; metrics may be noisy.")

    # Evaluate methods
    summary_metrics, total_q = evaluate_retrieval_methods(
        test_indices, docs, title_emb, scope_emb, cat_emb, doc_cat_idx, categories,
        k_list=K_LIST, title_w=TITLE_WEIGHT, scope_w=SCOPE_WEIGHT, cat_w=CAT_WEIGHT
    )

    # Build JSON friendly structure and save
    # Build JSON friendly structure and save
    out = {
        "n_queries_evaluated": total_q,
        "K_list": K_LIST,
        "weights": {"title": TITLE_WEIGHT, "scope": SCOPE_WEIGHT, "category": CAT_WEIGHT},
        "methods_summary": summary_metrics
    }
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved metrics to", metrics_path)

    # Create grouped bar chart for HitRate@K (Recall@K proxy) across methods
    methods = ["title", "title_scope", "title_scope_cat", "title_scope_cat_filter"]
    labels = ["Title", "Title+Scope", "Title+Scope+Cat", "Title+Scope+Cat (filter)"]
    x = np.arange(len(methods))
    width = 0.18
    plt.figure(figsize=(10,5))
    offsets = np.linspace(-1.5*width, 1.5*width, len(K_LIST))

    # For each K, collect values across methods (defensive: missing keys -> 0.0)
    for i, K in enumerate(K_LIST):
        vals = []
        for m in methods:
            # summary_metrics[m] has structure: {'accuracy_top1': ..., 'recall_at_k': {K: value, ...}}
            method_entry = summary_metrics.get(m, {})
            recall_map = method_entry.get('recall_at_k', {}) if isinstance(method_entry, dict) else {}
            # keys in recall_map should be ints; handle both int and str keys defensively
            v = recall_map.get(K, None)
            if v is None:
                # try string key fallback (if JSON-style conversion happened earlier)
                v = recall_map.get(str(K), None)
            vals.append(v if v is not None else 0.0)

        plt.bar(x + offsets[i], vals, width=width, label=f"@{K}")

    plt.xticks(x, labels, rotation=20)
    plt.ylim(0,1.0)
    plt.ylabel("HitRate@K (any same-primary-category in top-K)")
    plt.title("Comparison of retrieval methods (HitRate@K)")
    plt.legend(title="K")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "recall_at_k.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved plot to", fig_path)

    print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--art_dir", default="artifacts", help="Artifacts folder path")
    parser.add_argument("--test_csv", default=None, help="Optional test CSV to map to docs rows")
    parser.add_argument("--out_dir", default="artifacts", help="Output folder path for metrics/plots")
    args = parser.parse_args()
    main(args.art_dir, args.test_csv, args.out_dir)
