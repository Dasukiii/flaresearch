"""
evaluation_category.py

Simple evaluator comparing three retrieval modes:
  - title only
  - title + abstract
  - title + abstract + category (category appended as extra tokens)

Outputs:
  - prints aggregated Accuracy@K (HitRate@K), Precision@K, Recall@K for K in [1,3,5,10]
  - saves artifacts/category_eval_summary.json, per_query.csv and p_at_k_plot.png

Definitions used:
  - Relevant = same coarse label as query's ground-truth label (label field specified by --label_field).
  - Accuracy@K = HitRate@K (fraction queries with at least one relevant item in top-K).
  - Precision@K = avg over queries of (# relevant in top-K)/K.
  - Recall@K = avg over queries of (# relevant in top-K) / (# relevant in corpus excluding query).

Usage:
  python evaluate_category.py --art_dir artifacts --test_csv doaj_norm_test.csv --model all-MiniLM-L6-v2

Requires:
  numpy, pandas, sentence_transformers, scikit-learn, matplotlib
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
K_LIST = [1, 3, 5, 10]
DEFAULT_MODEL = "all-MiniLM-L6-v2"
WEIGHT_TITLE = 0.8   # weight for title similarity; scope weight = 1 - weight_title
OUT_DIR_DEFAULT = "artifacts"
# ----------------------------------------

def coarse_category(cat_raw):
    """Reduce 'A: B' -> 'A' etc. Return lower-cased string for comparisons."""
    if not isinstance(cat_raw, str):
        return ""
    s = cat_raw.strip()
    if s == "":
        return ""
    for sep in [':', '-', '|', ';', '/', '>']:
        if sep in s:
            first = s.split(sep)[0].strip()
            if first:
                return first.lower()
    return s.lower()

def load_artifacts(art_dir):
    docs_path = os.path.join(art_dir, "docs.csv")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Missing {docs_path}")
    docs = pd.read_csv(docs_path, dtype=str, low_memory=False).fillna("")

    def try_load(name):
        p = os.path.join(art_dir, name)
        return np.load(p) if os.path.exists(p) else None

    title_emb = try_load("doc_title_embeddings.npy")
    scope_emb = try_load("doc_scope_embeddings.npy")
    doc_emb = try_load("doc_embeddings.npy")
    # categories list optional
    cats = None
    cats_path = os.path.join(art_dir, "categories.pkl")
    if os.path.exists(cats_path):
        import pickle
        with open(cats_path, "rb") as f:
            cats = pickle.load(f)

    return docs, title_emb, scope_emb, doc_emb, cats

def compute_title_scope_embeddings_if_missing(docs, model, batch_size=256):
    titles = docs.get('title', docs.get('Journal title', pd.Series([""]*len(docs)))).astype(str).tolist()
    # build scope as subjects + keywords + text if present
    scopes = []
    for _, row in docs.iterrows():
        parts = []
        for k in ['subjects_raw','subjects','keywords_raw','keywords','text']:
            if k in row and isinstance(row[k], str) and row[k].strip():
                parts.append(row[k].strip())
        scopes.append(" ".join(parts))
    print("Computing title embeddings (may take a while)...")
    title_emb = model.encode(titles, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
    print("Computing scope embeddings (may take a while)...")
    scope_emb = model.encode(scopes, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
    return title_emb, scope_emb

def pick_label_field(docs, preferred_list=None):
    """Pick a label field from docs in order of preference; fallback to '__primary_cat__' or 'category'."""
    if preferred_list is None:
        preferred_list = ['journal','journal_name','journal_ref','__primary_cat__','category','subjects_raw','subjects','text']
    cols = docs.columns.tolist()
    for p in preferred_list:
        if p in cols:
            return p
    # fallback to first column named 'category' ignoring case
    for c in cols:
        if c.lower() == 'category':
            return c
    return None

def build_relevance_mask(docs, label_field):
    """Return array of coarse labels for each document and a dict mapping label -> indices."""
    labels = docs[label_field].fillna("").astype(str).apply(coarse_category).tolist()
    lab_to_idxs = {}
    for i, lab in enumerate(labels):
        lab_to_idxs.setdefault(lab, []).append(i)
    return np.array(labels), lab_to_idxs

def retrieve_topk_for_query(q_title_text, q_scope_text, q_cat, title_emb, scope_emb, weight_title, top_k):
    """Return indices of top_k docs by weighted combination of title & scope cosine similarity."""
    q_title_emb = model.encode([q_title_text], convert_to_numpy=True)
    q_scope_emb = model.encode([q_scope_text], convert_to_numpy=True)
    sims_title = cosine_similarity(q_title_emb, title_emb)[0]
    sims_scope = cosine_similarity(q_scope_emb, scope_emb)[0]
    combined = weight_title * sims_title + (1 - weight_title) * sims_scope
    order = np.argsort(combined)[::-1][:top_k]
    return order.tolist(), combined

def evaluate_modes_simple(art_dir, test_csv=None, model_name=DEFAULT_MODEL, weight_title=WEIGHT_TITLE, out_dir=OUT_DIR_DEFAULT):
    docs, title_emb, scope_emb, doc_emb, cats = load_artifacts(art_dir)

    # ensure embeddings present; compute if missing
    global model
    model = SentenceTransformer(model_name)
    if title_emb is None or scope_emb is None:
        title_emb, scope_emb = compute_title_scope_embeddings_if_missing(docs, model)

    n_docs = len(docs)
    print(f"Loaded {n_docs} docs; title_emb shape {title_emb.shape}, scope_emb shape {scope_emb.shape}")

    # choose label field (user can override via CLI)
    label_field = pick_label_field(docs)
    if label_field is None:
        raise RuntimeError("No suitable label field found (journal/category). Please provide a dataset with category/journal field.")
    print("Using label field for relevance:", label_field)

    labels_arr, lab_to_idxs = build_relevance_mask(docs, label_field)

    # prepare test indices
    if test_csv and os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv, dtype=str, low_memory=False).fillna("")
        # map test rows to docs by title or text exact match (best-effort)
        title_map = {str(t).strip().lower(): i for i,t in enumerate(docs.get('title', pd.Series([""]*len(docs))).astype(str).tolist())}
        text_map = {str(t).strip().lower(): i for i,t in enumerate(docs.get('text', pd.Series([""]*len(docs))).astype(str).tolist())}
        mapped = []
        for _, r in test_df.iterrows():
            t = str(r.get('title','')).strip().lower()
            idx = title_map.get(t)
            if idx is None:
                tx = str(r.get('text','')).strip().lower()
                idx = text_map.get(tx)
            if idx is not None:
                mapped.append(idx)
        test_indices = list(dict.fromkeys(mapped))  # unique preserve order
        if len(test_indices) == 0:
            raise RuntimeError("No test rows matched docs.csv. Provide a test CSV aligned with docs.csv.")
    else:
        # sample deterministic 20% of docs
        rng = np.random.RandomState(42)
        idxs = np.arange(n_docs)
        rng.shuffle(idxs)
        split = max(1, int(n_docs * 0.2))
        test_indices = idxs[:split].tolist()
        print(f"No test_csv given; using an internal 20% sample of {len(test_indices)} items for testing.")

    # Prepare storage for per-query results
    per_query_rows = []

    for qidx in test_indices:
        row = docs.iloc[qidx]
        q_title = str(row.get('title','')).strip()
        # build scope text: use abstract/text + subjects/keywords if available
        scope_parts = []
        for k in ['abstract','summary','text','subjects_raw','subjects','keywords_raw','keywords']:
            if k in row and isinstance(row[k], str) and row[k].strip():
                scope_parts.append(row[k].strip())
        q_scope = " ".join(scope_parts).strip()
        q_cat = ""
        for k in ['__primary_cat__','category','subjects_raw','subjects']:
            if k in row and isinstance(row[k], str) and row[k].strip():
                q_cat = row[k]; break
        # ground truth label
        true_label = coarse_category(str(row.get(label_field,"")))

        # --- Mode 1: title only
        order1, comb1 = retrieve_topk_for_query(q_title, q_title, q_cat, title_emb, scope_emb, weight_title, top_k=max(K_LIST)+1)
        # remove query itself if present
        order1 = [i for i in order1 if i != qidx][:max(K_LIST)]

        # --- Mode 2: title + abstract (q_scope)
        order2, comb2 = retrieve_topk_for_query(q_title, q_scope or q_title, q_cat, title_emb, scope_emb, weight_title, top_k=max(K_LIST)+1)
        order2 = [i for i in order2 if i != qidx][:max(K_LIST)]

        # --- Mode 3: title + abstract + category appended to scope
        q_scope_cat = (str(q_cat).strip() + " " + q_scope).strip() if q_cat else q_scope
        order3, comb3 = retrieve_topk_for_query(q_title, q_scope_cat or q_title, q_cat, title_emb, scope_emb, weight_title, top_k=max(K_LIST)+1)
        order3 = [i for i in order3 if i != qidx][:max(K_LIST)]

        # For each K compute counts
        row_out = {'qidx': int(qidx), 'true_label': true_label, 'q_title_preview': q_title[:120]}
        for mode_label, order in [('title', order1), ('title_abs', order2), ('title_abs_cat', order3)]:
            for K in K_LIST:
                topk = order[:K]
                # count relevant in topk (relevant = same coarse label as true)
                rel_in_topk = 0
                for idx in topk:
                    if coarse_category(str(docs.iloc[idx].get(label_field,""))) == true_label and true_label != "":
                        rel_in_topk += 1
                total_relevant = max(1, len(lab_to_idxs.get(true_label, [])) - (1 if true_label != "" and qidx in lab_to_idxs.get(true_label, []) else 0))
                # Precision@K
                prec = rel_in_topk / float(K)
                # Recall@K
                recall = rel_in_topk / float(total_relevant) if total_relevant > 0 else 0.0
                # Accuracy@K (HitRate): whether at least one relevant in top-K
                hit = 1.0 if rel_in_topk > 0 else 0.0
                row_out[f"{mode_label}_P@{K}"] = prec
                row_out[f"{mode_label}_R@{K}"] = recall
                row_out[f"{mode_label}_Acc@{K}"] = hit
        per_query_rows.append(row_out)

    # aggregate results
    perq_df = pd.DataFrame(per_query_rows)
    agg = {}
    modes = ['title','title_abs','title_abs_cat']
    for mode in modes:
        agg[mode] = {}
        for K in K_LIST:
            agg[mode][f"P@{K}"] = float(perq_df[f"{mode}_P@{K}"].mean())
            agg[mode][f"R@{K}"] = float(perq_df[f"{mode}_R@{K}"].mean())
            agg[mode][f"Acc@{K}"] = float(perq_df[f"{mode}_Acc@{K}"].mean())

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "category_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({'agg': agg, 'n_queries': len(perq_df)}, f, indent=2)
    perq_path = os.path.join(out_dir, "category_eval_per_query.csv")
    perq_df.to_csv(perq_path, index=False)

    # Plot comparisons: three subplots for Precision, Recall, Accuracy across K
    x = np.array(K_LIST)
    plt.figure(figsize=(10, 5))
    plt.subplot(1,3,1)
    for mode, label in zip(modes, ['Title','Title+Abstract','Title+Abstract+Category']):
        y = [agg[mode][f"P@{k}"] for k in K_LIST]
        plt.plot(x, y, marker='o', label=label)
    plt.title("Precision@K")
    plt.xlabel("K"); plt.ylabel("Precision"); plt.ylim(0,1); plt.xticks(K_LIST); plt.legend()

    plt.subplot(1,3,2)
    for mode, label in zip(modes, ['Title','Title+Abstract','Title+Abstract+Category']):
        y = [agg[mode][f"R@{k}"] for k in K_LIST]
        plt.plot(x, y, marker='o', label=label)
    plt.title("Recall@K")
    plt.xlabel("K"); plt.ylabel("Recall"); plt.ylim(0,1); plt.xticks(K_LIST)

    plt.subplot(1,3,3)
    for mode, label in zip(modes, ['Title','Title+Abstract','Title+Abstract+Category']):
        y = [agg[mode][f"Acc@{k}"] for k in K_LIST]
        plt.plot(x, y, marker='o', label=label)
    plt.title("Accuracy@K (HitRate)")
    plt.xlabel("K"); plt.ylabel("Accuracy"); plt.ylim(0,1); plt.xticks(K_LIST)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "category_eval_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Evaluation finished.")
    print(f"Queries evaluated: {len(perq_df)}")
    print(json.dumps(agg, indent=2))
    print("Saved:", summary_path, perq_path, plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--art_dir", default=OUT_DIR_DEFAULT, help="artifacts folder with docs.csv and embeddings")
    parser.add_argument("--test_csv", default=None, help="optional test CSV aligned with docs.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformer model (used only if embeddings missing)")
    parser.add_argument("--weight_title", default=WEIGHT_TITLE, type=float, help="weight for title similarity")
    parser.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="output folder for summary and plot")
    args = parser.parse_args()
    evaluate_modes_simple(args.art_dir, args.test_csv, args.model, args.weight_title, args.out_dir)
