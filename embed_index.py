# embed_index.py
"""
Compute and save embeddings for a journal-level normalized CSV (DOAJ-style).

Outputs (saved to --out_dir):
  - docs.csv                 (saved copy of normalized input, row order preserved)
  - doc_title_embeddings.npy (N x D) - embeddings for journal titles (float32)
  - doc_scope_embeddings.npy (N x D) - embeddings for journal scopes (subjects+keywords or text) (float32)
  - doc_embeddings.npy       (N x D) - combined embedding (avg(title, scope)) for backward compatibility (float32)
  - category_embeddings.npy  (C x D) - centroids per primary category (if category info exists) (float32)
  - categories.pkl           (list of category names corresponding to category_embeddings)
  - cat_to_doc_indices.pkl   (dict: category -> [doc_row_indices]) for fast filtering

Usage:
  python embed_index.py --input_csv doaj_norm_normalized.csv --model all-MiniLM-L6-v2 --out_dir artifacts
"""
import os
import argparse
from typing import List, Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

def find_title_column(columns: List[str]) -> str:
    candidates = ['title', 'journal title', 'journal', 'journal title', 'paper_title', 'name']
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fallback to first column
    return columns[0]

def build_scope_text(row, keys_priority=None):
    """Build a scope/description string from available fields.
    Default priority: subjects_raw, subjects, keywords_raw, keywords, text, text_raw."""
    if keys_priority is None:
        keys_priority = ['subjects_raw', 'subjects', 'keywords_raw', 'keywords', 'text', 'text_raw']
    parts = []
    for k in keys_priority:
        if k in row and isinstance(row[k], str) and row[k].strip():
            parts.append(row[k].strip())
    return " ".join(parts).strip()

def primary_category_token(cat_str: str) -> str:
    """Return coarse primary token for category grouping (before ':' or other separators)."""
    if not isinstance(cat_str, str) or not cat_str.strip():
        return ""
    for sep in [':', '-', '|', '/', ';', '>']:
        if sep in cat_str:
            return cat_str.split(sep)[0].strip()
    return cat_str.strip()

def batch_encode(model: SentenceTransformer, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Encode a list of texts in batches and return a numpy array (len x dim) of dtype float32."""
    if len(texts) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    embeddings = []
    rng = range(0, len(texts), batch_size)
    if show_progress:
        rng = tqdm(rng, desc="Encoding batches", unit="batch")
    for i in rng:
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    emb_all = np.vstack(embeddings)
    return np.asarray(emb_all, dtype=np.float32)

def main(input_csv: str, model_name: str = 'all-MiniLM-L6-v2', out_dir: str = 'artifacts', batch_size: int = 64):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading CSV:", input_csv)
    df = pd.read_csv(input_csv, dtype=str, encoding='utf-8', low_memory=False).fillna("")

    # ensure canonical title column
    title_col = find_title_column(df.columns.tolist())
    print("Detected title column:", title_col)
    df['title'] = df[title_col].astype(str)

    # build a scope column from preferred fields
    df['scope_raw'] = df.apply(lambda r: build_scope_text(r, ['subjects_raw', 'subjects', 'keywords_raw', 'keywords', 'text', 'text_raw']), axis=1)

    # primary category extraction: prefer '__primary_cat__' if present, else 'category', else try subjects_raw
    if '__primary_cat__' in df.columns and df['__primary_cat__'].astype(str).str.strip().any():
        df['__primary_cat__'] = df['__primary_cat__'].astype(str)
    elif 'category' in df.columns and df['category'].astype(str).str.strip().any():
        df['__primary_cat__'] = df['category'].astype(str).apply(primary_category_token)
    elif 'subjects_raw' in df.columns and df['subjects_raw'].astype(str).str.strip().any():
        df['__primary_cat__'] = df['subjects_raw'].astype(str).apply(primary_category_token)
    else:
        df['__primary_cat__'] = ""

    # save canonical docs copy (this ensures row alignment)
    docs_out_path = os.path.join(out_dir, "docs.csv")
    df.to_csv(docs_out_path, index=False)
    print("Saved docs copy to:", docs_out_path)

    # load model
    print("Loading SentenceTransformer model:", model_name)
    model = SentenceTransformer(model_name)

    # prepare texts for encoding
    titles = df['title'].astype(str).tolist()
    scopes = df['scope_raw'].astype(str).tolist()

    # encode
    print(f"Encoding title embeddings (N={len(titles)}) ...")
    title_emb = batch_encode(model, titles, batch_size=batch_size, show_progress=True)
    print("Title embeddings shape:", title_emb.shape)

    print(f"Encoding scope embeddings (N={len(scopes)}) ...")
    scope_emb = batch_encode(model, scopes, batch_size=batch_size, show_progress=True)
    print("Scope embeddings shape:", scope_emb.shape)

    # confirm dims match
    if title_emb.shape[1] != scope_emb.shape[1]:
        raise RuntimeError(f"Embedding dimension mismatch: title {title_emb.shape[1]} vs scope {scope_emb.shape[1]}")

    # combined doc embedding (simple average of title + scope)
    doc_emb = ((title_emb + scope_emb) / 2.0).astype(np.float32)

    # Save embeddings (float32 to save memory)
    np.save(os.path.join(out_dir, "doc_title_embeddings.npy"), title_emb)
    np.save(os.path.join(out_dir, "doc_scope_embeddings.npy"), scope_emb)
    np.save(os.path.join(out_dir, "doc_embeddings.npy"), doc_emb)
    print("Saved doc_title_embeddings.npy, doc_scope_embeddings.npy and doc_embeddings.npy to", out_dir)

    # Build category -> doc index mapping and compute centroids (if categories present)
    primary_cats = df['__primary_cat__'].astype(str).tolist()
    cat_to_indices: Dict[str, List[int]] = {}
    for idx, cat in enumerate(primary_cats):
        cat_clean = cat.strip()
        if cat_clean == "":
            continue
        cat_to_indices.setdefault(cat_clean, []).append(idx)

    if len(cat_to_indices) > 0:
        print(f"Found {len(cat_to_indices)} non-empty primary categories; computing centroids and saving index map...")
        categories = []
        cat_centroids = []
        for cat_name, indices in cat_to_indices.items():
            categories.append(cat_name)
            # use combined doc_emb rows for centroid (average)
            vecs = doc_emb[indices]
            centroid = np.mean(vecs, axis=0)
            cat_centroids.append(centroid)
        cat_centroids = np.vstack(cat_centroids).astype(np.float32)

        # Save category embeddings + list + mapping
        np.save(os.path.join(out_dir, "category_embeddings.npy"), cat_centroids)
        with open(os.path.join(out_dir, "categories.pkl"), "wb") as f:
            pickle.dump(categories, f)
        with open(os.path.join(out_dir, "cat_to_doc_indices.pkl"), "wb") as f:
            pickle.dump(cat_to_indices, f)
        print(f"Saved {len(categories)} category centroids and cat_to_doc_indices mapping.")
    else:
        print("No primary categories found — skipped category centroid calculation.")

    print("All artifacts written to", out_dir)
    print("DONE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Normalized CSV input (e.g., doaj_norm_normalized.csv)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--out_dir", default="artifacts", help="Output directory for artifacts")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    args = parser.parse_args()
    main(args.input_csv, model_name=args.model, out_dir=args.out_dir, batch_size=args.batch_size)
