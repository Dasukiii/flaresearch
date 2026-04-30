# app.py
"""
FlareSearch — Journal Finder powered by TF-IDF.
Uses the best-performing TF-IDF model (Optuna-tuned) for fast, accurate matching.

Features:
- TF-IDF with Optuna-optimised hyperparameters (ngram_max=3, max_features=3000).
- NLTK preprocessing applied to user queries (matching the training pipeline).
- Coarsened category dropdown for optional filtering.
- Shows only journals whose similarity score >= threshold.
"""
import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import nltk

# Ensure NLTK data is available (needed for Streamlit Cloud deployment)
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import nltk_clean

# ---------- CONFIG ----------
DATA_DIR = "."
RESULTS_DIR = "results"
SCORE_THRESHOLD = 0.05           # cosine similarity threshold (TF-IDF range ~0-1)
MAX_DISPLAY = 10                 # max journals to display

# Default TF-IDF params (Optuna best)
DEFAULT_PARAMS = {
    "ngram_max": 3,
    "max_features": 3000,
    "sublinear_tf": True,
    "min_df": 5
}

# ---------- CACHED RESOURCES ----------
@st.cache_data(show_spinner=False)
def load_docs():
    """Load the normalised journal dataset."""
    # Try artifacts/docs.csv first, then doaj_normalized.csv, then doaj_train.csv
    for path in ["artifacts/docs.csv", "doaj_normalized.csv", "doaj_train.csv"]:
        full = os.path.join(DATA_DIR, path)
        if os.path.exists(full):
            docs = pd.read_csv(full, dtype=str, low_memory=False).fillna("")
            return docs
    raise FileNotFoundError("No journal dataset found (docs.csv / doaj_normalized.csv / doaj_train.csv)")

@st.cache_data(show_spinner=False)
def load_best_params():
    """Load Optuna-tuned TF-IDF params if available."""
    path = os.path.join(RESULTS_DIR, "tfidf_best_params.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data["best_params"]
    return DEFAULT_PARAMS

@st.cache_resource(show_spinner=False)
def build_tfidf_index(docs_df, params):
    """Build TF-IDF vectoriser and transform all journal texts."""
    # Build scope text for each journal: title_clean + subjects_clean + keywords_clean + category
    texts = []
    for _, row in docs_df.iterrows():
        parts = []
        for col in ['title_clean', 'title', 'Journal title']:
            if col in row and str(row[col]).strip():
                parts.append(str(row[col]).strip())
                break
        for col in ['subjects_clean', 'subjects', 'subjects_raw']:
            if col in row and str(row[col]).strip():
                parts.append(str(row[col]).strip())
                break
        for col in ['keywords_clean', 'keywords', 'keywords_raw']:
            if col in row and str(row[col]).strip():
                parts.append(str(row[col]).strip())
                break
        for col in ['__primary_cat__', 'category']:
            if col in row and str(row[col]).strip():
                parts.append(str(row[col]).strip())
                break
        texts.append(" ".join(parts))

    vectoriser = TfidfVectorizer(
        ngram_range=(1, params.get("ngram_max", 3)),
        max_features=params.get("max_features", 3000),
        sublinear_tf=params.get("sublinear_tf", True),
        min_df=params.get("min_df", 5),
        dtype=np.float32
    )
    tfidf_matrix = vectoriser.fit_transform(texts)
    return vectoriser, tfidf_matrix

# ---------- UTILITIES ----------
def canonical_journal_title(row):
    for key in ['Journal title', 'journal title', 'title', 'journal_name', 'journal', 'venue', 'name']:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    return "(no title)"

def get_subjects_preview(row):
    for k in ['subjects_raw', 'subjects', 'keywords_raw', 'keywords', 'category', '__primary_cat__', 'text']:
        if k in row and pd.notna(row[k]) and str(row[k]).strip():
            s = str(row[k]).strip()
            return s if len(s) <= 200 else s[:200] + "..."
    return ""

def coarse_category(cat_raw):
    """Reduce 'A: B' or 'Agriculture: Forestry' -> 'Agriculture' (first segment)."""
    if not isinstance(cat_raw, str) or not cat_raw.strip():
        return ""
    for sep in [':', '-', '|', ';', '/', '>']:
        if sep in cat_raw:
            first = cat_raw.split(sep)[0].strip()
            if first:
                return first
    tokens = cat_raw.strip().split()
    return tokens[0] if tokens else cat_raw.strip()

def build_query_text(title, abstract, selected_category=None):
    """Build a query string from user input, matching the training text format."""
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if abstract and abstract.strip():
        parts.append(abstract.strip())
    if selected_category:
        parts.append(selected_category)
    return " ".join(parts)

# ---------- RECOMMENDATION ----------
def recommend(title, abstract, docs, vectoriser, tfidf_matrix, selected_categories=None):
    """
    Build a TF-IDF query vector from user input, compute cosine similarity
    against all journal vectors, and return top matches above threshold.
    """
    if not (title and title.strip()) and not (abstract and abstract.strip()):
        return []

    # Step 1: Clean title + abstract only (same NLTK pipeline as training data)
    cleaned_query = nltk_clean(build_query_text(title, abstract))

    # Step 2: Resolve category — map coarse selection back to full __primary_cat__ values
    #   The TF-IDF index contains raw __primary_cat__ (e.g. "Social Sciences: Economic theory.")
    #   The dropdown shows coarse categories (e.g. "Social Sciences")
    #   We append the full cat string so the terms match the index vocabulary.
    cat_col = next((c for c in ['__primary_cat__', 'category', 'subjects_raw', 'subjects'] if c in docs.columns), None)
    candidate_idx = np.arange(len(docs))

    if selected_categories and cat_col:
        mask = docs[cat_col].astype(str).apply(
            lambda x: any(coarse_category(sel).lower() in coarse_category(str(x)).lower() for sel in selected_categories)
        ).values
        filtered = np.where(mask)[0]
        if filtered.size > 0:
            candidate_idx = filtered
            # Use the most common full category string to enrich the query
            cat_counts = docs.iloc[filtered][cat_col].astype(str).value_counts()
            top_cat = cat_counts.index[0] if len(cat_counts) > 0 else ""
            if top_cat:
                cleaned_query = cleaned_query + " " + top_cat.lower()

    # Step 3: Transform cleaned query using the fitted vectoriser
    query_vec = vectoriser.transform([cleaned_query])

    # Compute cosine similarity only against candidates
    sims = cosine_similarity(query_vec, tfidf_matrix[candidate_idx])[0]

    # Build results
    results = []
    for local_idx, idx in enumerate(candidate_idx):
        score = float(sims[local_idx])
        if score >= SCORE_THRESHOLD:
            row = docs.iloc[int(idx)]
            results.append({
                'idx': int(idx),
                'journal_title': canonical_journal_title(row),
                'score': score,
                'subjects_preview': get_subjects_preview(row),
                'publisher': str(row.get('publisher', "")) if 'publisher' in row else "",
                'url': str(row.get('url', "")) if 'url' in row else ""
            })

    results = sorted(results, key=lambda x: x['score'], reverse=True)[:MAX_DISPLAY]
    return results

# ---------- STREAMLIT UI ----------
def main():
    st.set_page_config(page_title="FlareSearch - Aligning Manuscripts with Suitable Journal", layout="wide")
    st.markdown("## FlareSearch Journal Finder")
    st.caption("Powered by TF-IDF with Optuna-optimised hyperparameters")

    # Load data and build index
    try:
        docs = load_docs()
    except Exception as e:
        st.error(f"Failed to load journal data: {e}")
        return

    params = load_best_params()

    with st.spinner("Building TF-IDF index (first run only)..."):
        vectoriser, tfidf_matrix = build_tfidf_index(docs, params)

    # Build coarse categories list for dropdown
    cat_col = next((c for c in ['__primary_cat__', 'category', 'subjects_raw', 'subjects'] if c in docs.columns), None)
    if cat_col:
        coarse = docs[cat_col].astype(str).map(coarse_category)
        available_categories = sorted([c for c in coarse.unique() if c and c.strip()])
    else:
        available_categories = []

    # UI inputs
    st.markdown("### Manuscript input")
    title = st.text_input("Title")
    abstract = st.text_area("Abstract", height=200)

    if available_categories:
        st.markdown("### Optional: choose coarse subject(s) to narrow search")
        selected_cats = st.multiselect("Subject(s)", options=available_categories)
        if selected_cats == []:
            selected_cats = None
    else:
        selected_cats = None
        st.markdown("_No category field available in the dataset._")

    if st.button("Find matching journals"):
        if not title.strip() and not abstract.strip():
            st.warning("Enter title or abstract to search.")
        else:
            with st.spinner("Searching..."):
                results = recommend(
                    title=title, abstract=abstract, docs=docs,
                    vectoriser=vectoriser, tfidf_matrix=tfidf_matrix,
                    selected_categories=selected_cats
                )

            if not results:
                st.info("No suitable journal found. Try clearing category filter or expanding your abstract.")
            else:
                st.markdown(f"### Matched journals")
                for i, r in enumerate(results, start=1):
                    st.markdown(f"**{i}. {r['journal_title']}** — score: {r['score']:.3f}")
                    if r['url']:
                        st.markdown(f"[Link]({r['url']})")
                    if r['publisher']:
                        st.markdown(f"*Publisher:* {r['publisher']}")
                    if r['subjects_preview']:
                        st.markdown(f"**Scope:** {r['subjects_preview']}")
                    st.markdown("---")

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("© 2026 — FlareSearch Journal Finder ver 2.0 | TF-IDF engine")

if __name__ == "__main__":
    main()
