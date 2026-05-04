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
SCORE_THRESHOLD = 0.20           # cosine similarity threshold (TF-IDF range ~0-1)
MAX_DISPLAY = 10                 # max journals to display

# Default TF-IDF params (Optuna best)
DEFAULT_PARAMS = {
    "ngram_max": 3,
    "max_features": 3000,
    "sublinear_tf": True,
    "min_df": 5
}


# ---------- CUSTOM CSS ----------
CUSTOM_CSS = """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e94560, #ff6b6b, #ffa07a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #8899aa;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Stat cards row */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-card .num {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e94560;
    }
    .stat-card .label {
        font-size: 0.78rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Result card */
    .result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .result-card:hover {
        border-color: #e94560;
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .result-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f0f0f0;
    }
    .result-rank {
        font-size: 0.8rem;
        font-weight: 600;
        color: #0E1117;
        background: #e94560;
        border-radius: 6px;
        padding: 2px 10px;
        min-width: 28px;
        text-align: center;
    }

    /* Score bar */
    .score-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        transition: width 0.5s ease;
    }
    .score-text {
        font-size: 0.82rem;
        color: #8899aa;
    }

    /* Meta info */
    .result-meta {
        font-size: 0.85rem;
        color: #7a8a9a;
        margin-top: 0.3rem;
    }
    .result-scope {
        font-size: 0.83rem;
        color: #6a7a8a;
        margin-top: 0.4rem;
        padding: 0.5rem 0.7rem;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        border-left: 3px solid #e94560;
    }



    /* Footer */
    .footer {
        text-align: center;
        color: #556677;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}

    /* Adjust button styling */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c9384e);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff5a75, #e94560);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
    }
</style>
"""

# ---------- CACHED RESOURCES ----------
@st.cache_data(show_spinner=False)
def load_docs():
    """Load the normalised journal dataset."""
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

    # Step 2: Resolve category
    cat_col = next((c for c in ['__primary_cat__', 'category', 'subjects_raw', 'subjects'] if c in docs.columns), None)
    candidate_idx = np.arange(len(docs))

    if selected_categories and cat_col:
        mask = docs[cat_col].astype(str).apply(
            lambda x: any(coarse_category(sel).lower() in coarse_category(str(x)).lower() for sel in selected_categories)
        ).values
        filtered = np.where(mask)[0]
        if filtered.size > 0:
            candidate_idx = filtered
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
                'url': str(row.get('url', "")) if 'url' in row else "",
                'category': str(row.get('__primary_cat__', "")) if '__primary_cat__' in row else ""
            })

    results = sorted(results, key=lambda x: x['score'], reverse=True)[:MAX_DISPLAY]
    return results

# ---------- STREAMLIT UI ----------
def main():
    st.set_page_config(
        page_title="FlareSearch — Journal Finder",
        page_icon="🔍",
        layout="wide"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Load data
    try:
        docs = load_docs()
    except Exception as e:
        st.error(f"Failed to load journal data: {e}")
        return

    params = load_best_params()

    with st.spinner("Building TF-IDF index (first run only)..."):
        vectoriser, tfidf_matrix = build_tfidf_index(docs, params)

    # -- Hero banner --
    n_journals = len(docs)
    cat_col = next((c for c in ['__primary_cat__', 'category'] if c in docs.columns), None)
    n_cats = docs[cat_col].nunique() if cat_col else 0

    st.markdown(f"""
    <div class="hero">
        <h1>🔍 FlareSearch</h1>
        <p>Find the best journal for your manuscript — powered by Optuna-tuned TF-IDF</p>
    </div>
    """, unsafe_allow_html=True)

    # -- Stat cards --
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="num">{n_journals:,}</div>
            <div class="label">Journals Indexed</div>
        </div>
        <div class="stat-card">
            <div class="num">{n_cats}</div>
            <div class="label">Categories</div>
        </div>
        <div class="stat-card">
            <div class="num">88.84%</div>
            <div class="label">HitRate@1 Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="num">&lt;1s</div>
            <div class="label">Query Speed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -- Build coarse categories --
    if cat_col:
        coarse = docs[cat_col].astype(str).map(coarse_category)
        available_categories = sorted([c for c in coarse.unique() if c and c.strip()])
    else:
        available_categories = []

    # -- Input fields --
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### 📝 Manuscript Details")
        title = st.text_input("Title", placeholder="Enter your manuscript title...")
        abstract = st.text_area("Abstract", height=180,
                                placeholder="Paste your abstract here for more accurate matching...")

    with col_right:
        st.markdown("#### 🏷️ Category Filter (Optional)")
        if available_categories:
            selected_cats = st.multiselect(
                "Narrow by subject",
                options=available_categories
            )
            if not selected_cats:
                selected_cats = None
        else:
            selected_cats = None
        
        st.markdown("")
        st.markdown("")
        search_clicked = st.button("🔍 Find Matching Journals", use_container_width=True, type="primary")

    # -- Results --
    if search_clicked:
        if not title.strip() and not abstract.strip():
            st.warning("⚠️ Please enter a title or abstract to search.")
        else:
            with st.spinner("🔍 Searching journals..."):
                results = recommend(
                    title=title, abstract=abstract, docs=docs,
                    vectoriser=vectoriser, tfidf_matrix=tfidf_matrix,
                    selected_categories=selected_cats
                )

            if not results:
                st.info("No suitable journal found. Try clearing the category filter or expanding your abstract.")
            else:
                st.markdown(f"#### 📚 Top {len(results)} Matching Journals")

                for i, r in enumerate(results):

                    # Build URL link
                    url_html = ""
                    if r['url'] and r['url'].strip() and r['url'].startswith("http"):
                        url_html = f' · <a href="{r["url"]}" target="_blank" style="color:#e94560;text-decoration:none;">🔗 Visit Journal</a>'

                    # Publisher
                    pub_html = f" · 📰 {r['publisher']}" if r['publisher'] else ""

                    # Category badge
                    cat_html = ""
                    if r['category']:
                        cat_display = coarse_category(r['category'])
                        cat_html = f' <span style="background:rgba(233,69,96,0.15);color:#e94560;padding:2px 8px;border-radius:4px;font-size:0.75rem;">{cat_display}</span>'

                    # Scope
                    scope_html = ""
                    if r['subjects_preview']:
                        scope_html = f'<div class="result-scope">{r["subjects_preview"]}</div>'

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-title">{r['journal_title']}</span>
                            <span class="result-rank">#{i+1}</span>
                        </div>
                        <div class="score-text">
                            Score: {r['score']:.4f}{pub_html}{url_html} {cat_html}
                        </div>
                        {scope_html}
                    </div>
                    """, unsafe_allow_html=True)

    # -- Footer --
    st.markdown("""
    <div class="footer">
        FlareSearch v2.0 · TF-IDF Engine with Optuna-Tuned Hyperparameters · DOAJ Dataset<br>
        © 2026 Chin Hong An · Universiti Tunku Abdul Rahman
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
