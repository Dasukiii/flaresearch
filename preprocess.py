# preprocess.py
"""
Preprocess script adapted for DOAJ-style journal CSV.
- Detects title, subjects and keywords columns robustly (case-insensitive / substring match)
- Builds a representative 'text' per row by concatenating title + subjects + keywords
- Cleans text using nltk_clean from utils.py
- Extracts a primary category token (first subject / keyword)
- Splits dataset into train/test (attempts stratified split when possible)
- Saves train/test/combined normalized CSVs with out_prefix
"""

import pandas as pd
import argparse
import os
import nltk
from utils import nltk_clean

# Ensure NLTK resources are present (download if missing)
for res in ('punkt', 'stopwords', 'wordnet'):
    try:
        nltk.data.find(f'tokenizers/{res}') if res == 'punkt' else nltk.data.find(f'corpora/{res}')
    except LookupError:
        print(f"Downloading NLTK resource: {res}")
        nltk.download(res)

def find_col_by_candidates(cols_lower, candidates):
    """Return first column name matching any candidate exactly or by substring (case-insensitive)."""
    # exact match first
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cols_lower.index(cand.lower())][1]  # stored as (lowername, originalname)
    # substring match fallback
    for cand in candidates:
        for lowername, orig in cols_lower:
            if cand.lower() in lowername:
                return orig
    return None

def build_cols_lower(columns):
    """Return list of tuples (lowername, originalname) for search convenience."""
    return [(c.lower(), c) for c in columns]

def primary_token_from_field(val):
    """Return the first meaningful token from a multi-valued field (split on common separators)."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    for sep in (';', ',', '|', '/', ' - '):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if parts:
                return parts[0]
    # if no separator, return the full string or the first word as fallback
    return s if len(s.split()) <= 4 else " ".join(s.split()[:4])

def main(infile, out_prefix, test_size=0.2, random_state=42):
    print("Loading:", infile)
    df = pd.read_csv(infile, dtype=str, encoding='utf-8', low_memory=False)

    # Prepare case-insensitive lookup
    cols_lower = build_cols_lower(df.columns.tolist())

    # Candidate name lists to find appropriate columns (common variants)
    title_candidates = ['journal title', 'title', 'paper title', 'name']
    subjects_candidates = ['subjects', 'subject', 'lcc codes', 'scopus categories', 'discipline', 'subject area']
    keywords_candidates = ['keywords', 'key words', 'keyword']
    publisher_candidates = ['publisher', 'publisher name']
    url_candidates = ['journal url', 'url', 'doi', 'link']
    issn_candidates = ['issn', 'eissn', 'print issn', 'electronic issn']
    open_access_candidates = ['open access', 'oa', 'is_oa']
    license_candidates = ['license', 'licence']
    language_candidates = ['language', 'languages']
    country_candidates = ['country', 'publisher country', 'country of publication']
    id_candidates = ['journal id', 'id', 'doaj id', 'source id']

    title_col = find_col_by_candidates(cols_lower, title_candidates)
    subjects_col = find_col_by_candidates(cols_lower, subjects_candidates)
    keywords_col = find_col_by_candidates(cols_lower, keywords_candidates)
    publisher_col = find_col_by_candidates(cols_lower, publisher_candidates)
    url_col = find_col_by_candidates(cols_lower, url_candidates)
    issn_col = find_col_by_candidates(cols_lower, issn_candidates)
    oa_col = find_col_by_candidates(cols_lower, open_access_candidates)
    license_col = find_col_by_candidates(cols_lower, license_candidates)
    lang_col = find_col_by_candidates(cols_lower, language_candidates)
    country_col = find_col_by_candidates(cols_lower, country_candidates)
    id_col = find_col_by_candidates(cols_lower, id_candidates)

    print("Detected columns:")
    print("  title_col   ->", title_col)
    print("  subjects_col->", subjects_col)
    print("  keywords_col->", keywords_col)
    print("  publisher_col->", publisher_col)
    print("  url_col     ->", url_col)
    print("  issn_col    ->", issn_col)
    print("  oa_col      ->", oa_col)
    print("  license_col ->", license_col)
    print("  lang_col    ->", lang_col)
    print("  country_col ->", country_col)
    print("  id_col      ->", id_col)

    # Ensure at least a title column exists
    if title_col is None:
        raise ValueError("No title-like column found in CSV. Available columns: " + ", ".join(df.columns))

    # Create canonical columns
    df['title'] = df[title_col].fillna("").astype(str)

    # subjects and keywords may be missing — treat as empty string
    if subjects_col is not None:
        df['subjects_raw'] = df[subjects_col].fillna("").astype(str)
    else:
        df['subjects_raw'] = ""

    if keywords_col is not None:
        df['keywords_raw'] = df[keywords_col].fillna("").astype(str)
    else:
        df['keywords_raw'] = ""

    # optional metadata (if present)
    df['publisher'] = df[publisher_col].fillna("").astype(str) if publisher_col in df.columns else ""
    df['url'] = df[url_col].fillna("").astype(str) if url_col in df.columns else ""
    df['doi'] = df[url_col].fillna("").astype(str) if (url_col and 'doi' in (url_col.lower())) else df['url']  # best effort
    df['issn'] = df[issn_col].fillna("").astype(str) if issn_col in df.columns else ""
    df['open_access'] = df[oa_col].fillna("").astype(str) if oa_col in df.columns else ""
    df['license'] = df[license_col].fillna("").astype(str) if license_col in df.columns else ""
    df['language'] = df[lang_col].fillna("").astype(str) if lang_col in df.columns else ""
    df['country'] = df[country_col].fillna("").astype(str) if country_col in df.columns else ""
    df['journal_id'] = df[id_col].fillna("").astype(str) if id_col in df.columns else ""

    # create category from primary subjects or keywords (priority: subjects -> keywords)
    df['category'] = df['subjects_raw'].apply(lambda x: primary_token_from_field(x) if x and x.strip() else "")
    # fallback to keywords if category still empty
    df.loc[df['category'].str.strip() == "", 'category'] = df.loc[df['category'].str.strip() == "", 'keywords_raw'].apply(lambda x: primary_token_from_field(x) if x and x.strip() else "")

    # build the combined text that will be embedded:
    # include title + subjects + keywords + publisher (if available)
    def make_text(row):
        parts = [row.get('title','')]
        if row.get('subjects_raw'):
            parts.append(row['subjects_raw'])
        if row.get('keywords_raw'):
            parts.append(row['keywords_raw'])
        if publisher_col and pd.notna(row.get(publisher_col, "")):
            parts.append(str(row.get(publisher_col,"")))
        return " ".join([p for p in parts if p and str(p).strip()])

    df['text_raw'] = df.apply(make_text, axis=1)

    # Clean using NLTK pipeline (nltk_clean)
    print("Cleaning text fields (this may take some time)...")
    df['title_clean'] = df['title'].apply(nltk_clean)
    df['subjects_clean'] = df['subjects_raw'].apply(nltk_clean)
    df['keywords_clean'] = df['keywords_raw'].apply(nltk_clean)
    df['text'] = (df['title_clean'] + " " + df['subjects_clean'] + " " + df['keywords_clean']).str.strip()

    # create __primary_cat__ (first subject token) for easy filtering in the app
    df['__primary_cat__'] = df['category'].apply(lambda x: primary_token_from_field(x))

    # Drop rows where 'text' ends up empty after cleaning
    before = len(df)
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before-after} empty-text rows; {after} remain.")

    # Shuffle (deterministic) then split into train/test
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Prepare stratify if possible (enough samples per class)
    stratify_col = df['__primary_cat__'].replace("", pd.NA)
    try:
        # only stratify if every class has at least 2 members
        vc = stratify_col.value_counts(dropna=True)
        if len(vc) > 1 and (vc >= 2).all():
            train_df = df.iloc[int(len(df)*test_size):].reset_index(drop=True)
            test_df = df.iloc[:int(len(df)*test_size)].reset_index(drop=True)
        else:
            # fallback no stratify
            split_idx = int(len(df) * test_size)
            test_df = df.iloc[:split_idx].reset_index(drop=True)
            train_df = df.iloc[split_idx:].reset_index(drop=True)
    except Exception:
        split_idx = int(len(df) * test_size)
        test_df = df.iloc[:split_idx].reset_index(drop=True)
        train_df = df.iloc[split_idx:].reset_index(drop=True)

    # Save outputs (ONLY the relevant canonical columns + extra useful metadata)
    train_path = f"{out_prefix}_train.csv"
    test_path = f"{out_prefix}_test.csv"
    combined_path = f"{out_prefix}_normalized.csv"

    # Expanded canonical columns to keep (recommended)
    keep_cols = [
        'title', 'title_clean',
        'subjects_raw', 'subjects_clean',
        'keywords_raw', 'keywords_clean',
        'category', '__primary_cat__',
        'text_raw', 'text',
        # extra metadata for UI / provenance / evaluation
        'publisher', 'url', 'doi', 'issn', 'open_access', 'license', 'language', 'country', 'journal_id'
    ]

    # Ensure columns exist in each dataframe; if missing, create as empty strings to avoid KeyError
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
        if c not in train_df.columns:
            train_df[c] = ""
        if c not in test_df.columns:
            test_df[c] = ""

    # Reorder to keep_cols (only these columns)
    train_df = train_df[keep_cols].copy()
    test_df = test_df[keep_cols].copy()
    combined_df = df[keep_cols].copy()

    # Save CSVs
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    combined_df.to_csv(combined_path, index=False)

    print("Saved:", train_path, test_path, combined_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="input raw CSV path")
    parser.add_argument("--out_prefix", default="", help="prefix for outputs")
    parser.add_argument("--test_size", type=float, default=0.2, help="fraction for test set (0-1)")
    parser.add_argument("--random_state", type=int, default=42, help="random seed")
    args = parser.parse_args()
    main(args.infile, args.out_prefix, test_size=args.test_size, random_state=args.random_state)
