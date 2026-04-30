# utils.py
import re
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# make sure to download once in environment:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def nltk_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # remove URLs
    text = re.sub(r'http\S+', ' ', text)
    # remove weird characters
    text = re.sub(r'\s+', ' ', text)
    # tokenize
    tokens = word_tokenize(text)
    cleaned = []
    for t in tokens:
        # keep only alphabetic tokens
        if not t.isalpha():
            continue
        if t in STOPWORDS:
            continue
        cleaned.append(LEMMATIZER.lemmatize(t))
    return " ".join(cleaned)

def cos_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: n x d, B: m x d -> returns n x m similarity matrix
    # use sklearn pairwise if needed; here numpy-based but safe:
    return cosine_similarity(A, B)

def precision_at_k_batch(topk_indices, true_labels_per_query, K):
    """
    topk_indices: list (n_queries) of lists of predicted labels (length K)
    true_labels_per_query: list (n_queries) of the true label or list of true labels
    returns: average precision@K
    """
    n = len(topk_indices)
    total = 0.0
    for preds, true in zip(topk_indices, true_labels_per_query):
        if isinstance(true, (list, set, tuple)):
            true_set = set(true)
        else:
            true_set = {true}
        hits = len(true_set.intersection(set(preds[:K])))
        total += hits / K
    return total / n


# ---------- Shared metrics for FYP2 tuning & evaluation ----------

def hit_rate_at_k(ranked_cats, true_cat, k):
    """Check if the true category appears in the top-k ranked categories.
    Returns 1 if hit, 0 otherwise."""
    return 1 if true_cat in ranked_cats[:k] else 0


def mean_hit_rate_at_k(all_ranked_cats, all_true_cats, k):
    """Mean HitRate@K across all queries."""
    if len(all_ranked_cats) == 0:
        return 0.0
    total = sum(hit_rate_at_k(rc, tc, k) for rc, tc in zip(all_ranked_cats, all_true_cats))
    return total / len(all_ranked_cats)


def average_precision_at_k(ranked_cats, true_cat, k):
    """Average Precision at K for a single query (binary relevance: same category = relevant)."""
    hits = 0
    sum_prec = 0.0
    for i, cat in enumerate(ranked_cats[:k]):
        if cat == true_cat:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(k, 1) if hits == 0 else sum_prec / hits


def map_at_k(all_ranked_cats, all_true_cats, k):
    """Mean Average Precision at K across all queries."""
    if len(all_ranked_cats) == 0:
        return 0.0
    total = sum(average_precision_at_k(rc, tc, k) for rc, tc in zip(all_ranked_cats, all_true_cats))
    return total / len(all_ranked_cats)


def dcg_at_k(ranked_cats, true_cat, k):
    """Discounted Cumulative Gain at K (binary relevance)."""
    dcg = 0.0
    for i, cat in enumerate(ranked_cats[:k]):
        rel = 1.0 if cat == true_cat else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def ideal_dcg_at_k(n_relevant, k):
    """Ideal DCG at K: assumes all relevant items are ranked first."""
    idcg = 0.0
    for i in range(min(n_relevant, k)):
        idcg += 1.0 / np.log2(i + 2)
    return idcg


def ndcg_at_k(all_ranked_cats, all_true_cats, k):
    """Normalized DCG at K across all queries.
    Computes ideal DCG based on actual number of relevant items in the ranked list."""
    if len(all_ranked_cats) == 0:
        return 0.0
    total = 0.0
    for rc, tc in zip(all_ranked_cats, all_true_cats):
        actual_dcg = dcg_at_k(rc, tc, k)
        # Count how many relevant items exist in the full ranked list
        n_relevant = sum(1 for cat in rc if cat == tc)
        idcg = ideal_dcg_at_k(max(n_relevant, 1), k)
        total += actual_dcg / idcg if idcg > 0 else 0.0
    return total / len(all_ranked_cats)

