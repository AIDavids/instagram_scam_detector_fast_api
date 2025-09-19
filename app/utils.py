import re
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# === Preprocessing (must match training) ===
def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", "", text)

def preprocess_text(text: str) -> str:
    t = str(text)
    t = remove_urls(t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.lower()
    return t

# === Helper functions for numeric features ===
def count_emojis(text: str) -> int:
    return sum(char in "😀😃😄😁😆😅😂🤣😊😍😘😎👍🔥💯✅📲💵🎁🎉" for char in text)

def punctuation_ratio(text: str) -> float:
    if len(text) == 0:
        return 0.0
    puncts = sum(char in "!?.," for char in text)
    return puncts / len(text)

def has_link(text: str) -> int:
    return int("http" in text or "www" in text)

# === Feature preparation (consistent with training) ===
def prepare_features(texts, vectorizer):
    # Preprocess first
    clean_texts = [preprocess_text(t) for t in texts]

    df = pd.DataFrame({"text": clean_texts})
    df["length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().apply(len)
    df["emoji_count"] = df["text"].apply(count_emojis)
    df["punct_ratio"] = df["text"].apply(punctuation_ratio)
    df["has_link"] = df["text"].apply(has_link)

    # TF-IDF features (use trained vectorizer)
    X_tfidf = vectorizer.transform(df["text"])

    # Numeric features
    numeric_feats = csr_matrix(df[["length","word_count","emoji_count","punct_ratio","has_link"]].values)

    # Combine into final feature matrix
    return hstack([X_tfidf, numeric_feats])


# --- Explanation helper (optional, small, safe) ---
def get_lr_explanation(text, vectorizer, lr_model, top_k=5):
    """
    Returns top_k terms that contributed most toward the positive (scam) class for a LogisticRegression.
    Requires: vectorizer (fitted TfidfVectorizer) and lr_model (fitted LogisticRegression with coef_).
    If not logistic or vocab mismatch, returns empty list.
    """
    try:
        # only works for linear models with coef_
        if not hasattr(lr_model, "coef_"):
            return []

        # preprocess same as prepare_features uses
        from scipy.sparse import csr_matrix
        cleaned = [preprocess_text(text)]
        tf = vectorizer.transform(cleaned)  # shape (1, n_features_tfidf)
        coef = lr_model.coef_[0]  # shape (n_total_features,)
        # coef corresponds to tfidf features first, then numeric features
        n_tfidf = vectorizer.transform([""]).shape[1]  # quick way to get tfidf dim
        tfidf_coefs = coef[:n_tfidf]

        # get nonzero tfidf indices for the text
        row = tf.tocsr()
        nz = row.indices  # indices of tfidf terms present
        if len(nz) == 0:
            return []

        # map indices to (term, score = coef * tfidf_value)
        feature_names = vectorizer.get_feature_names_out()
        contributions = []
        for idx in nz:
            term = feature_names[idx]
            tfidf_val = row.data[list(row.indices).index(idx)]
            score = tfidf_coefs[idx] * tfidf_val
            contributions.append((term, float(score)))

        # sort by contribution descending
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:top_k]
    except Exception:
        return []
