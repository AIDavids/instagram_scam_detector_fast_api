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
