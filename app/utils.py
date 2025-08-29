import pandas as pd
from scipy.sparse import hstack, csr_matrix

# === Helper functions for numeric features ===
def count_emojis(text):
    return sum(char in "😀😃😄😁😆😅😂🤣😊😍😘😎👍🔥💯✅📲💵🎁🎉" for char in text)

def punctuation_ratio(text):
    if len(text) == 0:
        return 0
    puncts = sum(char in "!?.," for char in text)
    return puncts / len(text)

def has_link(text):
    return int("http" in text or "www" in text)

# === Feature preparation ===
def prepare_features(texts, vectorizer):
    df = pd.DataFrame({"text": texts})
    df["length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().apply(len)
    df["emoji_count"] = df["text"].apply(count_emojis)
    df["punct_ratio"] = df["text"].apply(punctuation_ratio)
    df["has_link"] = df["text"].apply(has_link)

    # TF-IDF features
    X_tfidf = vectorizer.transform(df["text"])

    # Numeric features
    numeric_feats = csr_matrix(df[["length","word_count","emoji_count","punct_ratio","has_link"]].values)

    # Combine
    return hstack([X_tfidf, numeric_feats])
