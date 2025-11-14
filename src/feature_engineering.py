from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(df, max_features=5000):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),
        stop_words='english'
    )
    X = tfidf.fit_transform(df["clean_sentence"])
    print(f"✅ TF-IDF shape: {X.shape}")
    return X, tfidf
