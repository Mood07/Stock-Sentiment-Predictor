import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Performs text cleaning: lowercase, punctuation removal,
    numbers removal, stopword removal.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in STOPWORDS)
    return text

def preprocess_dataset(df):
    df["clean_sentence"] = df["sentence"].apply(clean_text)
    print("🧹 Text cleaning completed!")
    return df
