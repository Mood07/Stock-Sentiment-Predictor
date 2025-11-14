# src/main.py — Full Pipeline for Stock Sentiment Predictor

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import modules
from data_ingestion import load_dataset
from data_preprocessing import preprocess_dataset
from feature_engineering import create_tfidf_features
from model_training import train_svm_model
from evaluation import evaluate_model
from utils import save_artifacts

# ============================================================
# PATH CONFIG (IMPORTANT)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "all-data.csv"

print("🚀 Starting Stock Sentiment Prediction Pipeline...")
print("Reading data from:", DATA_PATH)

# ============================================================
# 1️⃣ DATA INGESTION
# ============================================================
df = load_dataset(str(DATA_PATH))

# ============================================================
# 2️⃣ DATA PREPROCESSING
# ============================================================
df = preprocess_dataset(df)

# ============================================================
# 3️⃣ LABEL ENCODING
# ============================================================
label_encoder = LabelEncoder()
df["sentiment_label"] = label_encoder.fit_transform(df["sentiment"])
labels = label_encoder.classes_

# ============================================================
# 4️⃣ FEATURE EXTRACTION (TF-IDF)
# ============================================================
X, tfidf_vectorizer = create_tfidf_features(df)
y = df["sentiment_label"]

# ============================================================
# 5️⃣ MODEL TRAINING (SVM)
# ============================================================
model, X_train, X_test, y_train, y_test = train_svm_model(X, y)

# ============================================================
# 6️⃣ EVALUATION
# ============================================================
evaluate_model(
    model,
    X_test,
    y_test,
    labels,
    save_path=str(BASE_DIR / "assets" / "visuals" / "confusion_matrix_svm_pipeline.png")
)

# ============================================================
# 7️⃣ SAVE ARTIFACTS
# ============================================================
save_artifacts(
    model,
    tfidf_vectorizer,
    label_encoder,
    model_path=str(BASE_DIR / "models" / "svm_sentiment_model.pkl"),
    vectorizer_path=str(BASE_DIR / "models" / "tfidf_vectorizer.pkl"),
    encoder_path=str(BASE_DIR / "models" / "label_encoder.pkl")
)

print("\n🎉 Pipeline completed successfully!")
