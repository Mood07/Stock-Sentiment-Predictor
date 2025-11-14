import os
import joblib

def save_artifacts(model, vectorizer, encoder,
                   model_path="../models/svm_sentiment_model.pkl",
                   vectorizer_path="../models/tfidf_vectorizer.pkl",
                   encoder_path="../models/label_encoder.pkl"):
    os.makedirs("../models", exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(encoder, encoder_path)
    
    print("💾 Saved model + vectorizer + encoder!")
