import streamlit as st
import joblib
import numpy as np

# Load saved models
model = joblib.load("models/svm_sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Streamlit Page Settings
st.set_page_config(page_title="📈 Stock Market Sentiment Analysis", layout="centered")

st.title("📈 Stock Market Sentiment Predictor")
st.write("Analyze news headlines and predict sentiment (positive, neutral, negative).")

# Sidebar info
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app uses a **Linear SVM model** trained on the 
**Financial News Sentiment Dataset** to classify 
headlines into *positive*, *neutral*, or *negative*.
""")

# User input
headline = st.text_area("📝 Enter a financial news headline:", height=120)

if st.button("🔍 Predict Sentiment"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        # Convert text → vector
        X_input = vectorizer.transform([headline])

        # Predict
        pred = model.predict(X_input)[0]
        label = label_encoder.inverse_transform([pred])[0]

        # Display result
        if label == "positive":
            st.success(f"💹 **Positive** sentiment detected!")
        elif label == "negative":
            st.error(f"📉 **Negative** sentiment detected!")
        else:
            st.info(f"📊 **Neutral** sentiment detected!")
