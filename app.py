import os
import joblib
import streamlit as st

@st.cache_resource
def load_assets(model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """Load pickled TF-IDF Vectorizer and Model safely."""
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(f"Missing file: Ensure '{model_path}' and '{vectorizer_path}' exist.")
        return None, None

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading pkl files: {e}")
        return None, None

# Load assets
model, vectorizer = load_assets()

if model and vectorizer:
    st.title("Sentiment Analysis")
    user_input = st.text_area("Enter text to analyze:")
    
    if st.button("Predict"):
        if user_input.strip():
            transformed = vectorizer.transform([user_input])
            prediction = model.predict(transformed)
            st.success(f"Sentiment: {prediction[0]}")
        else:
            st.warning("Please enter text before analyzing.")
