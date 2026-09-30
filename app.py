import os
import pickle
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="centered")

st.title("🎭 Sentiment Analysis App")

# Function to load pickle files safely
@st.cache_resource
def load_models():
    model_path = "sentiment.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, "File missing error"
        
    try:
        with open(model_path, "rb") as f1:
            model = pickle.load(f1)
        with open(vectorizer_path, "rb") as f2:
            vectorizer = pickle.load(f2)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

# Load Models
model, vectorizer, error = load_models()

if error == "File missing error":
    st.error("❌ `sentiment.pkl` ya `vectorizer.pkl` file repository mein missing hai.")
elif error:
    st.error(f"❌ Pickle file load nahi hui: {error}")
else:
    # User Input Field
    text_input = st.text_area("Analysis ke liye text yahan likhein:", placeholder="Type your text here...")

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("⚠️ Kripya pehle text enter karein.")
        else:
            try:
                # Text Transform & Prediction
                data = vectorizer.transform([text_input])
                prediction = model.predict(data)[0]
                
                # Output Display
                st.success(f"**Predicted Sentiment:** {prediction}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
