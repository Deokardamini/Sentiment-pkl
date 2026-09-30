import os
import pickle
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="centered")

st.title("🎭 Sentiment Analysis App")

# Current Directory ka Path Resolve Karein
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Safe Loading Function
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None, "File missing error"
        
    try:
        with open(MODEL_PATH, "rb") as f1:
            model = pickle.load(f1)
        with open(VECTORIZER_PATH, "rb") as f2:
            vectorizer = pickle.load(f2)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

model, vectorizer, error = load_models()

if error == "File missing error":
    st.error("❌ `sentiment.pkl` ya `vectorizer.pkl` file repository mein nahi mili!")
    st.info("💡 GitHub repository ke root folder mein exact naam se files upload karein.")
elif error:
    st.error(f"❌ Pickle file load karne mein issue aaya: {error}")
else:
    # Text Input Form
    text_input = st.text_area("Analysis ke liye text yahan likhein:", placeholder="Type your text here...")

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("⚠️ Kripya pehle text enter karein.")
        else:
            try:
                data = vectorizer.transform([text_input])
                prediction = model.predict(data)[0]
                st.success(f"**Predicted Sentiment:** {prediction}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")import os
import pickle
import streamlit as st

# Page Setup
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="centered")

st.title("🎭 Sentiment Analysis App")

# Current directory path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Safe Loading Function
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None, "File missing error"
        
    try:
        with open(MODEL_PATH, "rb") as f1:
            model = pickle.load(f1)
        with open(VECTORIZER_PATH, "rb") as f2:
            vectorizer = pickle.load(f2)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

# Load Models
model, vectorizer, error = load_models()

if error == "File missing error":
    st.error("❌ `sentiment.pkl` ya `vectorizer.pkl` file repository mein nahi mili!")
    st.warning("💡 Solution: Apni GitHub repository ke root folder mein `sentiment.pkl` aur `vectorizer.pkl` files upload karein.")
elif error:
    st.error(f"❌ Pickle file load karne mein problem aayi: {error}")
else:
    # Text Input Form
    text_input = st.text_area("Analysis ke liye text yahan likhein:", placeholder="Type your text here...")

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("⚠️ Kripya pehle text enter karein.")
        else:
            try:
                # Vectorization & Prediction
                data = vectorizer.transform([text_input])
                prediction = model.predict(data)[0]
                
                # Display Result
                st.success(f"**Predicted Sentiment:** {prediction}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
