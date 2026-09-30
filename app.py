import os
import pickle
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="centered")

st.title("🎭 Sentiment Analysis App")

# Safe Model & Vectorizer Loader
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
    # Text Input Form
    text_input = st.text_area("Analysis ke liye text yahan likhein:", placeholder="Type your text here...")

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("⚠️ Kripya pehle text enter karein.")
        else:
            try:
                # Transform & Predict
                data = vectorizer.transform([text_input])
                prediction = model.predict(data)[0]
                
                # Display Result
                st.success(f"**Predicted Sentiment:** {prediction}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")import streamlit as st
import pickle
import os

# Page Config
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="centered")

# Title UI
st.title("🎭 Sentiment Analysis App")

# Function to load model & vectorizer safely
@st.cache_resource
def load_files():
    try:
        with open("sentiment.pkl", "rb") as f1:
            model = pickle.load(f1)
        with open("vectorizer.pkl", "rb") as f2:
            vectorizer = pickle.load(f2)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

model, vectorizer, error = load_files()

if error:
    st.error(f"❌ Pickle file load karne me issue aaya: {error}")
    st.info("💡 Make sure `sentiment.pkl` aur `vectorizer.pkl` dono files same directory me hain.")
else:
    # User Input
    text_input = st.text_area("Enter text for sentiment analysis:", placeholder="Type here...")

    if st.button("Predict"):
        if not text_input.strip():
            st.warning("⚠️ Kripya kuch text enter karein.")
        else:
            # Transform and Predict
            data = vectorizer.transform([text_input])
            prediction = model.predict(data)[0]
            
            # Display Output
            st.success(f"**Predicted Sentiment:** {prediction}")import streamlit as st
import pickle
import os

# 1. Page Setup
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="centered"
)

# 2. Custom Styling (CSS)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-bottom: 20px;
    }
    .result-card {
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        margin-top: 20px;
    }
    .pos-card {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .neg-card {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .neu-card {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header UI
st.markdown("""
    <div class="title-card">
        <h1>🎭 Sentiment Analysis Tool</h1>
        <p>Text enter karke Sentiment Category detect karein</p>
    </div>
""", unsafe_allow_html=True)

# 4. Safe Pickle Model Loading Function
@st.cache_resource
def load_assets():
    model_path = "sentiment.pkl"
    vectorizer_path = "vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, "File missing error"

    try:
        with open(model_path, "rb") as f_model:
            model = pickle.load(f_model)
        with open(vectorizer_path, "rb") as f_vec:
            vectorizer = pickle.load(f_vec)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

model, vectorizer, load_error = load_assets()

# Display error if pickle loading failed
if load_error:
    if load_error == "File missing error":
        st.error("❌ `sentiment.pkl` ya `vectorizer.pkl` file missing hai. Same folder me add karein.")
    else:
        st.error(f"❌ Pickle file load karne me problem aayi: {load_error}")
        st.info("💡 Tip: Scikit-learn ka version check karein (jo model train karte waqt tha, wahi environment me hona chahiye).")

# 5. User Input Form
user_input = st.text_area(
    "Analysis ke liye text yahan likhein:",
    height=120,
    placeholder="e.g., The product quality is amazing! I really loved it."
)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Kripya pehle kuch text likhein.")
    elif model is None or vectorizer is None:
        st.error("❌ Models load nahi ho paye. Process execution cancelled.")
    else:
        try:
            # Step 1: Text Transformation
            vec_input = vectorizer.transform([user_input])
            
            # Step 2: Prediction
            raw_prediction = model.predict(vec_input)[0]

            # Step 3: Mapping Outputs (Supports String, 0/1 Binary, and 0/1/2 Multiclass)
            clean_pred = str(raw_prediction).lower().strip()

            if clean_pred in ['1', 'positive', 'pos']:
                sentiment = "Positive 😊"
                card_class = "pos-card"
            elif clean_pred in ['0', 'negative', 'neg']:
                sentiment = "Negative 😡"
                card_class = "neg-card"
            elif clean_pred in ['2', 'neutral', 'neu']:
                sentiment = "Neutral 😐"
                card_class = "neu-card"
            else:
                sentiment = f"Category: {raw_prediction}"
                card_class = "neu-card"

            # Step 4: Display Output
            st.markdown(
                f'<div class="result-card {card_class}">Predicted Sentiment: {sentiment}</div>',
                unsafe_allow_html=True
            )

        except Exception as pred_err:
            st.error(f"Prediction Error: {pred_err}")
 
 
