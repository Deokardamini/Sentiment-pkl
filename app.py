import streamlit as st
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="centered"
)

# Custom CSS for UI with Shadow Effects and Modern Styling
st.markdown("""
    <style>
    /* Main Background Styling */
    .main {
        background-color: #f4f7f6;
    }
    
    /* Title Card with Soft Shadow */
    .title-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-bottom: 25px;
    }
    .title-card h1 {
        color: #2c3e50;
        margin: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title-card p {
        color: #7f8c8d;
        margin-top: 5px;
        font-size: 16px;
    }

    /* Result Card Styling with Elevate Shadow */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.12);
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    
    .pos-card {
        background-color: #e8f8f5;
        color: #27ae60;
        border: 2px solid #27ae60;
    }
    
    .neg-card {
        background-color: #fadbd8;
        color: #c0392b;
        border: 2px solid #c0392b;
    }
    
    .neu-card {
        background-color: #fcf3cf;
        color: #f39c12;
        border: 2px solid #f39c12;
    }

    /* Streamlit Text Area Customization */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #dcdde1 !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.04) !important;
    }

    /* Streamlit Button Customization */
    .stButton>button {
        width: 100%;
        border-radius: 10px !important;
        background-color: #3498db !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px 0px !important;
        box-shadow: 0px 4px 12px rgba(52, 152, 219, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #2980b9 !important;
        box-shadow: 0px 6px 15px rgba(41, 128, 185, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="title-card">
        <h1>🎭 Sentiment Analysis Tool</h1>
        <p>Aapka text daalein aur dekhein Sentiment Category (Positive / Negative / Neutral)</p>
    </div>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    model_path = 'sentiment.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    model = None
    vectorizer = None
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
    return model, vectorizer

model, vectorizer = load_models()

# Form Input Section
user_input = st.text_area("Analysis ke liye Text yahan likhein:", height=130, placeholder="Type your text here...")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Kripya pehle kuch text enter karein!")
    elif model is None or vectorizer is None:
        st.error("Error: `sentiment.pkl` ya `vectorizer.pkl` file nahi mili! File directory check karein.")
    else:
        # Vectorize Input Text
        text_vectorized = vectorizer.transform([user_input])
        
        # Predict Class/Category
        prediction = model.predict(text_vectorized)[0]
        
        # Categorical Column Format Mapping
        # Mapping values based on string or integer outputs from model
        category_map = {
            'negative': 'Negative 😡',
            'positive': 'Positive 😊',
            'neutral': 'Neutral 😐',
            0: 'Negative 😡',
            1: 'Positive 😊',
            2: 'Neutral 😐'
        }
        
        result_category = category_map.get(prediction, str(prediction))
        
        # Display Result with Custom Shadow Box
        if "Positive" in result_category or prediction in ['positive', 1]:
            st.markdown(f'<div class="result-card pos-card">Predicted Category: {result_category}</div>', unsafe_allow_html=True)
        elif "Negative" in result_category or prediction in ['negative', 0]:
            st.markdown(f'<div class="result-card neg-card">Predicted Category: {result_category}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card neu-card">Predicted Category: {result_category}</div>', unsafe_allow_html=True)
