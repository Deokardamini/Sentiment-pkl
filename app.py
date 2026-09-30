import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================================
# CUSTOM STYLING (CSS WITH SHADOW EFFECTS)
# ==========================================
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card wrapper with drop shadow */
    .css-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Title Styling */
    .title-text {
        color: #1e293b;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 8px;
        text-align: center;
    }
    
    .subtitle-text {
        color: #64748b;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 24px;
    }
    
    /* Custom Result Badges */
    .badge-positive {
        background-color: #dcfce7;
        color: #15803d;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-negative {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-neutral {
        background-color: #e2e8f0;
        color: #475569;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
",
    unsafe_allow_html=True,
)


# ==========================================
# RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_artifacts():
    """Load pickled TF-IDF Vectorizer and Model safely."""
    vectorizer_path = "vectorizer.pkl"
    model_path = "model.pkl"

    vectorizer = None
    model = None

    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return vectorizer, model


vectorizer, model = load_artifacts()

# ==========================================
# UI LAYOUT
# ==========================================
st.markdown('<div class="title-text">Sentiment Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Analyze customer feedback or text sentiment in real-time</div>',
    unsafe_allow_html=True,
)

# Card Container for Input
st.markdown('<div class="css-card">', unsafe_allow_html=True)
user_text = st.text_area("Input Text", placeholder="Type or paste your text here...", height=120)
analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Prediction Logic
if analyze_btn:
    if not user_text.strip():
        st.warning("Please enter valid text to analyze.")
    elif vectorizer is None:
        st.error("`vectorizer.pkl` file not found. Place it in the root directory.")
    else:
        # Transform input text
        X_vec = vectorizer.transform([user_text])

        # If a model exists, predict; otherwise, demonstrate vector output
        if model is not None:
            prediction = model.predict(X_vec)[0]

            # Probabilities (if model supports predict_proba)
            probs = (
                model.predict_proba(X_vec)[0]
                if hasattr(model, "predict_proba")
                else [0.5, 0.5]
            )

            # Categorical Format Display
            categories = ["Negative", "Positive"]
            category_dtype = pd.CategoricalDtype(categories=categories, ordered=True)
            
            # Map sentiment output into Pandas Categorical Format
            pred_label = categories[1] if prediction == 1 or str(prediction).lower() == "positive" else categories[0]
            cat_series = pd.Series([pred_label], dtype=category_dtype)

            # Results Card
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.subheader("Classification Summary")

            if cat_series[0] == "Positive":
                st.markdown('<span class="badge-positive">Positive Sentiment</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-negative">Negative Sentiment</span>', unsafe_allow_html=True)

            st.write("")
            st.markdown(f"**Categorical Value:** `{cat_series[0]}` (Dtype: `category`)")

            # Metrics display
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence Score", f"{max(probs)*100:.1f}%")
            with col2:
                st.metric("Extracted Features", f"{X_vec.nnz} tokens")

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Fallback if model pickle is missing
            st.info("Vectorizer processed successfully! Vector shape: " + str(X_vec.shape))
