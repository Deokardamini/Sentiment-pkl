import pickle
import numpy as np
import streamlit as st

# Page title & description
st.set_page_config(page_title="Text Classification Model Deployment", layout="centered")
st.title("Naive Bayes Sentiment / Text Classifier")
st.write("Enter text or feature values below to get predictions from the trained model.")

# Load the saved pickle model
@st.cache_resource
def load_model(file_path="model.pkl"):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    return model

# Load model
try:
    model = load_model("model.pkl")
    st.success("Model successfully loaded!")
except FileNotFoundError:
    st.error("Model file `model.pkl` not found. Please ensure the pickle file is saved in the working directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# User input form
st.subheader("Make a Prediction")

# Text input or feature array input based on model classes ('negative', 'positive')
text_input = st.text_area("Enter text to classify:", placeholder="Type your sentence here...")

if st.button("Predict Sentiment"):
    if not text_input.strip():
        st.warning("Please enter valid text before predicting.")
    else:
        # Note: If your model requires a Vectorizer (e.g., CountVectorizer / TfidfVectorizer),
        # ensure it is applied here or passed alongside the pickle pipeline.
        try:
            # Example direct prediction workflow
            prediction = model.predict([text_input])[0]
            
            # Predict probabilities if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text_input])[0]
                classes = model.classes_
                
                st.markdown(f"### Prediction Result: **{prediction.upper()}**")
                
                # Show probabilities
                st.write("Confidence scores:")
                for cls, prob in zip(classes, probs):
                    st.progress(float(prob), text=f"{cls}: {prob*100:.2f}%")
            else:
                st.markdown(f"### Prediction: **{prediction}**")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("If your model expects vectorized numerical features instead of raw text strings, load and apply your TF-IDF or CountVectorizer before `model.predict()`.")
