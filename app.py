import streamlit as st
import joblib
import requests
import os
import sys

# Force CPU usage BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # Import torch AFTER setting CUDA_VISIBLE_DEVICES

# Set default device to CPU
torch.set_default_device('cpu')  # Add this line

st.write(f"Python version: {sys.version}")
st.write(f"Torch version: {torch.__version__}")
st.write(f"CUDA is available: {torch.cuda.is_available()}")

MODEL_FILE = "band_predictor_model_torch.pth"
MODEL_URL = "https://huggingface.co/swahar2/AI-Tutor/resolve/main/band_predictor_model.pkl"

@st.cache_resource
def load_model():
    try:
        # Load the entire model using torch.load, mapping to CPU
        model = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
        st.success("Model loaded successfully using torch.load!")
        return model
    except Exception as e:
        st.error(f"Error loading the model with torch.load: {e}")
        return None

model = load_model()

st.title("IELTS Band Predictor")

if model:
    essay = st.text_area("Enter Essay Text")
    prompt = st.text_area("Enter Prompt Text")
    task = st.text_area("Enter Task Achievement Text")
    coherence = st.text_area("Enter Coherence and Cohesion Text")
    vocab = st.text_area("Enter Lexical Resource Text")
    grammar = st.text_area("Enter Grammatical Range and Accuracy Text")
    task = st.slider("Task Achievement", 0.0, 9.0, 6.0)
    coherence = st.slider("Coherence & Cohesion", 0.0, 9.0, 6.0)
    vocab = st.slider("Vocabulary", 0.0, 9.0, 6.0)
    grammar = st.slider("Grammatical Range and Accuracy", 0.0, 9.0, 6.0)

    if st.button("Predict Band"):
        try:
            # Adapt this part to match what your model expects
            prediction = model(essay, task, coherence, vocab, grammar)  # Replace with your actual prediction
            st.success(f"Predicted Band Score: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.warning("Model not loaded. Prediction is unavailable.")
