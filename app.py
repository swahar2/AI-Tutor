import streamlit as st
import joblib
import requests
import os

# Force CPU usage BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import sys 

st.write(f"Python version: {sys.version}")
st.write(f"Torch version: {torch.__version__}")

MODEL_FILE = "band_predictor_model_torch.pth"
MODEL_URL = "https://huggingface.co/swahar2/AI-Tutor/resolve/main/band_predictor_model.pkl"

def download_file(url, filename):
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    r = requests.get(url, headers=headers, stream=True)
    print(f"Status: {r.status_code}")
    r.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
              f.write(chunk)

# Download if not present
if not os.path.exists(MODEL_FILE):
    st.info("Downloading model…")
    download_file(MODEL_URL, MODEL_FILE)
    st.success("Model ready!")

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""    

# Load the model
model = torch.load(MODEL_FILE, map_location=torch.device('cpu'), weights_only=False)

st.title("IELTS Band Predictor")

essay = st.text_area("Enter Essay Text")
task = st.slider("Task Achievement", 0.0, 9.0, 6.0)
coherence = st.slider("Coherence & Cohesion", 0.0, 9.0, 6.0)
vocab = st.slider("Vocabulary", 0.0, 9.0, 6.0)
grammar = st.slider("Grammar", 0.0, 9.0, 6.0)

if st.button("Predict Band"):
    df = pd.DataFrame([{
        'essay_text': essay,
        'task_achievement_score': task,
        'coherence_score': coherence,
        'vocab_score': vocab,
        'grammar_score': grammar
    }])
    band = model.predict(df)[0]
    st.success(f"Predicted Band Score: {round(band,2)}")
