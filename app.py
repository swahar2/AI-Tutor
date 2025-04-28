import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
from transformers import AutoTokenizer, AutoModel
import torch

# Define Hugging Face Repository Details
repo_id = "swahar2/AI-Tutor"  # Replace with your repo if different
filename = "xgboost_best_model.pkl"  # The XGBoost model file

# Function to load XGBoost model
@st.cache_resource
def load_model(repo_id, filename):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = joblib.load(model_path)
    return model

# Function to lazily load RoBERTa
@st.cache_resource
def load_roberta_model():
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base')
    return tokenizer, model

# Load XGBoost model
model = load_model(repo_id, filename)

# Determine device (keep it CPU to save memory)
device = torch.device('cpu')
st.write(f"Using device: {device}")

# Streamlit app title
st.title("Band Score Prediction App 🎓")
st.write("This app predicts the IELTS band score based on your essay, prompt, and evaluation scores.")

# Input fields
st.header("Enter the Essay and Prompt")
essay_input = st.text_area("Essay:", "Write your essay here...")
prompt_input = st.text_area("Prompt:", "Write the essay prompt here...")

st.header("Enter the Evaluation Scores")
task_achievement_score = st.number_input(
    "Task Achievement Score (1-9):", min_value=1.0, max_value=9.0, value=6.0, step=0.5
)
coherence_score = st.number_input(
    "Coherence and Cohesion Score (1-9):", min_value=1.0, max_value=9.0, value=6.0, step=0.5
)
lexical_score = st.number_input(
    "Lexical Resource Score (1-9):", min_value=1.0, max_value=9.0, value=6.0, step=0.5
)
grammatical_score = st.number_input(
    "Grammatical Range and Accuracy Score (1-9):", min_value=1.0, max_value=9.0, value=6.0, step=0.5
)

# Function to generate embeddings
def get_embeddings(text, tokenizer, roberta_model):
    if not text.strip():
        return np.zeros((768,))
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    
    embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    
    return embeddings

# Predict button
if st.button("Predict Band Score"):
    if not essay_input.strip():
        st.error("Essay input is empty. Please provide a valid essay.")
        st.stop()

    if not prompt_input.strip():
        st.error("Prompt input is empty. Please provide a valid prompt.")
        st.stop()

    # Lazy load RoBERTa
    tokenizer, roberta_model = load_roberta_model()
    roberta_model = roberta_model.to(device)

    # Generate embeddings
    essay_embedding = get_embeddings(essay_input, tokenizer, roberta_model)
    prompt_embedding = get_embeddings(prompt_input, tokenizer, roberta_model)

    # Reshape embeddings
    essay_embedding = essay_embedding.reshape(1, -1)
    prompt_embedding = prompt_embedding.reshape(1, -1)

    # Combine features
    numeric_features = np.array([[task_achievement_score, coherence_score, lexical_score, grammatical_score]])
    input_features = np.hstack((essay_embedding, prompt_embedding, numeric_features))

    # Make prediction
    prediction = model.predict(input_features)[0]
    prediction = np.clip(prediction, 1, 9)

    st.subheader(f"Predicted Band Score: {prediction:.2f}")
