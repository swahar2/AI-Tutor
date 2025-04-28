import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
from transformers import AutoTokenizer, AutoModel
import torch

# Define Hugging Face Repository Details
repo_id = "swahar2/AI-Tutor"  # Replace with your repository name
filename = "xgboost_best_model.pkl"      # The name of the saved model in the repository

# Function to load the model from Hugging Face Hub
@st.cache_resource
def load_model(repo_id, filename):
    # Download the model from Hugging Face repository
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # Load the model using joblib
    model = joblib.load(model_path)
    return model

# Load the model
model = load_model(repo_id, filename)

# Load RoBERTa tokenizer and model (should match the training setup)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
roberta_model = AutoModel.from_pretrained('roberta-base', torch_dtype=torch.float32).to('cpu')  # Force full weight initialization on CPU

# Debugging: Check where the model is loaded
st.write("RoBERTa Model Device:", next(roberta_model.parameters()).device)

# Function to generate embeddings using RoBERTa
def get_embeddings(text):
    # Check if the input text is empty
    if not text.strip():
        return np.zeros((768,))  # Return a zero vector (768 is the default embedding size for RoBERTa)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to('cpu')  # Move inputs to CPU

    # Debugging: Check the tokenizer output shape
    st.write("Tokenizer Input IDs Shape:", inputs['input_ids'].shape)

    # Get the model's output
    with torch.no_grad():
        outputs = roberta_model(**inputs)

    # Debugging: Check the last_hidden_state tensor shape
    st.write("Last Hidden State Shape:", outputs.last_hidden_state.shape)

    # Extract the CLS token embedding (first token in the sequence)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # Move tensor to CPU before converting to NumPy

    return embeddings

# Streamlit app title
st.title("Band Score Prediction App 🎓")
st.write("This app predicts the band score based on your essay, prompt, and evaluation scores.")

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

# Predict button
if st.button("Predict Band Score"):
    # Check for empty inputs
    if not essay_input.strip():
        st.error("Essay input is empty. Please provide a valid essay.")
        st.stop()

    if not prompt_input.strip():
        st.error("Prompt input is empty. Please provide a valid prompt.")
        st.stop()

    # Generate RoBERTa embeddings for the essay and prompt
    essay_embedding = get_embeddings(essay_input)
    prompt_embedding = get_embeddings(prompt_input)

    # Debugging: Check shapes of features
    st.write("Essay Embedding Shape:", essay_embedding.shape)
    st.write("Prompt Embedding Shape:", prompt_embedding.shape)

    # Reshape embeddings to ensure they are 2D arrays
    essay_embedding = essay_embedding.reshape(1, -1)  # Reshape to (1, embedding_size)
    prompt_embedding = prompt_embedding.reshape(1, -1)  # Reshape to (1, embedding_size)

    # Combine embeddings with numeric features
    numeric_features = np.array([[task_achievement_score, coherence_score, lexical_score, grammatical_score]])
    input_features = np.hstack((essay_embedding, prompt_embedding, numeric_features))

    # Debugging: Check final shape
    st.write("Combined Input Features Shape:", input_features.shape)

    # Make prediction
    prediction = model.predict(input_features)[0]

    # Clip prediction to range [1, 9]
    prediction = np.clip(prediction, 1, 9)

    st.subheader(f"Predicted Band Score: {prediction:.2f}")
