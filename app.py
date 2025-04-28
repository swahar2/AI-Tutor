import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request

# Define the URL to your model on Hugging Face
model_url = "https://huggingface.co/swahar2/AI-Tutor/resolve/main/xgboost_best_model.pkl"

# Function to load the model from Hugging Face Hub
@st.cache_resource
def load_model(url):
    # Download the model file from the URL
    file_path, _ = urllib.request.urlretrieve(url, "xgboost_best_model.pkl")
    # Load the model using joblib
    model = joblib.load(file_path)
    return model

# Load the model
model = load_model(model_url)

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
    # Preprocess the input
    input_data = {
        'essay_cleaned': [essay_input],
        'prompt_cleaned': [prompt_input],
        'Task Achievement_score': [task_achievement_score],
        'Coherence and Cohesion_score': [coherence_score],
        'Lexical Resource_score': [lexical_score],
        'Grammatical Range and Accuracy_score': [grammatical_score]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Band Score: {prediction:.2f}")
