import streamlit as st
import torch
import os
import sys
import transformers
import requests
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset
import numpy as np

# Force CPU usage BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  

# Set default device to CPU
torch.set_default_device('cpu')   

st.write(f"Python version: {sys.version}")
st.write(f"Torch version: {torch.__version__}")
st.write(f"Transformers version: {transformers.__version__}")
st.write(f"CUDA is available: {torch.cuda.is_available()}")

MODEL_NAME = "swahar2/AI-Tutor"   

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

@st.cache_resource
def load_model():
    try:
        model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, map_location=torch.device('cpu'))
        st.success("Model loaded successfully from Hugging Face Hub!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

st.title("IELTS Band Predictor")

if model:
    essay = st.text_area("Enter Essay Text")
    prompt = st.text_area("Enter Prompt Text")

    if st.button("Predict Scores"):
        try:
            # Combine prompt and essay
            feature_text = f"{prompt} {essay}"

            # Preprocess the input data
            encoded_input = tokenizer(
                feature_text,
                padding='max_length',
                truncation=True,
                max_length=512,  
                return_tensors='pt'  
            )

            # Get the input IDs and attention mask
            input_ids = encoded_input['input_ids'].to('cpu')
            attention_mask = encoded_input['attention_mask'].to('cpu')

            # Pass the preprocessed data to the model
            with torch.no_grad():  # Disable gradient calculation during inference
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Assuming your model returns logits

                # Assuming your model outputs 5 values (band, TA, CC, LR, GRA)
                predictions = torch.sigmoid(logits).cpu().numpy()[0]  # Apply sigmoid for probabilities

                # Display the predictions
                st.write("### Predicted Scores:")
                st.write(f"Band: {predictions[0]:.2f}")
                st.write(f"Task Achievement: {predictions[1]:.2f}")
                st.write(f"Coherence & Cohesion: {predictions[2]:.2f}")
                st.write(f"Lexical Resource: {predictions[3]:.2f}")
                st.write(f"Grammatical Range & Accuracy: {predictions[4]:.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Model not loaded. Prediction is unavailable.")
