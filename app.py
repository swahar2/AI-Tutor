import streamlit as st
import joblib
import requests
import os
import sys
import transformers


# Force CPU usage BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # Import torch AFTER setting CUDA_VISIBLE_DEVICES

# Set default device to CPU
torch.set_default_device('cpu')  # Add this line

st.write(f"Python version: {sys.version}")
st.write(f"Torch version: {torch.__version__}")
st.write(f"CUDA is available: {torch.cuda.is_available()}")

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')  # Use the same tokenizer as during training

MODEL_FILE = "band_predictor_model_torch.pth"
MODEL_URL = "https://huggingface.co/swahar2/AI-Tutor/resolve/main/band_predictor_model_torch.pth"

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
    task = st.slider("Task Achievement", 0.0, 9.0, 6.0)
    coherence = st.slider("Coherence & Cohesion", 0.0, 9.0, 6.0)
    vocab = st.slider("Vocabulary", 0.0, 9.0, 6.0)
    grammar = st.slider("Grammatical Range and Accuracy", 0.0, 9.0, 6.0)

    if st.button("Predict Band"):
        try:
            # Preprocess the input data
            encoded_input = tokenizer(
                essay,
                padding='max_length',
                truncation=True,
                max_length=128,  # Or whatever max length you used
                return_tensors='pt'  # Return PyTorch tensors
            )

            # Get the input IDs and attention mask
            input_ids = encoded_input['input_ids'].to('cpu')
            attention_mask = encoded_input['attention_mask'].to('cpu')

            # Convert numerical features to tensors
            task_tensor = torch.tensor([task]).to('cpu')
            coherence_tensor = torch.tensor([coherence]).to('cpu')
            vocab_tensor = torch.tensor([vocab]).to('cpu')
            grammar_tensor = torch.tensor([grammar]).to('cpu')

            # Pass the preprocessed data to the model
            with torch.no_grad():  # Disable gradient calculation during inference
                outputs = model(input_ids, attention_mask=attention_mask)  # Replace with your actual model call
                logits = outputs.logits  # Assuming your model returns logits
                prediction = torch.argmax(logits, dim=-1).item()  # Get the predicted class

            st.success(f"Predicted Band Score: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Model not loaded. Prediction is unavailable.")
