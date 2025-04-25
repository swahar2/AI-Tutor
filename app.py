import streamlit as st
import torch
import os
import sys
import transformers
import requests

# Force CPU usage BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # Import torch AFTER setting CUDA_VISIBLE_DEVICES

# Set default device to CPU
torch.set_default_device('cpu')  # Add this line

st.write(f"Python version: {sys.version}")
st.write(f"Torch version: {torch.__version__}")
st.write(f"Transformers version: {transformers.__version__}")
st.write(f"CUDA is available: {torch.cuda.is_available()}")
st.write(f"Files in current directory: {os.listdir()}")

# Load tokenizer (outside the load_model function)
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')  # Use the same tokenizer as during training

MODEL_FILE = "band_predictor_model_torch.pth"
MODEL_URL = "https://huggingface.co/swahar2/AI-Tutor/resolve/main/band_predictor_model_torch.pth"

def download_model(url, filename):
    try:
        st.info(f"Downloading model from {url} to {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded successfully!")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        return False

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_FILE):
            st.warning("Model file not found locally. Downloading...")
            if not download_model(MODEL_URL, MODEL_FILE):
                st.error("Failed to download model. Aborting.")
                return None

        # Load the entire model using torch.load, mapping to CPU
        model = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
        st.success("Model loaded successfully using torch.load!")
        st.write(f"Model: {model}")  # Print the model
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error(f"Error type: {type(e).__name__}")  # Get the error type
        st.error(f"Error message: {str(e)}")        # Get the error message
        st.error(f"Full traceback: {sys.exc_info()[2]}") # Get the full traceback
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
            ).to('cpu')

            # Get the input IDs and attention mask
            input_ids = encoded_input['input_ids'].to('cpu')
            attention_mask = encoded_input['attention_mask'].to('cpu')

            # Convert numerical features to tensors
            task_tensor = torch.tensor([task]).to('cpu')
            coherence_tensor = torch.tensor([coherence]).to('cpu')
            vocab_tensor = torch.tensor([vocab]).to('cpu')
            grammar_tensor = torch.tensor([grammar]).to('cpu')

            # Print the shapes and types of the inputs
            st.write(f"Input IDs shape: {input_ids.shape}, type: {input_ids.dtype}")
            st.write(f"Attention Mask shape: {attention_mask.shape}, type: {attention_mask.dtype}")
            st.write(f"Task Tensor shape: {task_tensor.shape}, type: {task_tensor.dtype}")
            st.write(f"Coherence Tensor shape: {coherence_tensor.shape}, type: {coherence_tensor.dtype}")
            st.write(f"Vocab Tensor shape: {vocab_tensor.shape}, type: {vocab_tensor.dtype}")
            st.write(f"Grammar Tensor shape: {grammar_tensor.shape}, type: {grammar_tensor.dtype}")

            # Pass the preprocessed data to the model
            with torch.no_grad():  # Disable gradient calculation during inference
                # Adapt this part to match what your model expects
                # This is a placeholder for the actual prediction code
                # Replace this with your actual model call
                # Example: If your model takes input_ids, attention_mask, and other tensors
                outputs = model(input_ids, attention_mask=attention_mask,
                                task=task_tensor, coherence=coherence_tensor,
                                vocab=vocab_tensor, grammar=grammar_tensor)

                st.write(f"Model outputs: {outputs}") # Print the model outputs

                logits = outputs.logits  # Assuming your model returns logits
                st.write(f"Logits: {logits}")  # Print the logits
                prediction = torch.argmax(logits, dim=-1).item()  # Get the predicted class

            st.success(f"Predicted Band Score: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.error(f"Error type: {type(e).__name__}")  # Get the error type
            st.error(f"Error message: {str(e)}")        # Get the error message
            st.error(f"Full traceback: {sys.exc_info()[2]}") # Get the full traceback
        else:
            st.warning("Model not loaded. Prediction is unavailable.")
