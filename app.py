import streamlit as st
import torch
import os
import sys
import transformers
import requests
from torch import nn

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


class CustomBertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels, num_numerical_features):
        super(CustomBertClassifier, self).__init__()
        self.bert = transformers.BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        self.linear = nn.Linear(num_numerical_features, 768)  # Adjust input size
        self.classifier = nn.Linear(768 * 2, num_labels) # BERT output + numerical features

    def forward(self, input_ids, attention_mask, task, coherence, vocab, grammar):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output # Access pooler_output

        numerical_features = torch.cat((task.unsqueeze(1), coherence.unsqueeze(1), vocab.unsqueeze(1), grammar.unsqueeze(1)), dim=1).float()
        numerical_embedding = self.linear(numerical_features)

        # Concatenate the BERT output with the numerical features
        combined_features = torch.cat((pooled_output, numerical_embedding), dim=1) # Concatenate along the dimension 1

        # Pass the combined features to the classifier
        logits = self.classifier(combined_features)
        return transformers.modeling_outputs.SequenceClassifierOutput(logits=logits, loss=None) # Loss needs to be none


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
        model = CustomBertClassifier(bert_model_name='bert-base-uncased', num_labels=2, num_numerical_features=4)

        # Load the state_dict
        state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))

        # Load the state_dict into the model
        model.load_state_dict(state_dict)

        st.success("Model loaded successfully!")
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
            outputs = model(input_ids, attention_mask=attention_mask, task=task_tensor, coherence=coherence_tensor, vocab=vocab_tensor, grammar=grammar_tensor)  # Pass all inputs
            logits = outputs.logits  # Assuming your model returns logits
            prediction = torch.argmax(logits, dim=-1).item()  # Get the predicted class

        st.success(f"Predicted Band Score: {prediction}")
      except Exception as e:
        st.error(f"Prediction error: {e}")
