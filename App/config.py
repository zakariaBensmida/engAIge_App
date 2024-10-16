import os
import torch
import logging
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Set the number of threads for CPU
torch.set_num_threads(os.cpu_count())  # Use all available cores

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# General settings
MODEL_NAME = os.getenv("MODEL_NAME", "dbmdz/german-gpt2")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "distiluse-base-multilingual-cased-v2")

# Function to initialize the LLM
def get_llm():
    try:
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        
        return hf_pipeline
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return None

# Function to initialize the embeddings
def get_embeddings():
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return embedding_model
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None



