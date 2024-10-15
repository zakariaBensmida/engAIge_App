import os
import torch
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Set the number of threads for CPU
torch.set_num_threads(os.cpu_count())  # Use all available cores

load_dotenv()

# General settings
MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stablelm-tuned-alpha-7b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "distiluse-base-multilingual-cased-v2")

# LLM configuration
LLM_CONFIG = {
    "max_length": 150,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
}

# Embedding configuration (removed 'device')
EMBEDDING_CONFIG = {
    "model_name": EMBEDDING_MODEL_NAME,
    # Removed "device" parameter
}

# Function to initialize the LLM
def get_llm():
    try:
        device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
        hf_pipeline = pipeline("text-generation", model=MODEL_NAME, device=device)
        return HuggingFacePipeline(pipeline=hf_pipeline, **LLM_CONFIG)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# Function to initialize the embeddings
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(**EMBEDDING_CONFIG)  # Ensure this does not include 'device'
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


