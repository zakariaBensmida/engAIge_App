import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceLLM

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# General settings
MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stablelm-tuned-alpha-7b")  # Default model name for German
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/german-nlp")

# LLM configuration
LLM_CONFIG = {
    "model_name": MODEL_NAME,
    "max_length": 150,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "model_name": EMBEDDING_MODEL_NAME,
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available, else CPU
}

# Function to initialize the LLM
def get_llm():
    return HuggingFaceLLM(**LLM_CONFIG)

# Function to initialize the embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(**EMBEDDING_CONFIG)

