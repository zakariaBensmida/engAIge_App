import os
import torch
import logging
from dotenv import load_dotenv
from langchain_community.llms import HuggingFacePipeline, HuggingFaceEmbeddings  # Updated imports
from transformers import pipeline

# Set the number of threads for CPU
torch.set_num_threads(os.cpu_count())  # Use all available cores

# Configure logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

# General settings
MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stablelm-tuned-alpha-7b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "distiluse-base-multilingual-cased-v2")

# Function to initialize the LLM
def get_llm():
    try:
        device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
        hf_pipeline = pipeline("text-generation", model=MODEL_NAME, device=device)
        
        # Initialize HuggingFacePipeline with the pipeline only
        return HuggingFacePipeline(pipeline=hf_pipeline)
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return None

# Function to initialize the embeddings
def get_embeddings():
    try:
        # Initialize HuggingFaceEmbeddings with the new class
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)  # Ensure this matches the new API
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None
