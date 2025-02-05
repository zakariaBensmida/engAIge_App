# backend/models/llm.py - Loads Hugging Face LLM Optimized for CPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_llm():
    model_path = "mistralai/Mistral-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return lambda prompt: llm_pipeline(prompt, max_length=256, do_sample=True)
