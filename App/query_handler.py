# App/query_handler.py
from langchain.llms import HuggingFacePipeline
from typing import List

class QueryHandler:
    def __init__(self, llm: HuggingFacePipeline):
        self.llm = llm

    def get_answer(self, relevant_texts: List[str], query: str) -> str:
        # Combine relevant texts into a context
        context = "\n".join(relevant_texts)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate the answer using the LLM
        response = self.llm(prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.1)
        return response[0]['generated_text'].strip()

