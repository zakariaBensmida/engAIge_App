# App/query_handler.py
from langchain.llms import HuggingFacePipeline
from typing import List
from App.vector_store import VectorStore  # Adjust the import path as necessary

class QueryHandler:
    def __init__(self, llm: HuggingFacePipeline, vector_store: VectorStore):
        self.llm = llm
        self.vector_store = vector_store

    def get_relevant_texts(self, query: str, top_k: int = 5) -> List[str]:
        # Retrieve relevant texts from the vector store based on the query
        relevant_texts = self.vector_store.query(query, k=top_k)
        return relevant_texts

    def get_answer(self, query: str) -> str:
        # Retrieve relevant texts for the given query
        relevant_texts = self.get_relevant_texts(query)
        
        # Combine relevant texts into a context
        context = "\n".join(relevant_texts)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate the answer using the LLM
        response = self.llm(prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.1)
        return response[0]['generated_text'].strip()
