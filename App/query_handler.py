import os
from transformers import pipeline
from typing import List
from .vector_store import VectorStore  # Adjust the import path as necessary
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class QueryHandler:
    def __init__(self, vector_store: VectorStore):
        # Retrieve the LLM model name from the environment variables
        llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt2")  # Default to "gpt2" if not found
        self.llm = pipeline("text-generation", model=llm_model_name)
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
        response = self.llm(prompt, max_length=150, num_return_sequences=1, temperature=0.7)
        return response[0]['generated_text'].strip()

# Example usage
if __name__ == "__main__":
    from vector_store import VectorStore

    # Initialize the vector store (replace with your actual parameters)
    vector_store = VectorStore(store_path='./vector_store/index.faiss', embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME"))

    # Initialize QueryHandler
    query_handler = QueryHandler(vector_store=vector_store)

    # Example query
    query = "What is health insurance coverage?"
    answer = query_handler.get_answer(query)
    print("Answer:", answer)
