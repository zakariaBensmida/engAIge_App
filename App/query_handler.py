import os
import logging
from transformers import pipeline
from typing import List
from .vector_store import VectorStore  # Adjust the import path as necessary
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

class QueryHandler:
    def __init__(self, vector_store: VectorStore):
        # Retrieve the LLM model name from the environment variables
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt2")
        self.llm = pipeline("text-generation", model=self.llm_model_name)
        self.vector_store = vector_store

    def get_relevant_texts(self, query: str, top_k: int = 50) -> List[str]:
        try:
            relevant_texts = self.vector_store.query(query, k=top_k)
            logging.info(f"Retrieved {len(relevant_texts)} relevant texts for query: {query}")
            return relevant_texts
        except Exception as e:
            logging.error(f"Error retrieving texts from vector store: {e}")
            return []

    def get_answer(self, query: str, max_length: int = 150) -> str:
        relevant_texts = self.get_relevant_texts(query)
        
        # Combine relevant texts into a context
        context = "\n".join(relevant_texts)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate the answer using the LLM
        try:
            response = self.llm(prompt, truncation=True, max_length=max_length)
            answer = response[0]['generated_text'].strip()
            logging.info(f"Generated answer for query '{query}': {answer}")
            return answer
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer at this time."

# Example usage
def main():
    from .vector_store import VectorStore

    # Initialize the vector store (replace with your actual parameters)
    vector_store = VectorStore(store_path='./vector_store/index.faiss', 
                               embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME"))

    # Initialize QueryHandler
    query_handler = QueryHandler(vector_store=vector_store)

    # Example query
    query = "wie hoch ist die Grundzulage?"
    answer = query_handler.get_answer(query)
    print("Answer:", answer)

if __name__ == "__main__":
    main()

