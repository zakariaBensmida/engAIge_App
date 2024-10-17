import os
import logging
from transformers import pipeline
from typing import List
from App.vector_store import VectorStore  # Adjust the import path as necessary
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

class QueryHandler:
    def __init__(self, vector_store: VectorStore):
        # Retrieve the LLM model name from the environment variables
        llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt2")  # Default to "gpt2" if not found
        self.llm = pipeline("text-generation", model=llm_model_name)
        self.vector_store = vector_store

    def get_relevant_texts(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant texts from the vector store based on the query."""
        relevant_texts = self.vector_store.query(query, k=top_k)
        return relevant_texts

    def get_answer(self, query: str) -> str:
        """Generate an answer based on the given query and relevant texts."""
        relevant_texts = self.get_relevant_texts(query)

        if not relevant_texts:
            return "Ich kann leider keine relevanten Informationen finden."

        # Limit context to the first few texts or sentences
        context = "\n".join(relevant_texts[:3])  # Limit to the top 3 relevant texts
        prompt = f"Kontext:\n{context}\n\nFrage: {query}\nAntwort:"

        # Generate the answer using the LLM
        try:
            response = self.llm(prompt, max_length=150, num_return_sequences=1, temperature=0.5, truncation=True, pad_token_id=self.llm.tokenizer.eos_token_id)
            return response[0]['generated_text'].strip()
        except Exception as e:
            logging.error("Fehler bei der Generierung der Antwort: %s", e)
            return "Ein Fehler ist aufgetreten, während die Antwort generiert wurde."


# Example usage
if __name__ == "__main__":
    from App.vector_store import VectorStore

    # Initialize the vector store (replace with your actual parameters)
    vector_store = VectorStore(
        store_path='./vector_store/index.faiss', 
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME")
    )

    # Initialize QueryHandler
    query_handler = QueryHandler(vector_store=vector_store)

    # Example query
    query = "What is health insurance coverage?"
    answer = query_handler.get_answer(query)
    print("Answer:", answer)






