import os
from transformers import pipeline
from typing import List
from App.vector_store import VectorStore  # Adjust the import path as necessary
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
        """
        Retrieves relevant texts from the vector store based on the query.

        Args:
            query (str): The user's question or query.
            top_k (int): Number of relevant texts to retrieve.

        Returns:
            List[str]: A list of relevant texts.
        """
        relevant_texts = self.vector_store.query(query, k=top_k)
        return relevant_texts

    def get_answer(self, query: str) -> str:
        """
        Generates an answer to the query by retrieving relevant texts and using an LLM to generate a response.

        Args:
            query (str): The user's question or query.

        Returns:
            str: The generated answer.
        """
        try:
            # Retrieve relevant texts for the given query
            relevant_texts = self.get_relevant_texts(query)
            
            # Combine relevant texts into a context
            context = "\n".join(relevant_texts)
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

            # Generate the answer using the LLM
            inputs = self.llm.tokenizer(prompt, return_tensors='pt')
            response_ids = self.llm.model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7
            )

            answer = self.llm.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            return answer.strip()
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "An error occurred while generating the answer."

# Example usage
if __name__ == "__main__":
    from App.vector_store import VectorStore

    # Initialize the vector store (replace with your actual parameters)
    vector_store = VectorStore(store_path='./vector_store/index.faiss', embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME"))

    # Initialize QueryHandler
    query_handler = QueryHandler(vector_store=vector_store)

    # Example query
    query = "What is health insurance coverage?"
    answer = query_handler.get_answer(query)
    print("Answer:", answer)



