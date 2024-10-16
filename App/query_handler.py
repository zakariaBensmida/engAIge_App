from transformers import pipeline
from typing import List
from App.vector_store import VectorStore  # Adjust the import path as necessary

class QueryHandler:
    def __init__(self, llm_model_name: str, vector_store: VectorStore):
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
    from App.vector_store import VectorStore

    # Initialize the vector store (replace with your actual parameters)
    vector_store = VectorStore(store_path='./vector_store/index.faiss', embedding_model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Initialize QueryHandler
    query_handler = QueryHandler(llm_model_name='gpt2', vector_store=vector_store)  # Change to your preferred model

    # Example query
    query = "What is health insurance coverage?"
    answer = query_handler.get_answer(query)
    print("Answer:", answer)
