from .vector_store import VectorStore
from .llm import LLM

class QueryHandler:
    def __init__(self, llm: LLM, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = llm

    def handle_query(self, query: str) -> str:
        """Handles a query by retrieving relevant documents and generating an answer."""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.retrieve(query)
        context = " ".join([doc['text'] for doc in relevant_docs]) if relevant_docs else "No relevant documents found."

        # Print the context and the query for debugging
        print(f"Context: {context}")
        print(f"Query: {query}")

        # Generate an answer using the context
        answer = self.llm.generate(context, query)
        print(f"Answer: {answer}")  # Check what the LLM generates
        return answer

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    # Load environment variables
    load_dotenv()

    # Create an instance of the LLM class with the model name from the environment
    llm_model_name = os.getenv('LLM_MODEL_NAME', 'gpt2')
    llm_instance = LLM(model_name=llm_model_name)

    # Initialize the VectorStore with the specified path
    vector_store_path = os.getenv('VECTOR_STORE_PATH', 'C:\\Users\\zakar\\engAIge_App\\App\\vector_store.faiss')
    vector_store_instance = VectorStore(vector_store_path)

    # Initialize the QueryHandler with the LLM and VectorStore instances
    handler = QueryHandler(llm=llm_instance, vector_store=vector_store_instance)

    # Add sample texts to vector store
    sample_texts = ["This is document 1.", "This is document 2.", "This is document 3."]
    handler.vector_store.add_texts(sample_texts)

    # Handle a query
    query = "document"
    handler.handle_query(query)






