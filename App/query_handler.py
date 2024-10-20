from .vector_store import VectorStore
from llm import LLM

class QueryHandler:
    def __init__(self, llm: LLM):
        self.vector_store = VectorStore()
        self.llm = llm

    def handle_query(self, query: str) -> str:
        """Handles a query by retrieving relevant documents and generating an answer."""
        # Retrieve relevant documents (here, using dummy context for example)
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
    # Create an instance of the LLM class
    llm_instance = LLM()  # Make sure this is correctly initialized

    # Initialize the QueryHandler with the LLM instance
    handler = QueryHandler(llm=llm_instance)

    # Add sample texts to vector store
    sample_texts = ["This is document 1.", "This is document 2.", "This is document 3."]
    handler.vector_store.add_texts(sample_texts)

    # Handle a query
    query = "document"
    handler.handle_query(query)






