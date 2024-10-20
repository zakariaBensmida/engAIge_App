from vector_store import VectorStore
from llm import LLM

class QueryHandler:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = LLM()

    def handle_query(self, query: str) -> str:
        """Handles a query by retrieving relevant documents and generating an answer."""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.retrieve(query)
        context = " ".join([doc['text'] for doc in relevant_docs])

        # Generate an answer using the context
        answer = self.llm.generate(context, query)
        return answer

if __name__ == "__main__":
    # Example test
    handler = QueryHandler()

    # Add sample texts to vector store
    sample_texts = ["This is document 1.", "This is document 2.", "This is document 3."]
    handler.vector_store.add_texts(sample_texts)

    # Handle a query
    query = "document"
    print(handler.handle_query(query))




