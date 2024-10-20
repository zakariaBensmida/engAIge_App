from vector_store import VectorStore
from llm import LLM

class QueryHandler:
    def __init__(self, llm):
        self.llm = llm

    def handle_query(self, query):
        context = "This is document 3. This is document 2. This is document 1."
        print(f"Query: {query}")  # Check query content before processing
        answer = self.llm.generate(context, query)
        print(f"Answer: {answer}")  # Check what the LLM generates
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





