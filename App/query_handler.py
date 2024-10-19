from app.vector_store import VectorStore
from app.llm import LLM

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




