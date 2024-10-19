from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize with a model for embeddings and FAISS index."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_documents(self, documents):
        """Converts documents to embeddings and adds them to the FAISS index."""
        self.documents = documents
        embeddings = self.model.encode([doc['text'] for doc in documents], convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve the top-k most similar documents to the query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[idx] for idx in indices[0]]

# Example usage:
# vector_store = VectorStore()
# vector_store.add_documents(load_pdfs("./pdf"))
# results = vector_store.retrieve("Was ist KI?", top_k=3)





