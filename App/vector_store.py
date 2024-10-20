import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []  # Keep a list of original documents
    
    def add_texts(self, texts: list):
        """Embeds and adds texts to the FAISS index."""
        embeddings = self.model.encode(texts)
        if self.index is None:
            # Create a FAISS index if it doesn't exist
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents.extend(texts)

    def retrieve(self, query: str, top_k: int = 3):
        """Retrieves the most relevant documents based on the query."""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return the most relevant documents
        return [{'text': self.documents[idx], 'distance': distances[0][i]} 
                for i, idx in enumerate(indices[0])]

if __name__ == "__main__":
    # Example test of vector store
    vs = VectorStore()
    texts = ["This is a test document.", "Another document.", "More data to test."]
    vs.add_texts(texts)

    query = "document"
    results = vs.retrieve(query)
    print(f"Results: {results}")






