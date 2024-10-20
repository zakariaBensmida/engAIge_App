from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def embed_text(text: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """Convert text into embeddings."""
    model = SentenceTransformer(model_name)
    return model.encode([text])[0]

def create_vector_store(embeddings: np.ndarray):
    """Create a FAISS index from embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

if __name__ == "__main__":
    # Example: running this file directly to create a vector store.
    text = "This is a sample text for embedding."
    embedding = embed_text(text)
    print("Embedding created:", embedding)





