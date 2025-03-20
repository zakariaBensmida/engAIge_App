# backend/vector_manager.py - Manages FAISS vector search and embeddings
import faiss
from sentence_transformers import SentenceTransformer

def embed_documents(documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(documents, convert_to_numpy=True)

def create_vector_store(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
