# tests/test_vector_manager.py - Tests FAISS vector store
import numpy as np
from backend.vector_manager import embed_documents, create_vector_store

def test_vector_store():
    documents = ["Hello world", "This is a test."]
    embeddings = embed_documents(documents)
    assert embeddings.shape[0] == len(documents)
    
    index = create_vector_store(embeddings)
    assert index.is_trained

