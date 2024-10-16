import os
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, store_path, embedding_model_name):
        # Initialize the embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Get embeddings for a test query to determine dimensionality
        embedding = np.array(self.embedding_model.embed_query("test"))

        # Debugging: Check the shape and content of the embedding
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding content: {embedding}")

        # Handle the dimensionality of the embedding
        if len(embedding.shape) == 1:
            # Single embedding (1D array)
            embedding_dim = embedding.shape[0]
            print(f"Using shape[0], embedding_dim: {embedding_dim}")
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif len(embedding.shape) == 2:
            # Batch of embeddings (2D array)
            embedding_dim = embedding.shape[1]
            print(f"Using shape[1], embedding_dim: {embedding_dim}")
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")

        # Load an existing index if the store_path exists, or create a new one
        if store_path and os.path.exists(store_path):
            print(f"Loading existing FAISS index from {store_path}")
            self.index = faiss.read_index(store_path)
        else:
            print("Creating a new FAISS index store.")

    def save_index(self, store_path):
        """Save the FAISS index to a file."""
        if store_path:
            faiss.write_index(self.index, store_path)
            print(f"Index saved to {store_path}")
        else:
            raise ValueError("Store path cannot be None or empty.")

    def add_embeddings(self, documents):
        """Add document embeddings to the FAISS index."""
        embeddings = np.array([self.embedding_model.embed_query(doc) for doc in documents])
        print(f"Adding {len(documents)} embeddings to the index.")
        self.index.add(embeddings)

    def search(self, query, k=5):
        """Search the FAISS index for the top k results for a given query."""
        query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

# Example usage
if __name__ == "__main__":
    store_path = "C:/Users/zakar/engAIge_App/vector_store/index.faiss"
    embedding_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    
    # Initialize the VectorStore
    vector_store = VectorStore(store_path=store_path, embedding_model_name=embedding_model_name)
    
    # Example documents to add to the index
    documents = ["This is a test document.", "Another example text to embed."]
    
    # Add embeddings to the index
    vector_store.add_embeddings(documents)
    
    # Perform a search on the index
    query = "example query"
    distances, indices = vector_store.search(query, k=5)
    print(f"Search results for query '{query}': distances={distances}, indices={indices}")


