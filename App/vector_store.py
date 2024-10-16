# vector_store.py

import os
import faiss
import numpy as np
import pickle
import logging
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.schema import Document

class VectorStore:
    def __init__(self, store_path, embedding_model_name):
        self.store_path = store_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.texts = []

        # Get embeddings for a test query to determine dimensionality
        embedding = np.array(self.embedding_model.embed_query("test"))

        # Debugging: Check the shape and content of the embedding
        logging.debug(f"Embedding shape: {embedding.shape}")
        logging.debug(f"Embedding content: {embedding}")

        # Handle the dimensionality of the embedding
        if len(embedding.shape) == 1:
            # Single embedding (1D array)
            embedding_dim = embedding.shape[0]
            logging.debug(f"Using shape[0], embedding_dim: {embedding_dim}")
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif len(embedding.shape) == 2:
            # Batch of embeddings (2D array)
            embedding_dim = embedding.shape[1]
            logging.debug(f"Using shape[1], embedding_dim: {embedding_dim}")
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")

        # Load an existing index if the store_path exists, or create a new one
        if store_path and os.path.exists(store_path):
            logging.debug(f"Loading existing FAISS index from {store_path}")
            self.index = faiss.read_index(store_path)
            # Load the associated texts
            with open(f"{store_path}.pickle", "rb") as f:
                self.texts = pickle.load(f)
            logging.debug(f"Loaded {len(self.texts)} texts from the store.")
        else:
            logging.debug("Creating a new FAISS index store.")

    def save_store(self):
        """Save the FAISS index and texts to files."""
        if self.store_path:
            faiss.write_index(self.index, self.store_path)
            with open(f"{self.store_path}.pickle", "wb") as f:
                pickle.dump(self.texts, f)
            logging.debug(f"Index saved to {self.store_path} and texts saved to {self.store_path}.pickle")
        else:
            raise ValueError("Store path cannot be None or empty.")

    def add_texts(self, texts, embeddings=None):
        """Add document embeddings to the FAISS index."""
        if embeddings is None:
            embeddings = self.embedding_model.embed_documents(texts)
            embeddings = np.array(embeddings).astype('float32')
            logging.debug(f"Generated embeddings with shape: {embeddings.shape}")
        else:
            logging.debug("Using provided embeddings.")

        self.index.add(embeddings)
        self.texts.extend(texts)
        logging.debug(f"Total texts in store: {len(self.texts)}")

    def query(self, query, k=5):
        """Search the FAISS index for the top k results for a given query."""
        if not self.texts:
            logging.debug("No texts available in the vector store.")
            return []

        query_embedding = np.array(self.embedding_model.embed_query(query)).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        logging.debug(f"Query embedding shape: {query_embedding.shape}")
        logging.debug(f"Distances: {distances}")
        logging.debug(f"Indices: {indices}")

        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.texts):
                results.append(self.texts[idx])
            else:
                logging.debug(f"Index out of range or not found: {idx}")

        if not results:
            logging.debug(f"No relevant texts found for query: {query}")

        return results

    def has_texts(self):
        return len(self.texts) > 0

# Example usage for standalone testing
if __name__ == "__main__":
    import logging
    from config import get_embeddings  # Ensure this import is correct based on your project structure

    # Initialize logging
    logging.basicConfig(level=logging.DEBUG)

    # Define paths
    store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store/index.faiss")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "distiluse-base-multilingual-cased-v2")

    # Initialize VectorStore
    vector_store = VectorStore(store_path=store_path, embedding_model_name=embedding_model_name)

    # Example texts and embeddings
    texts = [
        "Dies ist ein Beispieltext.",
        "Hier ist ein weiterer Beispielsatz.",
        "Mehr Text für die Vektorspeicherung."
    ]
    embeddings = vector_store.embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings).astype('float32')

    # Add texts to the store
    vector_store.add_texts(texts, embeddings=embeddings)
    vector_store.save_store()



