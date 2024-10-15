# app/vector_store.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List
import os
import logging
import numpy as np

class VectorStore:
    def __init__(self, store_path: str, embedding_model_name: str):
        self.store_path = store_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.texts = []
        self.ids = []
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        
        if os.path.exists(self.store_path):
            logging.debug(f"Loading existing store from {self.store_path}")
            self.load_store()
        else:
            logging.debug("Creating a new store.")
            # Using FAISS IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_model.get_sentence_embedding_dimension())
    
    def add_texts(self, texts: List[str], embeddings=None):
        logging.debug(f"Adding {len(texts)} texts to the vector store.")
        if embeddings is None:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        else:
            logging.debug("Using provided embeddings.")
        
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.ids.extend(range(len(self.texts)))
        logging.debug(f"Total texts in store: {len(self.texts)}")
    
    def save_store(self):
        logging.debug(f"Saving the vector store to {self.store_path}")
        faiss.write_index(self.index, self.store_path)
        with open(self.store_path + ".pickle", "wb") as f:
            pickle.dump(self.texts, f)
        logging.debug("Vector store saved successfully.")
    
    def load_store(self):
        logging.debug(f"Loading vector store from {self.store_path}")
        self.index = faiss.read_index(self.store_path)
        with open(self.store_path + ".pickle", "rb") as f:
            self.texts = pickle.load(f)
        self.ids = list(range(len(self.texts)))
        logging.debug(f"Vector store loaded successfully. Total texts: {len(self.texts)}")
    
    def query(self, query_text: str, k: int = 5):
        if not self.texts:
            logging.debug("No texts available in the vector store.")
            return []
        
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        D, I = self.index.search(query_embedding, k)
    
        logging.debug(f"Query embedding shape: {query_embedding.shape}")
        logging.debug(f"Distances: {D}")
        logging.debug(f"Indices: {I}")
    
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])
            else:
                logging.debug(f"Index out of range: {idx}")
    
        if not results:
            logging.debug(f"No relevant texts found for query: {query_text}")
    
        return results
    
    def has_texts(self):
        return len(self.texts) > 0

if __name__ == "__main__":
    # Example usage: This section can be used for standalone testing
    from config import get_embeddings  # Ensure this import is correct based on your project structure

    # Initialize logging
    logging.basicConfig(level=logging.DEBUG)

    # Define paths
    store_path = "C:/Users/zakar/engAIge_App/vector_store/index.faiss"
    embedding_model_name = "distiluse-base-multilingual-cased-v2"

    # Initialize VectorStore
    vector_store = VectorStore(store_path=store_path, embedding_model_name=embedding_model_name)

    # Example texts and embeddings
    texts = ["Dies ist ein Beispieltext.", "Hier ist ein weiterer Beispielsatz.", "Mehr Text für die Vektorspeicherung."]
    embeddings = get_embeddings().encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # Add texts to the store
    vector_store.add_texts(texts, embeddings=embeddings)
    vector_store.save_store()

