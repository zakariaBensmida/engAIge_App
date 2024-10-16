# vector_store.py
import faiss
import pickle
from typing import List
import os
import logging
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, store_path: str, embedding_model_name: str):
        self.store_path = store_path
        self.embedding_model_name = embedding_model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
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
            # Using FAISS IndexFlatL2 for cosine similarity
            self.index = faiss.IndexFlatL2(np.array(self.embedding_model.embed_query("test")).shape[1])


    def add_texts(self, texts: List[str], embeddings=None):
        logging.debug(f"Adding {len(texts)} texts to the vector store.")
        if embeddings is None:
            embeddings = self.embedding_model.embed_documents(texts)
            logging.debug(f"Generated embeddings with shape: {len(embeddings)} x {len(embeddings[0])}")
        else:
            logging.debug("Using provided embeddings.")
        
        embeddings = np.array(embeddings).astype('float32')
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
        
        query_embedding = self.embedding_model.embed_query(query_text)
        query_embedding = np.array([query_embedding]).astype('float32')
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
    # Initialize logging
    logging.basicConfig(level=logging.DEBUG)

    # Define paths
    store_path = "C:/Users/zakar/engAIge_App/vector_store/index.faiss"
    embedding_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"

    # Initialize VectorStore
    vector_store = VectorStore(store_path=store_path, embedding_model_name=embedding_model_name)

    # Example texts
    texts = [
        "Dies ist ein Beispieltext.",
        "Hier ist ein weiterer Beispielsatz.",
        "Mehr Text für die Vektorspeicherung."
    ]

    # Add texts to the store
    vector_store.add_texts(texts)
    vector_store.save_store()
    logging.debug("Vector store populated and saved.")


