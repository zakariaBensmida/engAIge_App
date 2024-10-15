import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List
import os
import numpy as np

# Set your desired store path for the FAISS index
store_path = "C:/Users/zakar/vector_store/index.faiss"  # Add a filename to store the index

class VectorStore:
    def __init__(self, store_path: str, embedding_model_name: str):
        self.store_path = store_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.texts = []
        self.ids = []
        
        if os.path.exists(self.store_path):
            print("Loading existing store from", self.store_path)
            self.load_store()
        else:
            print("Creating a new store.")
            # Using FAISS IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_model.get_sentence_embedding_dimension())
    
    def add_texts(self, texts: List[str]):
        print(f"Adding {len(texts)} texts to the vector store.")
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        print("Embeddings shape:", embeddings.shape)  # Debugging line
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.ids.extend(range(len(self.texts)))
    
    def save_store(self):
        print("Saving the vector store to", self.store_path)
        faiss.write_index(self.index, self.store_path)
        with open(self.store_path + ".pickle", "wb") as f:
            pickle.dump(self.texts, f)
        print("Vector store saved successfully.")
    
    def load_store(self):
        print("Loading vector store from", self.store_path)
        self.index = faiss.read_index(self.store_path)
        with open(self.store_path + ".pickle", "rb") as f:
            self.texts = pickle.load(f)
        self.ids = list(range(len(self.texts)))
        print("Vector store loaded successfully. Total texts:", len(self.texts))
    
    def query(self, query_text: str, k: int = 5):
        if not self.texts:
            print("No texts available in the vector store.")  # Debugging line
            return []  # Return an empty list if no texts are available
        
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        D, I = self.index.search(query_embedding, k)

        print("Query embedding shape:", query_embedding.shape)  # Debugging line
        print("Distances:", D)  # Debugging line
        print("Indices:", I)     # Debugging line

        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])
            else:
                print(f"Index out of range: {idx}")  # Debugging line

        if not results:
            print("No relevant texts found for query:", query_text)  # Debugging line

        return results

# Example usage
if __name__ == "__main__":
    # Initialize VectorStore
    vector_store = VectorStore(store_path, "distiluse-base-multilingual-cased-v2")

    # Add some test texts
    vector_store.add_texts([
        "This is a test document.",
        "Another example document.",
        "A third document for querying."
    ])

    # Save the store
    vector_store.save_store()

    # Query for a known text
    results = vector_store.query("test document")
    print("Query Results:", results)
