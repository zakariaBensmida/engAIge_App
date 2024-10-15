import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List
import os
import numpy as np

class VectorStore:
    def __init__(self, store_path: str, embedding_model_name: str):
        self.store_path = store_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.texts = []
        self.ids = []
        
        if os.path.exists(self.store_path):
            self.load_store()
        else:
            # Using FAISS IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_model.get_sentence_embedding_dimension())
    
    def add_texts(self, texts: List[str]):
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.ids.extend(range(len(self.texts)))
    
    def save_store(self):
        faiss.write_index(self.index, self.store_path)
        with open(self.store_path + ".pickle", "wb") as f:
            pickle.dump(self.texts, f)
    
    def load_store(self):
        self.index = faiss.read_index(self.store_path)
        with open(self.store_path + ".pickle", "rb") as f:
            self.texts = pickle.load(f)
        self.ids = list(range(len(self.texts)))
    
    def query(self, query_text: str, k: int = 5):
        if not self.texts:
            return []  # Return an empty list if no texts are available
        
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        D, I = self.index.search(query_embedding, k)

        results = []
        print("Distances:", D)  # Debugging line
        print("Indices:", I)     # Debugging line

        for idx in I[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])
            else:
                print(f"Index out of range: {idx}")  # Debugging line

        return results

    



