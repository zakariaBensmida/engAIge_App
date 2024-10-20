import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle  # Import pickle for saving/loading texts

class VectorStore:
    def __init__(self, store_path: str, embedding_model_name: str):
        self.store_path = store_path
        self.embedding_model_name = embedding_model_name
        self.index = None
        self.texts = []
        
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(self.embedding_model_name)

        # Load existing index and texts if they exist
        if os.path.exists(self.store_path):
            self.load()

    def add_texts(self, texts):
        embeddings = self.model.encode(texts)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Initialize the FAISS index
        self.index.add(embeddings)  # Add embeddings to the index
        self.texts.extend(texts)  # Store texts for retrieval
        self.save()  # Save the updated index and texts

    def save(self):
        faiss.write_index(self.index, self.store_path)  # Save the index to disk
        with open(self.store_path.replace('.faiss', '_texts.pkl'), 'wb') as f:
            pickle.dump(self.texts, f)  # Save texts to a pickle file

    def load(self):
        self.index = faiss.read_index(self.store_path)  # Load index from disk
        texts_path = self.store_path.replace('.faiss', '_texts.pkl')
        if os.path.exists(texts_path):
            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)  # Load texts from pickle

    def retrieve(self, query, k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)  # Search for nearest neighbors
        return [{"text": self.texts[i]} for i in indices[0]]  # Return the relevant texts

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Get the paths from environment variables
    vector_store_path = os.getenv('VECTOR_STORE_PATH', 'C:\\Users\\zakar\\engAIge_App\\App\\vector_store.faiss')
    embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', 'distiluse-base-multilingual-cased-v2')

    # Create an instance of the VectorStore class
    vector_store_instance = VectorStore(vector_store_path, embedding_model_name)

    # Add sample texts to vector store (this will create the index)
    sample_texts = ["This is document 1.", "This is document 2.", "This is document 3."]
    vector_store_instance.add_texts(sample_texts)

    # Handle a query
    query = "document"
    relevant_docs = vector_store_instance.retrieve(query)
    print(relevant_docs)







