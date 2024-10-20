import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, vector_store_path: str, model_name: str):
        self.vector_store_path = vector_store_path
        self.model = SentenceTransformer(model_name)

        # Check if the vector store file exists; if not, initialize an empty index
        if os.path.exists(self.vector_store_path):
            self.index = faiss.read_index(self.vector_store_path)
        else:
            self.index = None
            self.documents = []  # To hold the documents for indexing

    def add_texts(self, texts):
        """Generate embeddings for the provided texts and add them to the FAISS index."""
        embeddings = self.model.encode(texts)  # Generate embeddings
        if self.index is None:
            self.documents.extend(texts)  # Store texts if index is not created yet
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Initialize FAISS index

        # Add embeddings to the index
        self.index.add(embeddings.astype(np.float32))
        
        # Save the updated index
        self.save_vector_store()

    def save_vector_store(self):
        """Saves the FAISS index to the specified file path."""
        faiss.write_index(self.index, self.vector_store_path)

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve the top K relevant documents for the given query."""
        query_embedding = self.model.encode([query]).astype(np.float32)  # Encode the query
        distances, indices = self.index.search(query_embedding, top_k)  # Search the index

        # Retrieve the documents based on indices
        return [{"text": self.documents[i], "distance": distances[0][j]} for j, i in enumerate(indices[0]) if i < len(self.documents)]

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






