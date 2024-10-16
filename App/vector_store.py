# App/vector_store.py

import os
import faiss
import numpy as np
import pickle
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self, store_path, embedding_model_name):
        self.store_path = store_path
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.texts = []

        if store_path and os.path.exists(store_path):
            logging.debug(f"Loading existing FAISS index from {store_path}")
            self.index = faiss.read_index(store_path)
            pickle_path = f"{store_path}.pickle"
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    self.texts = pickle.load(f)
                logging.debug(f"Loaded {len(self.texts)} texts from {pickle_path}")
            else:
                logging.warning(f"Pickle file {pickle_path} not found. Texts will not be loaded.")
        else:
            # Initialize a new FAISS index
            self.index = None  # Initialize index here to create later
            logging.debug("Creating a new FAISS index store.")

    def load_documents(self, pdf_directory):
        """Load and split documents from the specified directory."""
        loader = PyPDFDirectoryLoader(pdf_directory)
        documents = loader.load()

        # Split the documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        
        # Generate embeddings for the split documents
        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in final_documents])
        embeddings = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Use the correct embedding dimension
        self.index.add(embeddings)

        self.texts.extend([doc.page_content for doc in final_documents])
        logging.debug(f"Loaded and added {len(final_documents)} documents to the vector store.")

    def save_store(self):
        """Save the FAISS index and texts to files."""
        if self.store_path:
            faiss.write_index(self.index, self.store_path)
            pickle_path = f"{self.store_path}.pickle"
            with open(pickle_path, "wb") as f:
                pickle.dump(self.texts, f)
            logging.debug(f"Index saved to {self.store_path} and texts saved to {pickle_path}")
        else:
            raise ValueError("Store path cannot be None or empty.")

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
    from App.config import get_embeddings  # Ensure this import is correct

    # Initialize logging
    logging.basicConfig(level=logging.DEBUG)

    # Define paths
    store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store/index.faiss")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")  # Update model name

    # Initialize VectorStore
    vector_store = VectorStore(store_path=store_path, embedding_model_name=embedding_model_name)

    # Load documents from the PDF directory
    pdf_directory = './path_to_your_pdf_directory'  # Specify your PDF directory
    vector_store.load_documents(pdf_directory)

    # Save the vector store
    vector_store.save_store()

    # Example query
    query = "What is health insurance coverage?"
    results = vector_store.query(query)
    print("Query Results:", results)
