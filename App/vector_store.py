import os
import faiss
import numpy as np
import pickle
import logging
from sentence_transformers import SentenceTransformer
from pdf_extractor import PDFExtractor  # Assuming you have a modular pdf_extractor.py for PDF handling
from dotenv import load_dotenv  # Import dotenv to load environment variables

# Load environment variables from .env
load_dotenv()

# Get environment variables
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/distiluse-base-multilingual-cased-v2")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store/index.faiss")
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "./App/pdfs")

class VectorStore:
    def __init__(self, store_path, embedding_model_name):
        self.store_path = store_path
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.texts = []

        if store_path and os.path.exists(store_path):
            self._load_index_and_texts(store_path)
        else:
            logging.debug("Creating a new FAISS index store.")

    def _load_index_and_texts(self, store_path):
        """Load FAISS index and texts from stored files."""
        logging.debug(f"Loading existing FAISS index from {store_path}")
        self.index = faiss.read_index(store_path)
        pickle_path = f"{store_path}.pickle"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                self.texts = pickle.load(f)
            logging.debug(f"Loaded {len(self.texts)} texts from {pickle_path}")
        else:
            logging.warning(f"Pickle file {pickle_path} not found. Texts will not be loaded.")

    def embed_documents(self, documents):
        """Generate embeddings for a list of documents."""
        embeddings = self.model.encode(documents, show_progress_bar=True)
        logging.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def load_documents(self, pdf_directory):
        """Load, extract, and embed PDF documents from the specified directory."""
        pdf_extractor = PDFExtractor(pdf_directory)
        documents = pdf_extractor.extract_texts()
        
        if documents:
            self._add_documents_to_store(documents)
        logging.debug(f"Loaded and added {len(documents)} documents to the vector store.")

    def _add_documents_to_store(self, documents):
        """Embed and add documents to the FAISS index."""
        self.texts.extend(documents)
        embeddings = self.embed_documents(documents)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Initialize FAISS index
            logging.debug("Initialized new FAISS index.")

        self.index.add(embeddings)
        logging.debug(f"Added {embeddings.shape[0]} embeddings to the FAISS index.")

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

        query_embedding = self.embed_documents([query])
        distances, indices = self.index.search(query_embedding, k)
        results = self._get_texts_from_indices(indices)

        return results

    def _get_texts_from_indices(self, indices):
        """Retrieve texts based on indices from FAISS search results."""
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.texts):
                results.append(self.texts[idx])
            else:
                logging.debug(f"Index out of range or not found: {idx}")
        return results

    def has_texts(self):
        return len(self.texts) > 0

# Example usage for standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    vector_store = VectorStore(store_path=VECTOR_STORE_PATH, embedding_model_name=EMBEDDING_MODEL_NAME)

    # Load documents from the PDF directory
    vector_store.load_documents(PDF_STORAGE_PATH)

    # Save the vector store
    vector_store.save_store()

    # Example query
    query = "Wie hoch ist die Grundzulage?"
    results = vector_store.query(query)
    print("Query Results:", results)



