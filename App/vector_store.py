import os
import faiss
import numpy as np
import pickle
import logging
import fitz  # PyMuPDF for PDF extraction
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env
load_dotenv()

# Get environment variables
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH")

class VectorStore:
    def __init__(self, store_path: str, embedding_model_name: str):
        self.store_path = store_path
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
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
            self.index = None
            logging.debug("Creating a new FAISS index store.")

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for a list of documents."""
        embeddings = self.model.encode(documents, show_progress_bar=True)
        logging.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def load_documents(self, pdf_directory: str) -> None:
        """Load and split documents from the specified directory."""
        documents = []

        # Iterate over all PDF files in the directory
        for pdf_file in os.listdir(pdf_directory):
            if pdf_file.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, pdf_file)
                logging.debug(f"Loading PDF: {file_path}")

                try:
                    # Load and extract text from each PDF using fitz
                    with fitz.open(file_path) as pdf_document:
                        for page_num in range(pdf_document.page_count):
                            page = pdf_document.load_page(page_num)
                            text = page.get_text()
                            if text.strip():  # Only add if the page contains text
                                documents.append(text)
                                logging.debug(f"Extracted text from page {page_num} of {pdf_file}: {text[:100]}...")  # Log first 100 characters
                except Exception as e:
                    logging.error(f"Error loading PDF {file_path}: {e}")
                    continue

        # Embed and store the documents if any are found
        if documents:
            self.texts.extend(documents)
            embeddings = self.embed_documents(documents)
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Initialize FAISS index with dimension
                logging.debug("Initialized new FAISS index.")
            self.index.add(embeddings)
            logging.debug(f"Added {embeddings.shape[0]} embeddings to the FAISS index.")
        else:
            logging.warning("No documents loaded; embeddings not created.")

    def save_store(self) -> None:
        """Save the FAISS index and texts to files."""
        if self.store_path:
            faiss.write_index(self.index, self.store_path)
            pickle_path = f"{self.store_path}.pickle"
            with open(pickle_path, "wb") as f:
                pickle.dump(self.texts, f)
            logging.debug(f"Index saved to {self.store_path} and texts saved to {pickle_path}")
        else:
            raise ValueError("Store path cannot be None or empty.")

    def query(self, query: str, k: int = 5) -> List[str]:
        """Search the FAISS index for the top k results for a given query."""
        if not self.texts:
            logging.debug("No texts available in the vector store.")
            return []

        query_embedding = self.embed_documents([query])
        logging.debug(f"Query embedding shape: {query_embedding.shape}")

        distances, indices = self.index.search(query_embedding, k)
        logging.debug(f"Distances: {distances}")
        logging.debug(f"Indices: {indices}")

        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.texts):
                results.append(self.texts[idx])
            else:
                logging.debug(f"Index out of range or not found: {idx}")

        if not results:
            logging.debug(f"No relevant texts found for query: {query}")

        logging.debug(f"Query results for '{query}':")
        for i, result in enumerate(results):
            logging.debug(f"Result {i + 1}: {result[:200]}...")  # Log first 200 characters

        return results

    def has_texts(self) -> bool:
        """Check if there are texts in the vector store."""
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





