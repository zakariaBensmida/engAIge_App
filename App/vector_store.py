import os
import faiss
import numpy as np
import pickle
import logging
import fitz  # PyMuPDF for PDF extraction
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, store_path, embedding_model_name):
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

    def embed_documents(self, documents):
        """Generate embeddings for a list of documents."""
        embeddings = self.model.encode(documents, show_progress_bar=True)
        logging.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def load_documents(self, pdf_directory):
        """Load and split documents from the specified directory."""
        documents = []

        # Iterate over all PDF files in the directory
        for pdf_file in os.listdir(pdf_directory="./App/pdfs"):
            if pdf_file.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, pdf_file)
                logging.debug(f"Loading PDF: {file_path}")

                try:
                    # Load and extract text from each PDF
                    with fitz.open(file_path) as pdf_document:
                        text = ""
                        for page_num in range(pdf_document.page_count):
                            page = pdf_document.load_page(page_num)
                            text += page.get_text()
                        documents.append(text)
                        logging.debug(f"Extracted text from {pdf_file}: {text[:100]}...")  # Log first 100 characters
                except Exception as e:
                    logging.error(f"Error loading PDF {file_path}: {e}")
                    continue

        # Embed and store the documents
        self.texts.extend(documents)
        if documents:
            embeddings = self.embed_documents(documents)
            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Initialize FAISS index with dimension
                logging.debug("Initialized new FAISS index.")
            self.index.add(embeddings)
            logging.debug(f"Added {embeddings.shape[0]} embeddings to the FAISS index.")

        logging.debug(f"Loaded and added {len(documents)} documents to the vector store.")

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

        return results

    def has_texts(self):
        return len(self.texts) > 0

# Example usage for standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store/index.faiss")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/distiluse-base-multilingual-cased-v2")  # Update as necessary

    vector_store = VectorStore(store_path=store_path, embedding_model_name=embedding_model_name)

    # Load documents from the PDF directory
    pdf_directory = './path_to_your_pdf_directory'  # Specify your PDF directory
    vector_store.load_documents(pdf_directory)

    # Save the vector store
    vector_store.save_store()

    # Example query
    query = "Wie hoch ist die Grundzulage?"
    results = vector_store.query(query)
    print("Query Results:", results)


