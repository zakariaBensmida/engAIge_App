import fitz  # PyMuPDF
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_folder: str, paragraph_delimiter: str = '\n\n'):
    """Extrahiere den Text aus allen PDFs im Ordner und gebe ein Dictionary von Absätzen zurück."""
    documents = {}
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            try:
                with fitz.open(filepath) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()  # Text aus jeder Seite extrahieren
                    paragraphs = text.split(paragraph_delimiter)  # Zerlegen in Absätze durch den angegebenen Trenner
                    documents[filename] = paragraphs
                    logging.info(f"Successfully extracted text from {filename}.")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
    return documents



