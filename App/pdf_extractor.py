import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_folder: str):
    """Extrahiere den Text aus allen PDFs im Ordner und gebe eine Liste von Absätzen zurück."""
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            with fitz.open(filepath) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()  # Text aus jeder Seite extrahieren
                paragraphs = text.split('\n\n')  # Zerlegen in Absätze durch doppelte Zeilenumbrüche
                documents.append(paragraphs)
    return documents


