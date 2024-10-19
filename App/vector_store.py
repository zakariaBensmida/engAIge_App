from sentence_transformers import SentenceTransformer

def vectorize_text(paragraphs):
    """Vektorisiere eine Liste von Absätzen."""
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(paragraphs, convert_to_numpy=True)
    return embeddings





