# backend/document_loader.py - Loads documents from data folder
def load_documents(data_path: str):
    import os
    docs = []
    for file in os.listdir(data_path):
        with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
            docs.append(f.read())
    return docs
