# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.models import UploadResponse, QueryRequest, QueryResponse
from app.pdf_extractor import extract_main_content
from app.vector_store import VectorStore
from app.query_handler import QueryHandler
from app.config import settings
from app.utils import ensure_directory
import os

app = FastAPI(title="LangChain RAG Chatbot (German)")

# Ensure PDF storage directory exists
ensure_directory(settings.PDF_STORAGE_PATH)

# Initialize VectorStore
vector_store = VectorStore(store_path=settings.VECTOR_STORE_PATH, embedding_model_name=settings.EMBEDDING_MODEL_NAME)

# Initialize QueryHandler
query_handler = QueryHandler(llm_model_name=settings.LLM_MODEL_NAME)

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_location = os.path.join(settings.PDF_STORAGE_PATH, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Extract main content from PDF
    try:
        content = extract_main_content(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF content: {e}")
    
    # Add content to vector store
    texts = content.split('\n')
    vector_store.add_texts(texts)
    vector_store.save_store()
    
    return UploadResponse(message="PDF uploaded and content extracted successfully.", filename=file.filename)

@app.post("/query", response_model=QueryResponse)
def query_pdf(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # Retrieve relevant documents
    try:
        relevant_texts = vector_store.query(query, k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying vector store: {e}")
    
    # Generate answer using LLM
    try:
        answer = query_handler.get_answer(relevant_texts, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")
    
    return QueryResponse(answer=answer)

