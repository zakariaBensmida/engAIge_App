import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from .models import UploadResponse, QueryRequest, QueryResponse
from .pdf_extractor import extract_main_content
from .vector_store import VectorStore
from .query_handler import QueryHandler
from .config import get_llm, get_embeddings
from .utils import ensure_directory
import os

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Set up the templates directory
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../templates"))

# Initialize LLM and embeddings
llm = get_llm()
embeddings = get_embeddings()

# Ensure PDF storage directory exists
ensure_directory(os.getenv("PDF_STORAGE_PATH", "./pdfs"))

# Initialize VectorStore
vector_store = VectorStore(store_path=os.getenv("VECTOR_STORE_PATH", "./vector_store"), embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "distiluse-base-multilingual-cased-v2"))

# Initialize QueryHandler with the loaded LLM
query_handler = QueryHandler(llm=llm, vector_store=vector_store)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_location = os.path.join(os.getenv("PDF_STORAGE_PATH", "./pdfs"), file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Extract main content from PDF
    try:
        content = extract_main_content(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF content: {e}")
    
    # Add content to vector store
    texts = content.split('\n')
    vector_store.add_texts(texts, embeddings=embeddings)
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
        logging.debug(f"Retrieved relevant texts for query '{query}': {relevant_texts}")
        if not relevant_texts:
            raise HTTPException(status_code=404, detail="No relevant texts found.")
    except Exception as e:
        logging.error(f"Error querying vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying vector store: {e}")
    
    # Generate answer using LLM
    try:
        answer = query_handler.get_answer(relevant_texts, query)
        if not answer:
            raise HTTPException(status_code=500, detail="No answer generated.")
        logging.debug(f"Generated answer for query '{query}': {answer}")
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")
    
    return QueryResponse(answer=answer)

