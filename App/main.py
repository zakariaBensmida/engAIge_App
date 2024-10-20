import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel  # Import BaseModel for Pydantic models
from .pdf_extractor import extract_text_from_pdf
from .vector_store import VectorStore
from .query_handler import QueryHandler
from .config import get_llm, get_embeddings
from .utils import ensure_directory

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Set up the templates directory
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../templates"))

# Initialize LLM and embeddings
llm = get_llm()
embeddings = get_embeddings()

# Ensure PDF storage directory exists
pdf_storage_path = os.getenv("PDF_STORAGE_PATH")
ensure_directory(pdf_storage_path)

# Initialize VectorStore
vector_store = VectorStore(
    store_path=os.getenv("VECTOR_STORE_PATH", "./vector_store/index.faiss"),
    embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "distiluse-base-multilingual-cased-v2")
)

# Define Pydantic models
class UploadResponse(BaseModel):
    message: str
    filename: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# Initialize QueryHandler
query_handler = QueryHandler(vector_store=vector_store, llm=llm)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_location = os.path.join(pdf_storage_path, file.filename)
    
    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        logging.error(f"Error saving PDF file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving PDF file: {e}")

    logging.debug(f"PDF saved at: {file_location}")

    # Extract main content from PDF
    try:
        content = extract_text_from_pdf(file_location)
        logging.debug(f"Extracted content length: {len(content)}")
    except Exception as e:
        logging.error(f"Error extracting PDF content: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting PDF content: {e}")

    # Add content to vector store
    texts = content.split('\n')
    logging.debug(f"Number of texts to add: {len(texts)}")

    vector_store.add_texts(texts)
    vector_store.save_store()

    return UploadResponse(message="PDF uploaded and content extracted successfully.", filename=file.filename)

@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logging.debug(f"Query received: {query}")

    # Generate answer using LLM
    try:
        answer = query_handler.get_answer(query)
        if not answer:
            raise HTTPException(status_code=500, detail="No answer generated.")
        logging.debug(f"Generated answer for query '{query}': {answer}")
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    return QueryResponse(answer=answer)










