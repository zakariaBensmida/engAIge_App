import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from App.models import UploadResponse, QueryRequest, QueryResponse
from App.pdf_extractor import extract_main_content
from App.vector_store import VectorStore
from App.query_handler import QueryHandler
from App.config import get_llm, get_embeddings
from App.utils import ensure_directory

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Set up the templates directory
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../templates"))

# Initialize LLM and embeddings
llm = get_llm()
embeddings = get_embeddings()

# Ensure PDF storage directory exists
pdf_storage_path = os.getenv("PDF_STORAGE_PATH", "./App/pdfs")
ensure_directory(pdf_storage_path)

# Initialize VectorStore
vector_store = VectorStore(
    store_path=os.getenv("VECTOR_STORE_PATH", "./vector_store/index.faiss"),
    embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "distiluse-base-multilingual-cased-v2")
)

# Initialize QueryHandler with the loaded LLM
query_handler = QueryHandler(vector_store=vector_store)  # LLM is now initialized inside QueryHandler


# Function to remove redundant phrases in the generated answer
def remove_redundant_phrases(text):
    sentences = text.split('. ')
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return '. '.join(unique_sentences)


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_location = os.path.join(pdf_storage_path, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    logging.debug(f"PDF saved at: {file_location}")

    # Extract main content from PDF with chunking
    try:
        chunks = extract_main_content(file_location, chunk_size=300, overlap=50)  # Improved chunking
        logging.debug(f"Extracted {len(chunks)} content chunks")
    except Exception as e:
        logging.error(f"Error extracting PDF content: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting PDF content: {e}")
    
    # Add content chunks to vector store
    vector_store.add_texts(chunks)
    vector_store.save_store()
    
    return UploadResponse(message="PDF uploaded and content extracted successfully.", filename=file.filename)


@app.post("/query", response_model=QueryResponse)
def query_pdf(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    logging.debug(f"Query received: {query}")

    # Generate answer using LLM with response fine-tuning parameters
    try:
        answer = query_handler.get_answer(query, temperature=0.7, max_tokens=150, top_p=0.9)  # Customize LLM response
        if not answer:
            raise HTTPException(status_code=500, detail="No answer generated.")
        logging.debug(f"Generated answer for query '{query}': {answer}")
        
        # Remove redundant phrases in post-processing
        cleaned_answer = remove_redundant_phrases(answer)
        logging.debug(f"Cleaned answer: {cleaned_answer}")
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")
    
    return QueryResponse(answer=cleaned_answer)


