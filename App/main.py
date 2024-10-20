# main.py
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
from .query_handler import QueryHandler
from .llm import LLM
from .vector_store import VectorStore

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Get the model names and paths from the environment variables
llm_model_name = os.getenv('LLM_MODEL_NAME', 'bigscience/bloom-560m')
vector_store_path = os.getenv('VECTOR_STORE_PATH', 'C:\\Users\\zakar\\engAIge_App\\App\\vector_store.faiss')
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', 'distiluse-base-multilingual-cased-v2')

# Create an instance of the LLM class
llm_instance = LLM(model_name=llm_model_name)

# Initialize the VectorStore
vector_store_instance = VectorStore(vector_store_path, embedding_model_name)

# Optionally, add some initial texts to the vector store
initial_texts = [
    "This is the first document.",
    "Here is the second document.",
    "And this is the third one."
]
vector_store_instance.add_texts(initial_texts)  # Populate with initial texts

# Initialize QueryHandler with the LLM instance and VectorStore instance
handler = QueryHandler(llm=llm_instance, vector_store=vector_store_instance)

# Example GET endpoint for the HTML page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST endpoint to process queries
@app.post("/query")
async def process_query(query: str = Form(...)):
    try:
        answer = handler.handle_query(query)
        if not answer:
            raise HTTPException(status_code=404, detail="Keine Antwort erhalten.")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)








