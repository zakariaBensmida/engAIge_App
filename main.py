# main.py - FastAPI Entry Point
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from backend.query_handler import get_response
from backend.document_loader import load_documents
from backend.vector_manager import create_vector_store, embed_documents
from backend.models.llm import load_llm

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

documents = load_documents("Data/")  # Load web-scraped docs on startup
embeddings = embed_documents(documents)
vector_store = create_vector_store(embeddings)
llm = load_llm()

def stream_response(query: str):
    for chunk in get_response(query, documents, llm):
        yield chunk

@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        query = await websocket.receive_text()
        async for response_chunk in stream_response(query):
            await websocket.send_text(response_chunk)

# Run with: uvicorn main:app --reload
