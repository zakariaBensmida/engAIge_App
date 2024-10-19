from fastapi import FastAPI
from .query_handler import QueryHandler

app = FastAPI()
query_handler = QueryHandler()

@app.get("/ask")
def ask_question(query: str):
    """Endpoint to handle a question."""
    answer = query_handler.handle_query(query)
    return {"answer": answer}




