from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Datenmodell für die POST-Anfrage
class QueryRequest(BaseModel):
    query: str

# POST-Endpunkt, um eine Anfrage zu verarbeiten
@app.post("/query")
def process_query(request: QueryRequest):
    return {"answer": f"Die Anfrage lautet: {request.query}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)





