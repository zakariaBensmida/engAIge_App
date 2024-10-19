from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Beispiel-Datenmodell
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def process_query(request: QueryRequest):
    # Hier kommt die Logik für die Anfrageverarbeitung hin
    return {"answer": f"Die Anfrage lautet: {request.query}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



