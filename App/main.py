from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Assuming vector_store_instance is already created
vector_store_instance = None  # Initialize your vector store instance here

@app.post("/query")
async def query(request: QueryRequest):
    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Vector store instance not initialized.")
    
    try:
        response = vector_store_instance.retrieve(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Add the code to initialize your vector store instance
    pass









