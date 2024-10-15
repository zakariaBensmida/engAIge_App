
from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str
    filename: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
