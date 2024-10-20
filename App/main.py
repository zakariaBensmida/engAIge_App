from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from query_handler import QueryHandler  # Ensure this import matches your file structure

app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize QueryHandler
handler = QueryHandler()  # Make sure to pass the required llm if necessary

# Example GET endpoint for the HTML page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST endpoint to process queries
@app.post("/query")
async def process_query(query: str = Form(...)):
    try:
        answer = handler.handle_query(query)  # Use the QueryHandler to get the answer
        if not answer:
            raise HTTPException(status_code=404, detail="Keine Antwort erhalten.")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)






