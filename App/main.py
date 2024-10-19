from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

# Initialisierung der Jinja2-Templates
templates = Jinja2Templates(directory="templates")

# Beispiel-GET-Endpunkt für die HTML-Seite
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST-Endpunkt zum Verarbeiten von Anfragen
@app.post("/query")
def process_query(query: str = Form(...)):
    # Simulierte Logik zur Beantwortung der Anfrage
    answer = f"Die Antwort auf '{query}' ist: [hier die Antwort einfügen]."
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)





