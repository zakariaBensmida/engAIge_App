# tests/test_query_handler.py - Tests query response
from backend.query_handler import get_response

def test_get_response():
    documents = ["This is a test document."]
    query = "What is this?"
    response = list(get_response(query, documents))
    assert isinstance(response, list)
    assert len(response) > 0
