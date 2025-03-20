def get_response(query: str, documents: list, llm):
    print(f"Received query: {query}")  # Debugging output
    response = llm(query)
    print(f"LLM raw response: {response}")  # Debugging output

    for chunk in response:
        print(f"Streaming chunk: {chunk['generated_text']}")  # Debugging output
        yield chunk['generated_text']


