# backend/query_handler.py - Handles user queries
def get_response(query: str, documents: list):
    from langchain.llms import HuggingFacePipeline
    from transformers import pipeline
    
    llm_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct")
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    response = llm(query)
    for chunk in response:
        yield chunk['generated_text']
