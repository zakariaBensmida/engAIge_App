
from transformers import pipeline

class LLM:
    def __init__(self, model_name="google/flan-t5-small"):
        """Initialize the language model for question-answering."""
        self.model = pipeline("text2text-generation", model=model_name)

    def generate(self, context: str, question: str) -> str:
        """Generate an answer based on the context and the question."""
        input_text = f"Question: {question}\nContext: {context}"
        response = self.model(input_text)
        return response[0]['generated_text']

# Example usage:
# llm = LLM()
# answer = llm.generate("Berlin is the capital of Germany.", "What is the capital of Germany?")
