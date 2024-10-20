
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the LLM model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, context: str, query: str, max_new_tokens: int = 50) -> str:
        """Generates an answer based on context and query."""
        input_text = f"Context: {context}\nQuery: {query}"
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # Generate a response with controlled max_new_tokens
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,  # Set the maximum number of new tokens
            do_sample=True,
            temperature=0.7,  # Adjust temperature for randomness
        )

        # Decode the output tokens back into a string
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    llm = LLM()
    context = "This is the context of relevant documents."
    query = "What is the content of the documents?"
    answer = llm.generate(context, query, max_new_tokens=100)  # Use max_new_tokens instead of max_length
    print(answer)

