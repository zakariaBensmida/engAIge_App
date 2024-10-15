# app/query_handler.py
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import List

class QueryHandler:
    def __init__(self, llm_model_name: str):
        self.llm_model_name = llm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=-1)  # Use GPU if available
    
    def get_answer(self, context: List[str], question: str, max_length: int = 512) -> str:
        """
        Generates an answer based on the provided context and question.
        
        Args:
            context (List[str]): Relevant text chunks.
            question (str): User's question.
            max_length (int): Maximum length of the generated answer.
        
        Returns:
            str: Generated answer.
        """
        combined_context = "\n".join(context)
        prompt = f"Context:\n{combined_context}\n\nQuestion: {question}\nAnswer:"
        response = self.generator(prompt, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        answer = response[0]['generated_text'].split("Answer:")[-1].strip()
        return answer

