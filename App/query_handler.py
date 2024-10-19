import os
import logging
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer
import torch
from .vector_store import VectorStore  # Import the VectorStore class

# Load environment variables
load_dotenv()

# Initialize the LLM for text generation
model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load environment variables for the vector store
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# Initialize the VectorStore
vector_store = VectorStore(store_path=VECTOR_STORE_PATH, embedding_model_name=EMBEDDING_MODEL_NAME)

def query_and_generate_answer(query: str) -> str:
    """Query the vector store, retrieve relevant documents, and generate an answer using the LLM."""
    logging.debug(f"Received query: {query}")
    
    # Query the vector store for relevant documents
    retrieved_texts = vector_store.query(query)
    
    if not retrieved_texts:
        return "No relevant information found."

    # Prepare the context for the LLM
    context = "\n".join(retrieved_texts)
    prompt = f"Answer the question using the context below.\nContext: {context}\nQuestion: {query}\nAnswer:"
    
    logging.debug(f"Generated prompt for LLM: {prompt[:500]}")  # Log first 500 chars of the prompt
    
    # Use the LLM to generate an answer based on the prompt
    answer = generate_answer(prompt)
    
    logging.debug(f"Generated answer: {answer}")
    return answer


def generate_answer(prompt: str) -> str:
    """Generate an answer using the LLM."""
    try:
        sequences = llm_pipe(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            top_k=10,
            return_full_text=False,
        )
        
        # Extract and return the generated answer from the sequence
        answer = sequences[0]['generated_text'].strip()
        
        # Optionally trim the answer to the first sentence
        answer = answer.split('.')[0] + '.' if '.' in answer else answer
        
        return answer
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer at this time."


# Example usage for testing the functions
if __name__ == "__main__":

    from .vector_store import VectorStore

    # Initialize the vector store (replace with your actual parameters)
    vector_store = VectorStore( store_path='./vector_store/index.faiss',embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME") )

    # Initialize QueryHandler
    query_handler = QueryHandler(vector_store=vector_store)

    # Example query
    
    query = "Wie hoch ist die Grundzulage?"
    answer = query_and_generate_answer(query)
    print("Answer:", answer)


