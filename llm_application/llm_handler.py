# llm_application/llm_handler.py (CPU Version)

import torch
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from .prompt_template import MASTER_PROMPT

def load_pipeline():
    """Loads and returns the LLM pipeline to run on the CPU."""
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    print("Loading Phi-3-mini model for CPU...")
    
    # Load the model without any quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create the pipeline. It will default to CPU if no compatible GPU is found.
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    print("LLM pipeline loaded successfully.")
    return llm_pipeline

def generate_decision_json(llm_pipeline, query: str, retrieved_docs: list) -> dict:
    """Uses the LLM to generate a structured JSON decision."""
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = MASTER_PROMPT.format(context=context, query=query)
    
    outputs = llm_pipeline(
        prompt, max_new_tokens=256, do_sample=False,
        eos_token_id=llm_pipeline.tokenizer.eos_token_id
    )
    raw_response = outputs[0]['generated_text'].split("<|assistant|>")[1]
    
    try:
        json_part = raw_response[raw_response.find('{'):raw_response.rfind('}')+1]
        decision = json.loads(json_part)
        return decision
    except Exception as e:
        print(f"Error parsing LLM response: {e}\nRaw Response: {raw_response}")
        return {"decision": "Error", "amount": 0.0, "justification": "Failed to parse valid JSON."}