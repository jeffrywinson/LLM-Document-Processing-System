# your-llm-project/llm_application/llm_handler.py

import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompt_template import MASTER_PROMPT
import config

def load_model_and_tokenizer():
    """Loads and returns the LLM model and tokenizer."""
    print(f"Loading {config.LLM_MODEL_ID} model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_ID,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_decision_json(model, tokenizer, query: str, retrieved_docs: list) -> dict:
    """Uses the model directly to generate a structured JSON decision."""
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = MASTER_PROMPT.format(context=context, query=query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        raw_response = full_response.split("<|assistant|>")[1]
        match = re.search(r"```json\s*\n(.*?)\n\s*```", raw_response, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    except Exception as e:
        print(f"Error parsing LLM response: {e}\nRaw Response: {full_response}")
        return {"decision": "Error", "amount": 0.0, "justification": "Failed to parse valid JSON."}