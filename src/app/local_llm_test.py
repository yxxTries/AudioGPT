from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

def load_model(model_name="Qwen/Qwen2-0.5B-Instruct"):
    print(f"[LLM] Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def run_inference(text, tokenizer, model, stream=True):
    # Qwen2 chat format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Create streamer for real-time output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer
    )
    
    # Decode only the new tokens (exclude input)
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def main():
    tokenizer, model = load_model()

    # Example input text
    # Read input text from src/app/output.txt
    input_path = "src/app/output.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    print(f"[INPUT] {input_text}")

    result = run_inference(input_text, tokenizer, model)
    print("\n[OUTPUT]")
    print(result)

if __name__ == "__main__":
    main()
