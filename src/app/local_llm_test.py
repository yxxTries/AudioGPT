from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_name="google/flan-t5-large"):
    print(f"[LLM] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def run_inference(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=1,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
