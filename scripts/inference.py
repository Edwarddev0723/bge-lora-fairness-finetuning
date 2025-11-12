from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from src.data_loader import DataLoader
from src.lora_model import LoRAModel
from src.utils import load_model

def load_inference_model(model_path):
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    return model, tokenizer

def run_inference(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs

def main():
    model_path = os.path.join("models", "lora_adapters", "your_model_adapter")  # Update with your model adapter path
    model, tokenizer = load_inference_model(model_path)

    input_text = "Your input text here."  # Replace with your input text
    outputs = run_inference(model, tokenizer, input_text)

    print("Inference Outputs:", outputs)

if __name__ == "__main__":
    main()