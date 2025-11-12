import argparse
import torch
from src.data_loader import DataLoader
from src.lora_model import LoRAModel
from src.fairness_metrics import calculate_fairness_metrics
from src.utils import load_model
from configs.model_config import MODEL_CONFIG
from configs.training_config import TRAINING_CONFIG
from configs.fairness_config import FAIRNESS_CONFIG

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate the LoRA fine-tuned model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG['batch_size'], help="Batch size for evaluation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = LoRAModel(MODEL_CONFIG)
    load_model(model, args.model_path)
    model.to(device)

    # Load the test dataset
    data_loader = DataLoader(args.data_path, batch_size=args.batch_size)

    # Evaluate the model
    preds, labels = evaluate_model(model, data_loader, device)

    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(preds, labels, FAIRNESS_CONFIG)
    
    print("Evaluation completed.")
    print("Fairness Metrics:", fairness_metrics)

if __name__ == "__main__":
    main()