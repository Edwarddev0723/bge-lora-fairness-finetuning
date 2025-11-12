import os
import json
import torch
from torch.utils.data import DataLoader
from src.data_loader import DataLoader as CustomDataLoader
from src.lora_model import LoRAModel
from src.trainer import Trainer
from src.utils import load_config

def main():
    # Load configurations
    model_config = load_config('configs/model_config.py')
    training_config = load_config('configs/training_config.py')
    fairness_config = load_config('configs/fairness_config.py')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data loader
    data_loader = CustomDataLoader(
        raw_data_path=os.path.join('data', 'raw'),
        processed_data_path=os.path.join('data', 'processed'),
        batch_size=training_config['batch_size'],
        balance_by='educational_background'
    )

    # Load the model
    model = LoRAModel(model_config)
    model.to(device)

    # Initialize trainer
    trainer = Trainer(model, data_loader, training_config, fairness_config, device)

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()