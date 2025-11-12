from src.lora_model import LoRAModel
from src.trainer import Trainer
from src.data_loader import DataLoader
from src.fairness_metrics import calculate_fairness_metrics
import pytest

def test_model_training():
    # Initialize the model
    model = LoRAModel()
    
    # Load the dataset
    data_loader = DataLoader(batch_size=32)
    train_data, val_data = data_loader.load_data()
    
    # Initialize the trainer
    trainer = Trainer(model=model, train_data=train_data, val_data=val_data)
    
    # Train the model
    trainer.train(num_epochs=3)
    
    # Validate the model
    val_loss, val_accuracy = trainer.validate()
    
    assert val_loss < 0.5, "Validation loss is too high!"
    assert val_accuracy > 0.7, "Validation accuracy is below acceptable threshold!"

def test_fairness_metrics():
    # Simulate predictions and ground truth
    predictions = [0, 1, 0, 1, 0, 1]
    ground_truth = [0, 1, 1, 1, 0, 0]
    educational_backgrounds = ['high_school', 'college', 'high_school', 'college', 'high_school', 'college']
    
    # Calculate fairness metrics
    fairness_results = calculate_fairness_metrics(predictions, ground_truth, educational_backgrounds)
    
    assert fairness_results['demographic_parity'] >= 0.8, "Demographic parity is below acceptable level!"
    assert fairness_results['equal_opportunity'] >= 0.75, "Equal opportunity is below acceptable level!"