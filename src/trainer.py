from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import torch
from src.lora_model import LoRAModel
from src.fairness_metrics import calculate_fairness_metrics
from src.utils import save_model, load_model

class Trainer:
    def __init__(self, model_name, train_dataset, val_dataset, config):
        self.model = LoRAModel(model_name)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            evaluation_strategy="epoch",
            learning_rate=self.config['learning_rate'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            num_train_epochs=self.config['num_epochs'],
            weight_decay=self.config['weight_decay'],
            logging_dir=self.config['logging_dir'],
            logging_steps=self.config['logging_steps'],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        save_model(self.model, self.config['model_save_path'])

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        accuracy = np.sum(preds == labels) / len(labels)
        fairness_metrics = calculate_fairness_metrics(preds, labels, self.val_dataset)
        return {
            'accuracy': accuracy,
            **fairness_metrics
        }