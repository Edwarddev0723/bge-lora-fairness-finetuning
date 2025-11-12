from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn

class LoRAModel(nn.Module):
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", lora_rank=8):
        super(LoRAModel, self).__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_layers = self._init_lora_layers()

    def _init_lora_layers(self):
        lora_layers = {}
        for name, param in self.base_model.named_parameters():
            if "weight" in name:
                lora_layers[name] = nn.Parameter(torch.zeros_like(param))
        return lora_layers

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        
        # Apply LoRA adjustments
        for name, lora_param in self.lora_layers.items():
            if name in self.base_model.state_dict():
                logits += lora_param
        
        return outputs

    def save_pretrained(self, save_directory):
        self.base_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def load_pretrained(self, load_directory):
        self.base_model = AutoModelForSequenceClassification.from_pretrained(load_directory)
        self.tokenizer = AutoTokenizer.from_pretrained(load_directory)