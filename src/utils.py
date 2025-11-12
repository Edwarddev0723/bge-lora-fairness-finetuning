def save_model(model, path):
    model.save_pretrained(path)

def load_model(model_class, path):
    return model_class.from_pretrained(path)

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)