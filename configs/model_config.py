# Configuration settings for the BAAI/bge-large-en-v1.5 model with LoRA fine-tuning

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_TYPE = "transformers"
USE_LORA = True

# Hyperparameters for LoRA
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Model architecture settings
NUM_CLASSES = 10  # Adjust based on the number of classes in your dataset
MAX_LENGTH = 512  # Maximum input length for the model

# Training settings
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
EPOCHS = 3
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging settings
LOGGING_DIR = "./logs"
SAVE_MODEL_DIR = "./models/checkpoints"

# Fairness settings
FAIRNESS_METRICS = ["demographic_parity", "equal_opportunity"]
EDUCATIONAL_BACKGROUND_GROUPS = ["High School", "Bachelor's", "Master's", "PhD"]  # Example groups

# Seed for reproducibility
SEED = 42