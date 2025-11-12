# Training configuration for LoRA fine-tuning of BAAI/bge-large-en-v1.5 model

# Training parameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
NUM_EPOCHS = 5
MAX_LENGTH = 256
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2

# Fairness-aware training parameters
FAIRNESS_METRIC_THRESHOLD = 0.1
BALANCE_SAMPLES = True  # Whether to balance samples across educational backgrounds
EDUCATIONAL_BACKGROUND_GROUPS = ['High School', 'Bachelor', 'Master', 'PhD']  # Define the groups for balancing

# Logging and saving parameters
LOGGING_STEPS = 100
SAVE_STEPS = 500
OUTPUT_DIR = './models/checkpoints'  # Directory to save model checkpoints

# Seed for reproducibility
SEED = 42