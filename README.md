# BGE LoRA Fairness Fine-Tuning

This project aims to fine-tune the BAAI/bge-large-en-v1.5 model using Low-Rank Adaptation (LoRA) techniques while ensuring fairness across different educational background groups. The goal is to develop a model that not only performs well on the task but also maintains equitable performance across diverse demographic groups.

## Project Structure

The project is organized as follows:

- **src/**: Contains the main source code for data loading, preprocessing, model definition, training, and evaluation.
  - **data_loader.py**: Handles loading and batching the dataset, including balancing samples across educational backgrounds.
  - **preprocessing.py**: Includes functions for data preprocessing and balancing the dataset.
  - **lora_model.py**: Defines the LoRAModel class for the BAAI/bge-large-en-v1.5 model with LoRA fine-tuning.
  - **fairness_metrics.py**: Provides functions to calculate fairness metrics for model evaluation.
  - **trainer.py**: Manages the training loop, including validation and logging of metrics.
  - **utils.py**: Contains utility functions for saving/loading models and handling configurations.

- **configs/**: Contains configuration files for model, training, and fairness settings.
  - **model_config.py**: Model architecture and hyperparameters.
  - **training_config.py**: Training parameters like learning rate and batch size.
  - **fairness_config.py**: Settings related to fairness-aware training.

- **data/**: Directory for storing raw and processed dataset files.
  - **raw/**: For raw dataset files.
  - **processed/**: For processed dataset files.

- **notebooks/**: Jupyter notebooks for data exploration, model training, and fairness evaluation.
  - **01_data_exploration.ipynb**: Explore the dataset and visualize distributions.
  - **02_model_training.ipynb**: Workflow for training the LoRA model.
  - **03_fairness_evaluation.ipynb**: Evaluate the model's fairness metrics.

- **models/**: Directory for storing model checkpoints and trained LoRA adapters.
  - **checkpoints/**: Model checkpoints during training.
  - **lora_adapters/**: Trained LoRA adapters.

- **reports/**: Directory for storing figures and metrics reports.
  - **figures/**: Figures generated during analysis.
  - **metrics/**: Metrics reports after model evaluation.

- **tests/**: Contains unit tests for various components of the project.
  - **test_preprocessing.py**: Unit tests for preprocessing functions.
  - **test_model.py**: Unit tests for model training and evaluation.
  - **test_fairness.py**: Unit tests for fairness metrics calculations.

- **scripts/**: Scripts for training, evaluating, and running inference with the model.
  - **train.py**: Initiates the training process.
  - **evaluate.py**: Evaluates the trained model on the test set.
  - **inference.py**: Runs inference on new data.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd bge-lora-fairness-finetuning
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place your raw dataset files in the `data/raw` directory. Use the preprocessing functions to clean and balance the dataset based on educational backgrounds.

2. **Training the Model**: Run the training script or use the Jupyter notebook `02_model_training.ipynb` to fine-tune the model with LoRA techniques.

3. **Evaluating Fairness**: After training, evaluate the model's performance and fairness metrics using the `03_fairness_evaluation.ipynb` notebook.

4. **Inference**: Use the `scripts/inference.py` script to run inference on new data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.