# Fairness Configuration for LoRA Fine-Tuning

FAIRNESS_CONFIG = {
    "fairness_criteria": {
        "demographic_parity": True,
        "equal_opportunity": True,
        "tpr_threshold": 0.05,  # Threshold for True Positive Rate difference
        "dp_threshold": 0.1,     # Threshold for Demographic Parity difference
    },
    "balancing_strategy": {
        "enabled": True,
        "method": "stratified",  # Method for balancing samples
        "groups": ["educational_background"],  # Groups to balance
    },
    "logging": {
        "log_fairness_metrics": True,
        "log_interval": 10,  # Log every 10 epochs
    },
    "evaluation": {
        "eval_on_fairness": True,
        "eval_interval": 5,  # Evaluate fairness every 5 epochs
    }
}