import pytest
from src.fairness_metrics import calculate_demographic_parity, calculate_equal_opportunity

def test_demographic_parity():
    # Sample predictions and ground truth
    predictions = [0, 1, 1, 0, 1, 0]
    ground_truth = [0, 1, 0, 0, 1, 1]
    groups = [0, 1, 1, 0, 1, 0]  # Educational background groups

    dp = calculate_demographic_parity(predictions, ground_truth, groups)
    assert dp >= 0, "Demographic parity should be non-negative"

def test_equal_opportunity():
    # Sample predictions and ground truth
    predictions = [0, 1, 1, 0, 1, 0]
    ground_truth = [0, 1, 0, 0, 1, 1]
    groups = [0, 1, 1, 0, 1, 0]  # Educational background groups

    eo = calculate_equal_opportunity(predictions, ground_truth, groups)
    assert eo >= 0, "Equal opportunity should be non-negative"