from src.preprocessing import preprocess_data, balance_by_education

def test_preprocess_data():
    # Sample raw data for testing
    raw_data = [
        {"text": "This is a sample text.", "education": "Bachelor"},
        {"text": "Another sample text.", "education": "Master"},
        {"text": "More sample text.", "education": "PhD"},
        {"text": "Text with no education.", "education": None},
    ]
    
    # Preprocess the data
    processed_data = preprocess_data(raw_data)
    
    # Check if the processed data is not empty
    assert len(processed_data) > 0, "Processed data should not be empty."
    
    # Check if all texts are cleaned (e.g., no unwanted characters)
    for entry in processed_data:
        assert isinstance(entry['text'], str), "Text should be a string."
    
def test_balance_by_education():
    # Sample processed data for testing
    processed_data = [
        {"text": "Sample text 1.", "education": "Bachelor"},
        {"text": "Sample text 2.", "education": "Master"},
        {"text": "Sample text 3.", "education": "Bachelor"},
        {"text": "Sample text 4.", "education": "PhD"},
        {"text": "Sample text 5.", "education": "Master"},
    ]
    
    # Balance the dataset by education
    balanced_data = balance_by_education(processed_data)
    
    # Check if the balanced data has the same number of samples for each education level
    education_counts = {}
    for entry in balanced_data:
        edu = entry['education']
        if edu in education_counts:
            education_counts[edu] += 1
        else:
            education_counts[edu] = 1
    
    # Ensure all education levels have the same count
    counts = list(education_counts.values())
    assert all(count == counts[0] for count in counts), "Data should be balanced across education levels."