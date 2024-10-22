import json
import pandas as pd

def preprocess_data(file_path):
    """Load and preprocess the MedQuAD dataset."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check if 'question' and 'answer' keys exist
        if isinstance(data, list) and all('question' in d and 'answer' in d for d in data):
            return pd.DataFrame(data)
        else:
            print("Error: 'question' or 'answer' keys missing in the dataset.")
            return pd.DataFrame()  # Return an empty DataFrame if keys are missing

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return pd.DataFrame()
