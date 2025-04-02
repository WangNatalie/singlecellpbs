import pandas as pd
import numpy as np
import json
from helper_functions import mrrmse_np

def calculate_validation_score(submission_path, ground_truth_path):
    """
    Calculate MRRMSE score between submission and ground truth.
    
    Args:
        submission_path (str): Path to submission.csv
        ground_truth_path (str): Path to de_test_split.parquet
    
    Returns:
        float: MRRMSE score
    """
    # Read the files
    submission = pd.read_csv(submission_path)
    ground_truth = pd.read_parquet(ground_truth_path)
 
    # Get the prediction columns (all columns after the first 5)
    pred_columns = submission.columns[6:]
    
    # Extract predictions and ground truth values
    y_pred = submission[pred_columns].values
    y_true = ground_truth[pred_columns].values

    print(y_pred)
    print(y_true)
    # Print shapes for debugging
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_true shape: {y_true.shape}")
    
    # Calculate MRRMSE
    score = mrrmse_np(y_pred, y_true)
    
    return score
    

if __name__ == "__main__":
    # Read settings
    with open("./SETTINGS.json") as file:
        settings = json.load(file)
    
    # Paths to files
    submission_path = f'{settings["SUBMISSION_DIR"]}submission.csv'
    ground_truth_path = settings["TEST_RAW_DATA_PATH"]
    
    # Calculate and print score
    score = calculate_validation_score(submission_path, ground_truth_path)
    print(f"\nValidation MRRMSE Score: {score:.6f}") 