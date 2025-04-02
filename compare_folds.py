import json
import numpy as np
from pathlib import Path

def calculate_fold_metrics(fold_num):
    """Calculate average MRRMSE for a specific fold."""
    file_path = f"results/LSTM_light_fold{fold_num}.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Calculate average of last 10 epochs for both train and val
        train_avg = np.mean(data['train_mrrmse'][-10:])
        val_avg = np.mean(data['val_mrrmse'][-10:])
        return train_avg, val_avg
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return None, None

def analyze_folds():
    """Print average MRRMSE for each fold."""
    folds = range(4)  # 0 to 3
    
    print("Fold Averages:")
    print("-" * 50)
    
    for fold in folds:
        train_avg, val_avg = calculate_fold_metrics(fold)
        if train_avg is not None:
            print(f"Fold {fold}:")
            print(f"  Training MRRMSE: {train_avg:.4f}")
            print(f"  Validation MRRMSE: {val_avg:.4f}")
            print()

if __name__ == "__main__":
    analyze_folds()