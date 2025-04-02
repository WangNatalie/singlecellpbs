import json
import numpy as np
from pathlib import Path

def calculate_fold_metrics(model_type, fold_num):
    """Calculate average MRRMSE for a specific model and fold."""
    file_path = f"results/{model_type}_light_fold{fold_num}.json"
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
    """Print average MRRMSE for each model and fold."""
    models = ['LSTM', 'GRU']
    folds = range(3)  # 0 to 2
    
    for model in models:
        print(f"\n{model} Results:")
        print("-" * 50)
        
        train_scores = []
        val_scores = []
        
        for fold in folds:
            train_avg, val_avg = calculate_fold_metrics(model, fold)
            if train_avg is not None:
                train_scores.append(train_avg)
                val_scores.append(val_avg)
                print(f"Fold {fold}:")
                print(f"  Training MRRMSE: {train_avg:.4f}")
                print(f"  Validation MRRMSE: {val_avg:.4f}")
                print()
        
        if train_scores:
            print(f"Average across folds 0-2:")
            print(f"  Training MRRMSE: {np.mean(train_scores):.4f}")
            print(f"  Validation MRRMSE: {np.mean(val_scores):.4f}")
            print()

if __name__ == "__main__":
    analyze_folds()