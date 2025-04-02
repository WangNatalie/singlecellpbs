import os
import time
import pandas as pd
import numpy as np
import json
from helper_functions import combine_features, load_trained_models, average_prediction

def read_data(settings):
    de_train = pd.read_parquet(settings["TRAIN_RAW_DATA_PATH"])
    id_map = pd.read_parquet(settings["TEST_RAW_DATA_PATH"])

    return de_train, id_map


if __name__ == "__main__":
    ## Read settings and config files
    with open("./SETTINGS.json") as file:
        settings = json.load(file)
        
    ## Read train, test and sample submission data # train data is needed for columns
    print("\nReading data...")
    de_train, id_map = read_data(settings)
    
    ## Build input features
    mean_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_cell_type.csv')
    std_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_cell_type.csv')
    mean_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_sm_name.csv')
    std_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_sm_name.csv')
    quantiles_df = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}quantiles_cell_type.csv')
    test_chem_feat = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_test.npy')
    test_chem_feat_mean = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_test_mean.npy')
    one_hot_test = pd.DataFrame(np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}one_hot_test.npy'))
    
    test_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    
    ## Load trained models
    print("\nLoading trained models...")
    trained_models = load_trained_models(path=f'{settings["MODEL_DIR"]}')
    fold_weights = [0.2421, 0.2814, 0.2500, 0.2088]
    ## Start predictions
    print("\nStarting predictions...")
    t0 = time.time()
    pred1 = average_prediction(test_vec_light, trained_models['light'])

    t1 = time.time()
    print("Prediction time: ", t1-t0, " seconds")
    print("\nEnsembling predictions and writing to file...")
    col = list(de_train.columns[5:])

    submission = []

    submission[col] = pred1

    if not os.path.exists(settings["SUBMISSION_DIR"]):
        os.mkdir(settings["SUBMISSION_DIR"])
    submission.to_csv(f'{settings["SUBMISSION_DIR"]}submission.csv')
    print("\nDone.")