from utils.evaluation_utils import calc_mean_std, create_scores_dataframe
import config
import os
import pandas as pd

DATA_PATH = os.path.join(config.PATH_TO_RESULTS, 'ss')
SAVE_PATH = os.path.join(config.OUTPUT_PATH, "results", "ss")
PATH_TO_ANOM_SCORES = os.path.join(config.PATH_TO_ANOM, 'ss')

TRAIN_PLATEAU_EPOCH = 400

MOD_NAME="mod_2"

if __name__=="__main__":
    os.makedirs(SAVE_PATH, exist_ok=True)

    #files_total = os.listdir(DATA_PATH)
    files_total = os.listdir(PATH_TO_ANOM_SCORES)


    files = [file for file in files_total if (('epoch_' + str(TRAIN_PLATEAU_EPOCH) ) in file) & ('on_test_set' not in file ) & (MOD_NAME in file)]
    
    #df = pd.read_csv(os.path.join(DATA_PATH, files[0]))
    # print(df.head())
    df = create_scores_dataframe(PATH_TO_ANOM_SCORES, files, 'centre_mean')
    print(df.head())
    # means, stds = calc_mean_std(files, DATA_PATH, 'centre_mean')
    # print(means)
    