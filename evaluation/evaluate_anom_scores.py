import config
import os
import pandas as pd
from utils.label_eval_utils import *

x = 'label'
y = 'anoms_count'
values = ['label', 'anoms_count', 'sim', 'av']

SAVE_PATH = config.OUTPUT_PATH
pseudolabels_path= os.path.join(SAVE_PATH, 'pseudolabels')

pseudolabel_name = "ss_training_mod_2_epoch_400_margin_1.130199999.csv"

save_path = os.path.join(SAVE_PATH, 'graphs')
os.makedirs(save_path, exist_ok=True)

if __name__=="__main__":
    df = pd.read_csv(os.path.join(pseudolabels_path, pseudolabel_name), index_col=False)

    scatter_plot(df, save_path, x = x, y = y)

