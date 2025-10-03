import config
import os
import pandas as pd
from IPython.display import display

base_dir = config.PROC_DATA_PATH

df_name = "inmodi_data_personalinformation_unpivoted.csv"
folder_name = "2025-07-03_data_exploration"

if __name__=='__main__':
    file_path = os.path.join(base_dir, folder_name, df_name)
    df = pd.read_csv(file_path)

    df['visit'] = df['visit'].astype(int).astype('str')

    #IM0001_1_left.npy

    df['file_name'] = df.apply(
    lambda row: f"{row['record_id']}_{row['visit']}_{'left' if row['side'] == 'l' else 'right'}.npy",
    axis=1)

    display(df.head())

    df.to_csv(file_path, index=False)

    