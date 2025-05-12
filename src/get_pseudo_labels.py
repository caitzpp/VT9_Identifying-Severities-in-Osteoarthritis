import config
import os
import pandas as pd
import numpy as np

STAGE = 'ss'
EPOCH = 400
MOD_PREFIX = 'mod_2'
on_test_set = False
metric = 'centre_mean'
margin = 1.130199999
meta_data = "meta"

PATH_TO_ANOMS = config.PATH_TO_ANOM #dfs path
SAVE_PATH = config.OUTPUT_PATH
save_path = os.path.join(SAVE_PATH, 'pseudolabels')
sim_path = os.path.join(config.DIR_PATH, meta_data)
os.makedirs(save_path, exist_ok=True)


path_to_anoms = os.path.join(PATH_TO_ANOMS, STAGE)
seeds = [1001, 71530] #[1001,71530,138647,875688,985772,44,34,193,244959,8765]


if __name__=="__main__":
    files_total = os.listdir(path_to_anoms)

    if on_test_set==True:
        files = [f for f in files_total if ( ('epoch_' + str(EPOCH) in f) &  ('on_test_set_' in f)  &  (MOD_PREFIX in f)  ) ]
    else:
        files = [f for f in files_total if ( ('epoch_' + str(EPOCH) in f) &  ('on_test_set_' not in f)  &  (MOD_PREFIX in f)  ) ]
    for i,seed in enumerate(seeds):

        for file in files:
          if (('_' + str(seed) + '_') in file) :
            logs = pd.read_csv(os.path.join(path_to_anoms, file))

            if i ==0:
                df = logs.iloc[:,:3]

            scores = logs[['id',metric]]
            if metric == 'w_centre':
                scores.loc[:,metric] = (scores.loc[:,metric] + 2) / 4
            else:
                scores.loc[:,metric] = (scores.loc[:,metric] + 1) / 2

            df = df.merge(scores, on='id', how='left')

            df.columns = np.concatenate((df.columns.values[:-1] , np.array(['col_{}'.format(seed)])))

            df['col_{}'.format(seed)] = df['col_{}'.format(seed)] / np.max(df['col_{}'.format(seed)])
    
    df['av']=df.iloc[:,3:].mean(axis=1)
    df['std'] = df.iloc[:,3:-1].std(axis=1)

    # TODO: mark the cases where the anomaly score is larger than margin * np.max('av'), need to "count" them per seed
    for seed in seeds:
        df[f'anom_{seed}'] = 0

    df["anoms_count"]= 0
    for i in range(len(df)):
        for j in range(len(seeds)):
            seed = seeds[j]

            if df[f'col_{seed}'].iloc[i] > (margin * np.max(df['av'])): #here using the average over the entire train df
                df.loc[i, f'anom_{seed}']=1
                df.loc[i, f'anoms_count']+=1
    df['id'] = df['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1])
    sim = pd.read_csv(os.path.join(sim_path, "sim_scores.csv"), index_col=False)
    sim = sim.iloc[:, 2:]
    sim['id'] = sim['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    df = df.merge(sim,on='id', how='left')
    df['sim>95th'] = df['sim'].apply(lambda x: 0 if x < np.percentile(df['sim'], 95) else 1) # 1 if larger than 95th percentile
    if on_test_set==True:
        df.to_csv(os.path.join(save_path, f"{STAGE}_training_{MOD_PREFIX}_on_test_set_epoch_{EPOCH}_margin_{margin}.csv"), index=False)
    else:
        df.to_csv(os.path.join(save_path, f"{STAGE}_training_{MOD_PREFIX}_epoch_{EPOCH}_margin_{margin}.csv"), index=False)

    
    





