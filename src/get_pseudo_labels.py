import config
import os
import pandas as pd
import numpy as np

STAGE = 'ss'
PATH_TO_ANOMS = config.PATH_TO_ANOM #dfs path

path_to_anoms = os.path.join(PATH_TO_ANOMS, STAGE)
seeds = [1001] #[1001,71530,138647,875688,985772,44,34,193,244959,8765]
EPOCH = 400

MOD_PREFIX = 'mod_1'
on_test_set = False
metric = 'centre_mean'


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

            df.columns = np.concatenate((df.columns.values[:-1] , np.array(['col_{}'.format(i)])))

            df['col_{}'.format(i)] = df['col_{}'.format(i)] / np.max(df['col_{}'.format(i)])
    
    df['av']=df.iloc[:,3:].mean(axis=1)
    df['std'] = df.iloc[:,3:-1].std(axis=1)

    print(df.head())
    
    





