from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import os
import json


def load_image_folder_as_array(folder, image_size=(64, 64)):
    X = []
    y = []
    names = []

    for label_folder in sorted(Path(folder).iterdir()):
        if label_folder.is_dir():
            label = label_folder.name
            for img_path in label_folder.glob("*.png"):  
                img = Image.open(img_path).convert("L").resize(image_size)
                X.append(np.array(img).flatten())  # Flatten to vector
                y.append(int(label))
                names.append(img_path.stem)
    return np.array(X), np.array(y), np.array(names)

def load_npy_folder_as_array(folder, flatten=True):
    X = []
    y = []
    names = []

    for label_folder in sorted(Path(folder).iterdir()):
        if label_folder.is_dir():
            label = label_folder.name
            #print(label)
            for npy_path in label_folder.glob("*.npy"):
                #print(npy_path)
                arr = np.load(npy_path)

                if flatten:
                    arr = arr.flatten()

                X.append(arr)
                y.append(int(label))
                names.append(npy_path.stem)

    return np.array(X), np.array(y), np.array(names)
def load_image(img):
    return Image.open(img)

def fix_id(id_str):
    if id_str.endswith('_l'):
        return id_str[:-2] + '_left'
    elif id_str.endswith('_r'):
        return id_str[:-2] + '_right'
    else:
        return id_str 
    


def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()  # or .tolist() if you prefer index-free
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    else:
        return obj  # already JSON-compatible
    
def get_next_run_folder(base_path):
    i = 1
    while True:
        folder_name = f"run{i}"
        full_path = os.path.join(base_path, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return folder_name, full_path
        i += 1


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_csv(self, filename: str):
        df_path = os.path.join(self.data_path, filename)
        df = pd.read_csv(df_path)
        return df
    
    def load_json(self, filename: str):
        json_path = os.path.join(self.data_path, filename)
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def load_numpy(self, filename: str):
        npy_path = os.path.join(self.data_path, filename)
        data = np.load(npy_path)
        return data

class HDBSCAN_DataLoader(DataLoader):
    def __init__(self, base_path: str, folder: str, run: str, modality: str = 'pipeline'):
        file_path = os.path.join(base_path, folder, modality, run)
        self.base_path = base_path
        super().__init__(file_path)
        self.run = run
        self.modality = modality
        self.df = None
        self.embeddings = None
        self.ids = None

    def load_pipeline_data(self):
        df_filename = f'pipeline_{self.run}_umap_hdbscan_scaled_allpoints_wKL.csv'
        json_filename = f'pipeline_{self.run}_umap_hdbscan_scaled_model_info.json'
        embeddings_filename = 'X_umap_embeddings.npy'
        
        df = self.load_csv(df_filename)
        model_info = self.load_json(json_filename)
        ids = model_info['files']['ids']
        embeddings = self.load_numpy(embeddings_filename)

        self.df = df
        self.embeddings = embeddings
        self.ids = ids
        
        return df, model_info, embeddings, ids
    
    def get_mapping(self, columns: list = ['cluster_label', 'KL-Score']):
        if self.df is None:
            df, _, _, ids = self.load_pipeline_data()
        else:
            df = self.df
            # embeddings = self.embeddings
            ids = self.ids
        
        base_mapping = {
            i: {
                col: df.loc[df['id'] == i, col].values[0]
                for col in columns
            }
            for i in ids
        }
        return base_mapping
    
    def load_data_by_kl(self, columns: list = ['cluster_label', 'KL-Score']):
        if self.df is None:
            df, _, embeddings, ids = self.load_pipeline_data()
        else:
            df = self.df
            embeddings = self.embeddings
            ids = self.ids
        
        kl_values = df['KL-Score'].unique()
        
        result = {}
        embeddings_kl_d = {}
        for kl in kl_values:
            df_kl = df[df['KL-Score'] == kl]
            #embeddings_kl = embeddings[df_kl.index, :]
            ids_kl = df_kl['id'].values

            #get index in ids for ids in ids_kl
            idx_kl = [np.where(ids == np.array(id_))[0][0] for id_ in ids_kl]

            mapping_kl = {
                i: {
                    col: df_kl.loc[df_kl['id'] == i, col].values[0]
                    for col in columns
                }
                for i in ids_kl
            }

            embeddings_kl = embeddings[idx_kl, :]
            embeddings_kl_d[str(int(kl))] = embeddings_kl
            result[str(int(kl))] = mapping_kl

        return result, embeddings_kl_d
    
    def load_data_by_filter(self, filter_column: str, filter_value):
        if self.df is None:
            df, _, embeddings, ids = self.load_pipeline_data()
        else:
            df = self.df
            embeddings = self.embeddings
            ids = self.ids
        
        df_filt = df[df[filter_column] == filter_value]
        ids_filt = df_filt['id'].values

        #get index in ids for ids in ids_filt
        idx_filt = [np.where(ids == np.array(id_))[0][0] for id_ in ids_filt]

        mapping_filt = {
            i: {
                "cluster_label": df_filt.loc[df_filt['id'] == i, 'cluster_label'].values[0],
                "KL-Score": df_filt.loc[df_filt['id'] == i, 'KL-Score'].values[0]
            }
            for i in ids_filt
        }

        embeddings_filt = embeddings[idx_filt, :]

        return mapping_filt, embeddings_filt
    
    def load_data_by_binaryfilter(self, filter_column: str, filter_values: int):
        if self.df is None:
            df, _, embeddings, ids = self.load_pipeline_data()
        else:
            df = self.df
            embeddings = self.embeddings
            ids = self.ids

        df['filter_temp'] = df[filter_column].apply(lambda x: 'other' if x != filter_values else "noise")

        mapping_filt = {
            i: {
                "cluster_label": df.loc[df['id'] == i, 'cluster_label'].values[0],
                "KL-Score": df.loc[df['id'] == i, 'KL-Score'].values[0],
                "noise_label": df.loc[df['id'] == i, 'filter_temp'].values[0],
            }
            for i in ids
        }

        return mapping_filt

class ExtendedDataLoader(HDBSCAN_DataLoader):
    def __init__(self, raw_data_path: str, base_path: str, folder: str, run: str, modality: str = 'pipeline'):
        super().__init__(base_path, folder, run, modality)
        self.raw_data_path = raw_data_path
        self.as_df =None

    def merge_mri_data(self, mri_filename: str):
        if self.df is None:
            raise ValueError("Call load_pipeline_data() before merging MRI data.")
        
        mri_path = os.path.join(self.raw_data_path, mri_filename)
        mri_df = pd.read_csv(mri_path)

        merge_cols = [
            'mri_bml_yn', 'mri_cart_yn', 'mri_osteo_yn',
            'mri_syn_yn', 'mri_mnsc_yn', 'mri_lig_yn'
        ]

        self.df = self.df.merge(
            mri_df[['id'] + merge_cols],
            on='id',
            how='left',
            validate='one_to_one'
        )

        return self.df
    
    def clean_id(self, id_column: str = 'id', split_char: str = '.'):
        self.as_df[id_column] = self.as_df[id_column].apply(lambda x: os.path.basename(x).split(split_char)[0])
        return self.as_df
    
    def load_anomaly_scores(self, as_folder: str, as_file: str):
        as_path = os.path.join(self.base_path, as_folder)
        as_dataloader = DataLoader(as_path)
        self.as_df = as_dataloader.load_csv(as_file)
        self.as_df = self.clean_id(id_column='id', split_char='.')
        return self.as_df
    
    def merge_anomaly_scores(self):
        if self.as_df is None:
            raise ValueError("Call load_anomaly_scores() before merging anomaly scores.")
        
        self.df = self.df.merge(
            self.as_df,
            on='id',
            how='left',
            validate='one_to_one'
        )
        return self.df
