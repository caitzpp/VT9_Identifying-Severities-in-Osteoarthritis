from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import os

def load_image_folder_as_array(folder, image_size=(64, 64)):
    X = []
    y = []
    for label_folder in sorted(Path(folder).iterdir()):
        if label_folder.is_dir():
            label = label_folder.name
            for img_path in label_folder.glob("*.png"):  
                img = Image.open(img_path).convert("L").resize(image_size)
                X.append(np.array(img).flatten())  # Flatten to vector
                y.append(int(label))
    return np.array(X), np.array(y)

def load_npy_folder_as_array(folder, flatten=True):
    X = []
    y = []

    for label_folder in sorted(Path(folder).iterdir()):
        if label_folder.is_dir():
            label = label_folder.name
            for npy_path in label_folder.glob("*.npy"):
                arr = np.load(npy_path)

                if flatten:
                    arr = arr.flatten()

                X.append(arr)
                y.append(int(label))

    return np.array(X), np.array(y)
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