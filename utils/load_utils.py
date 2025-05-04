from pathlib import Path
from PIL import Image
import numpy as np

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
