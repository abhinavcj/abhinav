# utils/data_loader.py
import os
import numpy as np
from utils.image_utils import preprocess_image

def load_images_and_labels(base_dir, target_size=(160, 160)):
    images = []
    labels = []

    for label_name in sorted(os.listdir(base_dir)):
        label_path = os.path.join(base_dir, label_name)
        try:
            label = int(label_name[-1])
        except:
            print(f"Skipping folder: {label_name}")
            continue

        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = preprocess_image(img_path, target_size)
            if img is not None:
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)
