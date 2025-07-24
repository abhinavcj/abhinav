# utils/image_utils.py
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(160, 160)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)               # (160, 160, 3)
    img_array = img_array / 255.0                      # normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)     # (1, 160, 160, 3)
    return img_array