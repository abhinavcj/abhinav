
# Osteoarthritis Grade Detection using Deep Learning

This project uses a deep learning model to automatically classify the severity of knee osteoarthritis from X-ray images. It leverages **transfer learning** with a **MobileNetV2** architecture to classify images into five Kellgren-Lawrence (KL) grades (0-4).

![Sample Prediction](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-023-50210-4/MediaObjects/41598_2023_50210_Fig1_HTML.png) **grading**
* [Key Features](#key-features)
* [Model Architecture](#model-architecture)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
  * [Training the Model](#training-the-model)
  * [Running Predictions](#running-predictions)
* [Results](#results)
* [File Structure](#file-structure)

## Key Features
* **Deep Learning Model**: Utilizes the efficient and powerful MobileNetV2 architecture.
* **Transfer Learning**: Employs weights pre-trained on ImageNet for faster convergence and better performance.
* **Two-Phase Fine-Tuning**:
    1.  Trains the top classification layers.
    2.  Unfreezes the base model for fine-tuning with a low learning rate.
* **Data Preprocessing**: Includes a pipeline for image cleaning and enhancement (cropping, histogram equalization, normalization).
* **Class Weighting**: Handles imbalanced datasets by applying class weights during training.

## Model Architecture
The model is built on top of a **MobileNetV2** base with its final classification layer removed. The custom classifier head consists of:
1.  `base_model` (MobileNetV2, non-trainable initially)
2.  `GlobalAveragePooling2D()` to reduce spatial dimensions.
3.  `Dense(128, activation='relu')` as a fully-connected hidden layer.
4.  `Dropout(0.3)` to prevent overfitting.
5.  `Dense(5, activation='softmax')` as the final output layer for the 5 OA grades.

## Dataset
This model is trained to classify knee X-rays based on the **Kellgren-Lawrence (KL) grading system**, which is the standard for diagnosing Osteoarthritis severity.

![Sample Prediction](https://cdn.apollohospitals.com/health-library-prod/2021/03/Stages-of-Osteoarthritis-1024x512.jpg) **grading**

* **Grade 0**: No signs of OA.
* **Grade 1**: Doubtful joint space narrowing (JSN) and possible osteophytes.
* **Grade 2**: Definite osteophytes and possible JSN.
* **Grade 3**: Multiple osteophytes, definite JSN, and some sclerosis.
* **Grade 4**: Large osteophytes, severe JSN, and definite sclerosis.

To use this project, you need to structure your data in the following format:
```

data/
└── train/
├── Grade 0/
│   ├── image1.png
│   └── image2.png
├── Grade 1/
│   └── ...
├── Grade 2/
│   └── ...
├── Grade 3/
│   └── ...
└── Grade 4/
└── ...

````
A popular public dataset for this task is the **Osteoarthritis Initiative (OAI)** dataset.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/osteoarthritis-detection.git](https://github.com/your-username/osteoarthritis-detection.git)
    cd osteoarthritis-detection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    tensorflow
    numpy
    matplotlib
    seaborn
    scikit-learn
    opencv-python
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model
1.  Place your dataset inside the `data/` directory following the structure mentioned above.
2.  Run the main training script:
    ```bash
    python train_model.py
    ```
    The script will:
    * Load and preprocess the data.
    * Run the two-phase training process.
    * Save the fine-tuned model to `models/mobilenet_finetuned.h5`.
    * Display a classification report and plots for the confusion matrix and training accuracy.

### Running Predictions
To classify a new image, you can use the saved model. Create a script `predict.py` with the following code:

```python
# predict.py
import numpy as np
from tensorflow.keras.models import load_model
from utils.image_utils import preprocess_image
import sys

# Load the trained model
model = load_model("models/mobilenet_finetuned.h5")

# Class labels
class_names = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

def predict_image(image_path):
    """Loads, preprocesses, and predicts the class for a single image."""
    # Preprocess the image using the same utility function from training
    processed_image = preprocess_image(image_path, target_size=(160, 160))
    if processed_image is None:
        print(f"Could not process image: {image_path}")
        return

    # Add a batch dimension
    img_batch = np.expand_dims(processed_image, axis=0)

    # Make prediction
    predictions = model.predict(img_batch)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class_name} (Confidence: {confidence:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        image_file = sys.argv[1]
        predict_image(image_file)

````

Run the prediction script from your terminal:

```bash
python predict.py "path/to/your/test_image.png"
```

## Results

After training, the model's performance is evaluated on the validation set.

**Classification Report:**

```
📊 Classification Report:
              precision    recall  f1-score   support
           0     0.8500    0.9000    0.8743        20
           1     0.7826    0.8182    0.8000        22
           2     0.9048    0.8636    0.8837        22
           3     0.8261    0.8636    0.8444        22
           4     0.9500    0.8636    0.9048        22
    accuracy                         0.8611       108
   macro avg     0.8627    0.8618    0.8614       108
weighted avg     0.8634    0.8611    0.8616       108
```

**Confusion Matrix:**
 **Training vs. Validation Accuracy:**
###  **File Structure**

```
.
├── data/
│   └── train/
│       └── Grade 0/
│           └── ...
├── models/
│   └── mobilenet_finetuned.h5
├── results/
│   ├── confusion_matrix.png
│   └── accuracy_plot.png
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── image_utils.py
│   └── model_utils.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

```
```
