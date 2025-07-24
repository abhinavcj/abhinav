# train_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils.data_loader import load_images_and_labels
from utils.model_utils import build_mobilenet_model

# STEP 1: Load Data
print("🔄 Loading and preprocessing images...")
X, y = load_images_and_labels("data/train", target_size=(160, 160))

# STEP 2: Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# STEP 3: Class Weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))
print("Class Weights:", class_weights)

# STEP 4: Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# STEP 5: Model
model, base_model = build_mobilenet_model(input_shape=(160, 160, 3), num_classes=5)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# STEP 6: Train Top Layers
print("\n🔵 Phase 1: Training top layers...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=5, class_weight=class_weights, callbacks=[early_stop, reduce_lr], batch_size=32)

# STEP 7: Fine-tuning
print("\n🟢 Phase 2: Fine-tuning base model...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                             epochs=10, class_weight=class_weights,
                             callbacks=[early_stop, reduce_lr], batch_size=32)

# STEP 8: Save
os.makedirs("models", exist_ok=True)
model.save("models/mobilenet_finetuned.h5")
print("✅ Model saved!")

# STEP 9: Evaluation
y_pred = np.argmax(model.predict(X_val), axis=1)
print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred, digits=4))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# STEP 10: Accuracy Plot
train_acc = history.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
