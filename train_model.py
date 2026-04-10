import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
TRAIN_DIR = 'data/training'
TEST_DIR = 'data/testing'
MODEL_SAVE_PATH = 'saved_model/fraud_detector.h5'

# --- 1. Setup Directories & Data Pipelines ---
print(">>> Step 1: Setting up data pipelines...")
os.makedirs('saved_model', exist_ok=True)

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
print(f"Found classes: {class_names} (Order is important! 0={class_names[0]}, 1={class_names[1]})")

# --- 2. Handle Class Imbalance ---
print(">>> Step 2: Calculating class weights to handle data imbalance...")
train_labels = np.concatenate([y for x, y in train_dataset], axis=0)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels.flatten()
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Calculated Class Weights: {class_weight_dict}")

# --- 3. Data Augmentation & Performance Optimization ---
print(">>> Step 3: Setting up data augmentation and performance optimization...")
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 4. Model Building with Transfer Learning ---
print(">>> Step 4: Building the model...")
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)
base_model.trainable = False  # Freeze the base model

x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
x = Dropout(0.3, name="dropout_out")(x)
outputs = Dense(1, activation="sigmoid", name="output")(x)
model = Model(inputs, outputs)

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=METRICS)

model.summary()

# --- 5. Training the Model ---
print("\n>>> Step 5: Starting model training...")

# Save the model that has the best recall on the validation set
checkpoint_cb = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_recall', mode='max')
early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight_dict
)

print(f"\n>>> Training complete. Best model saved to {MODEL_SAVE_PATH}")

# --- 6. Visualize Training History ---
print(">>> Step 6: Visualizing training history...")

def plot_history(history):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Recall')
    plt.legend()
    plt.grid(True)

    plt.suptitle("Model Training History")
    plt.show()

plot_history(history)