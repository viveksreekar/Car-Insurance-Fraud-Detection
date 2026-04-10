import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_PATH = 'saved_model/fraud_detector.h5'
TEST_DIR = 'data/testing'
IMG_SIZE = 224
BATCH_SIZE = 32 # Use the same batch size as in training/testing
CLASS_NAMES = ['fraud', 'non_fraud'] # Should match the folder names alphabetically

# --- 1. Load the Model and Test Data ---
print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run train_model.py successfully first.")
    exit()

print(f"Loading test data from: {TEST_DIR}")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False  # IMPORTANT: Do not shuffle test data for evaluation
)

# --- 2. Make Predictions on the Entire Test Set ---
print("Making predictions on the test dataset...")

# We need to iterate through the dataset to get all predictions and true labels
true_labels = []
predicted_labels = []

for images, labels in test_dataset:
    predictions = model.predict(images)
    # Convert probabilities to binary class labels (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int).flatten()
    
    true_labels.extend(labels.numpy().flatten().astype(int))
    predicted_labels.extend(binary_predictions)

# Convert lists to numpy arrays for scikit-learn
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# --- 3. Generate and Print the Classification Report ---
print("\n" + "="*50)
print("           CLASSIFICATION REPORT")
print("="*50)

# Generate the report from scikit-learn
report = classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES)
print(report)


# --- 4. Generate and Visualize the Confusion Matrix ---
print("\n" + "="*50)
print("             CONFUSION MATRIX")
print("="*50)

# Generate the matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot the matrix using Seaborn for a nice visual
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()