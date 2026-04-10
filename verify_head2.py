import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATA_DIR = r"D:\J\data of the cars"
MODEL_PATH = r"d:\Projects\Car-Insurance-Fraud-Detection-main\saved_model\part_classifier.h5"

# Setup generator just to see class indices
datagen = ImageDataGenerator(rescale=1./255)
gen = datagen.flow_from_directory(DATA_DIR, target_size=(224, 224), batch_size=1)

print("\n--- Model Class Indices ---")
print(gen.class_indices)

# Try loading the model to ensure it works
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found at {MODEL_PATH}")
