import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
DATA_DIR    = r"D:\J\data of the cars"
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 15
MODEL_SAVE  = "saved_model/part_classifier.h5"

# ─────────────────────────────────────────────────────────────────
# 1. Data Generators
# ─────────────────────────────────────────────────────────────────
# IMPORTANT: No horizontal flipping for Left/Right classification!
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 20% validation
    rotation_range=15,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"Classes found: {train_gen.class_indices}")

# ─────────────────────────────────────────────────────────────────
# 2. Model Creation (EfficientNetV2-B0)
# ─────────────────────────────────────────────────────────────────
base_model = EfficientNetV2B0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Freeze initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_all_probs' if train_gen.num_classes == 1 else 'categorical_crossentropy',
    metrics=['accuracy']
)

# ─────────────────────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    MODEL_SAVE,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

print("\n>>> Starting Stage 1 Training (Frozen Base)...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[checkpoint, early_stop]
)

# Fine-tuning: Unfreeze top layers
print("\n>>> Starting Stage 2 Training (Fine-tuning)...")
base_model.trainable = True
# Only unfreeze the top 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print(f"\n✅ Training complete! Best model saved to {MODEL_SAVE}")
