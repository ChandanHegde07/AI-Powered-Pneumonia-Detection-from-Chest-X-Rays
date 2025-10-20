# Simple Deep Learning Model for Chest X-ray Classification

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20

base_dir = 'data/chest_xray/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

print("="*70)
print("SIMPLE CHEST X-RAY DEEP LEARNING MODEL")
print("="*70)

print("\n[STEP 1] Loading data...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"✓ Training dataset loaded")
print(f"✓ Validation dataset loaded")

print("\n[STEP 2] Normalizing data...")

normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

print("✓ Data normalized (pixel values: 0-1)")

print("\n[STEP 3] Building simple CNN model...")

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(1, activation='sigmoid')
])

print("✓ Model architecture created")
print(f"✓ Total parameters: {model.count_params():,}")

print("\nModel Summary:")
model.summary()

print("\n[STEP 4] Compiling model...")

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled")
print("  - Optimizer: Adam")
print("  - Loss: Binary Crossentropy")
print("  - Metrics: Accuracy")

print("\n[STEP 5] Setting up callbacks...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("✓ Early stopping configured (patience=5)")

print("\n[STEP 6] Training model...")
print("-" * 70)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping],
    verbose=1
)

print("-" * 70)
print("✓ Training completed!")

print("\n[STEP 7] Evaluating model...")

val_loss, val_accuracy = model.evaluate(validation_dataset, verbose=0)
print(f"✓ Validation Loss: {val_loss:.4f}")
print(f"✓ Validation Accuracy: {val_accuracy:.4f}")

print("\n[STEP 8] Saving model...")

model_path = './simple_model.h5'
model.save(model_path)
print(f"✓ Model saved to: {model_path}")

print("\n[STEP 9] Creating visualizations...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./simple_model_history.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: simple_model_history.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Model trained for {len(acc)} epochs")
print(f"Best validation accuracy: {max(val_acc):.4f}")
print(f"Final validation accuracy: {val_accuracy:.4f}")
print(f"Model parameters: {model.count_params():,}")
print("\nFiles saved:")
print(f"  • {model_path}")
print(f"  • simple_model_history.png")
print("="*70)