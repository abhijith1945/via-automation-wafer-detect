"""
🔬 Vision Model Training - REAL NEU Surface Defect Dataset
Uses 1000 images subset for fast training (~5 mins on CPU)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import random

print("=" * 60)
print("🔬 VISION MODEL - REAL DEFECT IMAGES")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\abhij\Downloads\archive (2)\NEU-DET\train\images"
IMG_SIZE = 64  # Resize to 64x64 for fast training
SAMPLES_PER_CLASS = 166  # ~1000 total images (166 x 6 classes)
EPOCHS = 12
BATCH_SIZE = 32

# Map NEU classes to our 4 classes for hackathon demo
# NEU has: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
CLASS_MAPPING = {
    'scratches': 0,       # Scratch
    'crazing': 1,         # Edge Ring (similar pattern)
    'inclusion': 2,       # Particle (similar to contamination)
    'patches': 2,         # Particle
    'pitted_surface': 2,  # Particle
    'rolled-in_scale': 1  # Edge Ring
}

CLASS_NAMES = ['Scratch', 'Edge Ring', 'Particle']

print(f"\n📂 Dataset: {DATASET_PATH}")
print(f"📊 Target: {SAMPLES_PER_CLASS * 6} images → 3 classes")
print(f"🖼️ Image Size: {IMG_SIZE}x{IMG_SIZE}")

# ============================================================
# LOAD REAL IMAGES
# ============================================================

print("\n[1/4] Loading Real Defect Images...")

X_data = []
y_data = []

dataset_path = Path(DATASET_PATH)

for class_folder in dataset_path.iterdir():
    if class_folder.is_dir():
        class_name = class_folder.name
        if class_name not in CLASS_MAPPING:
            print(f"   ⚠️ Skipping unknown class: {class_name}")
            continue
            
        label = CLASS_MAPPING[class_name]
        
        # Get all images in this class
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.bmp"))
        
        # Random sample
        sample_size = min(SAMPLES_PER_CLASS, len(images))
        selected = random.sample(images, sample_size)
        
        print(f"   Loading {sample_size} images from '{class_name}' → Class {label} ({CLASS_NAMES[label]})")
        
        for img_path in selected:
            try:
                # Load and preprocess image
                img = tf.io.read_file(str(img_path))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize to [0, 1]
                
                X_data.append(img.numpy())
                y_data.append(label)
            except Exception as e:
                pass  # Skip corrupted images

X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data)

print(f"\n   ✅ Loaded {len(X_data)} images")
print(f"   Classes: {CLASS_NAMES}")
print(f"   Class distribution: {np.bincount(y_data)}")

# Shuffle data
indices = np.random.permutation(len(X_data))
X_data = X_data[indices]
y_data = y_data[indices]

# Split train/val
split_idx = int(0.8 * len(X_data))
X_train, X_val = X_data[:split_idx], X_data[split_idx:]
y_train, y_val = y_data[:split_idx], y_data[split_idx:]

print(f"   Train: {len(X_train)} | Validation: {len(X_val)}")

# ============================================================
# BUILD CNN MODEL
# ============================================================

print("\n[2/4] Building CNN Model...")

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Conv Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Dense
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"   ✅ Model built: {model.count_params():,} parameters")

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n[3/4] Training Model...")
print(f"   This will take ~5-8 minutes on CPU...")
print()

# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
best_val_acc = max(history.history['val_accuracy'])

print(f"\n   ✅ Training Complete!")
print(f"   Final Training Accuracy: {final_acc:.1%}")
print(f"   Best Validation Accuracy: {best_val_acc:.1%}")

# ============================================================
# SAVE MODEL
# ============================================================

print("\n[4/4] Saving Model...")

os.makedirs('models', exist_ok=True)
model.save('models/vision_model.h5')
print("   ✅ Model saved: models/vision_model.h5")

# Save sample images for dashboard
os.makedirs('assets/wafer_images', exist_ok=True)

# Save a few sample images
for i, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y_data == i)[0]
    if len(class_indices) > 0:
        sample_img = X_data[class_indices[0]]
        
        # Save using TensorFlow (no matplotlib needed)
        img_uint8 = tf.cast(sample_img * 255, tf.uint8)
        encoded = tf.io.encode_png(img_uint8)
        tf.io.write_file(f'assets/wafer_images/{class_name.lower().replace(" ", "_")}.png', encoded)
        print(f"   ✅ Saved sample: {class_name.lower().replace(' ', '_')}.png")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("✅ REAL DEFECT MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"""
📊 Training Summary:
   • Dataset: NEU Surface Defect (REAL images)
   • Images Used: {len(X_data)}
   • Classes: {', '.join(CLASS_NAMES)}
   • Best Validation Accuracy: {best_val_acc:.1%}

📁 Files Created:
   • models/vision_model.h5
   • assets/wafer_images/*.png

🚀 Next: Run 'streamlit run app.py' to see the dashboard!
""")
