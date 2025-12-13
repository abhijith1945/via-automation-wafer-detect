"""
Vision Model for Wafer Defect Classification
Trains a lightweight CNN to classify wafer defects:
- Class 0: Clean (No defect)
- Class 1: Scratch
- Class 2: Edge Ring
- Class 3: Particle
"""

import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("🔬 WAFER VISION MODEL TRAINER")
print("=" * 50)

# Check for TensorFlow
print("\n[0/5] Checking TensorFlow installation...")
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow.keras import layers, models
    print(f"   ✅ TensorFlow {tf.__version__} found")
except ImportError:
    print("   ❌ TensorFlow not found. Installing...")
    os.system('pip install tensorflow')
    import tensorflow as tf
    from tensorflow.keras import layers, models

# 1. DEFINE A TINY CNN (Optimized for Hackathons)
print("\n[1/5] Building CNN Architecture...")

model = models.Sequential([
    # First Conv Block
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Conv Block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Conv Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Dense Layers
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 Classes: Clean, Scratch, Ring, Particle
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("   ✅ Model built successfully")
print(f"   Total parameters: {model.count_params():,}")

# 2. CREATE SYNTHETIC TRAINING DATA
print("\n[2/5] Generating Synthetic Training Data...")

def generate_clean_wafer(size=64):
    """Generate a clean wafer image (uniform gray with slight noise)"""
    img = np.ones((size, size, 3)) * 0.7  # Gray background
    img += np.random.randn(size, size, 3) * 0.05  # Slight noise
    return np.clip(img, 0, 1)

def generate_scratch_wafer(size=64):
    """Generate a wafer with scratch defect (diagonal lines)"""
    img = generate_clean_wafer(size)
    # Add diagonal scratch
    for i in range(size):
        if 0 <= i < size:
            thickness = np.random.randint(1, 4)
            for t in range(thickness):
                if i + t < size:
                    img[i, i + t] = [0.9, 0.2, 0.2]  # Red scratch
                j_pos = i - t + 10
                if i - t >= 0 and 0 <= j_pos < size:
                    img[i, j_pos] = [0.9, 0.2, 0.2]
    return np.clip(img, 0, 1)

def generate_ring_wafer(size=64):
    """Generate a wafer with edge ring defect"""
    img = generate_clean_wafer(size)
    center = size // 2
    # Add ring pattern
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if 20 < dist < 26:  # Ring radius
                img[i, j] = [0.9, 0.3, 0.1]  # Orange ring
    return np.clip(img, 0, 1)

def generate_particle_wafer(size=64):
    """Generate a wafer with particle contamination"""
    img = generate_clean_wafer(size)
    # Add random particles
    num_particles = np.random.randint(5, 12)
    for _ in range(num_particles):
        x, y = np.random.randint(8, size-8, 2)
        radius = np.random.randint(2, 5)
        for i in range(max(0, x-radius), min(size, x+radius)):
            for j in range(max(0, y-radius), min(size, y+radius)):
                if (i-x)**2 + (j-y)**2 < radius**2:
                    img[i, j] = [0.2, 0.2, 0.9]  # Blue particles
    return np.clip(img, 0, 1)

# Generate training data
n_samples_per_class = 250
X_train = []
y_train = []

print("   Generating Clean wafers...", end=" ")
for _ in range(n_samples_per_class):
    X_train.append(generate_clean_wafer())
    y_train.append(0)
print("✓")

print("   Generating Scratch defects...", end=" ")
for _ in range(n_samples_per_class):
    X_train.append(generate_scratch_wafer())
    y_train.append(1)
print("✓")

print("   Generating Edge Ring defects...", end=" ")
for _ in range(n_samples_per_class):
    X_train.append(generate_ring_wafer())
    y_train.append(2)
print("✓")

print("   Generating Particle defects...", end=" ")
for _ in range(n_samples_per_class):
    X_train.append(generate_particle_wafer())
    y_train.append(3)
print("✓")

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train)

# Shuffle
indices = np.random.permutation(len(X_train))
X_train = X_train[indices]
y_train = y_train[indices]

print(f"\n   ✅ Generated {len(X_train)} training images")
print(f"   Shape: {X_train.shape}")

# 3. TRAIN THE MODEL
print("\n[3/5] Training Vision Model...")
print("   This may take 1-2 minutes...\n")

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\n   ✅ Training complete!")
print(f"   Final Accuracy: {final_acc:.1%}")
print(f"   Validation Accuracy: {final_val_acc:.1%}")

# 4. SAVE THE MODEL
print("\n[4/5] Saving Model...")
os.makedirs('models', exist_ok=True)
model.save('models/vision_model.h5')
print("   ✅ Vision Model Saved: models/vision_model.h5")

# 5. SAVE SAMPLE IMAGES
print("\n[5/5] Saving Sample Defect Images...")

os.makedirs('assets/wafer_images', exist_ok=True)

# Save individual samples
samples = [
    (generate_clean_wafer(), "clean.png", "Clean"),
    (generate_scratch_wafer(), "scratch.png", "Scratch"),
    (generate_ring_wafer(), "ring.png", "Edge Ring"),
    (generate_particle_wafer(), "particle.png", "Particle")
]

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    for img, filename, label in samples:
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'assets/wafer_images/{filename}', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: assets/wafer_images/{filename}")
    
    # Save combined image
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (img, filename, label) in zip(axes, samples):
        ax.imshow(img)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('assets/wafer_images/all_defects.png', dpi=150)
    plt.close()
    print("   ✅ Saved: assets/wafer_images/all_defects.png")
    
except Exception as e:
    print(f"   ⚠️ Could not save images with matplotlib: {e}")
    # Save as numpy arrays instead
    for img, filename, label in samples:
        np.save(f'assets/wafer_images/{filename.replace(".png", ".npy")}', img)
        print(f"   ✅ Saved: assets/wafer_images/{filename.replace('.png', '.npy')}")

print("\n" + "=" * 50)
print("✅ VISION MODEL TRAINING COMPLETE!")
print("=" * 50)
print(f"""
📊 Model Performance:
   - Training Accuracy: {final_acc:.1%}
   - Validation Accuracy: {final_val_acc:.1%}
   - Classes: Clean, Scratch, Edge Ring, Particle

📁 Saved Files:
   - models/vision_model.h5
   - assets/wafer_images/*.png

🚀 Next Step: Run 'streamlit run app.py' to see the upgraded dashboard!
""")
