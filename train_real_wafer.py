"""
Train Vision CNN and VAE on Real WM-811K Wafer Dataset
Extracts 2000 wafer images from LSWMD.pkl and trains both models
"""

import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Paths
DATASET_PATH = r"C:\Users\abhij\Downloads\archive (1)\LSWMD.pkl"
OUTPUT_DIR = "assets/real_wafers"
MODEL_DIR = "models"
NUM_SAMPLES = 2000
IMG_SIZE = 64  # For VAE
CNN_IMG_SIZE = 128  # For CNN

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("🔬 REAL WAFER DATASET TRAINING")
print("=" * 60)

# ============================================================
# STEP 1: Load and Extract Wafer Data
# ============================================================
print("\n📂 Loading WM-811K dataset (this may take a minute)...")

# Fix for older pandas pickle format
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

with open(DATASET_PATH, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

df = pd.DataFrame(data)
print(f"✅ Loaded {len(df)} wafer records")
print(f"   Columns: {list(df.columns)}")

# Check for failureType column
if 'failureType' in df.columns:
    print(f"\n📊 Failure Type Distribution:")
    print(df['failureType'].value_counts())
    
    # Filter for labeled samples only (not unlabeled)
    # The dataset has labeled failure types as lists/arrays
    df_labeled = df[df['failureType'].apply(lambda x: len(x) > 0 if hasattr(x, '__len__') else False)]
    print(f"\n✅ Found {len(df_labeled)} labeled samples")
else:
    df_labeled = df

# ============================================================
# STEP 2: Extract Wafer Map Images
# ============================================================
print(f"\n🎨 Extracting {NUM_SAMPLES} wafer images...")

def wafer_to_image(wafer_map, size=64):
    """Convert wafer map array to RGB image"""
    if wafer_map is None or not hasattr(wafer_map, 'shape'):
        return None
    
    # Wafer maps are typically 2D arrays with values:
    # 0 = background/no die
    # 1 = good die (pass)
    # 2 = defective die (fail)
    
    # Normalize to 0-1
    wafer_map = np.array(wafer_map)
    
    # Create RGB image
    h, w = wafer_map.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color mapping for wafer visualization
    # Background (0) = dark gray
    # Pass (1) = green
    # Fail (2) = red
    img[wafer_map == 0] = [40, 40, 40]      # Dark gray background
    img[wafer_map == 1] = [0, 180, 0]       # Green for pass
    img[wafer_map == 2] = [220, 50, 50]     # Red for fail
    
    # Resize to target size
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((size, size), Image.Resampling.LANCZOS)
    
    return np.array(pil_img)

# Get failure type labels
def get_failure_label(failure_type):
    """Extract failure label from the dataset format"""
    if failure_type is None:
        return "none"
    if hasattr(failure_type, '__len__') and len(failure_type) > 0:
        # The dataset stores failure types as encoded arrays
        # Common types: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none
        label_map = {
            0: "none",
            1: "Center", 
            2: "Donut",
            3: "Edge-Loc",
            4: "Edge-Ring",
            5: "Loc",
            6: "Random",
            7: "Scratch",
            8: "Near-full"
        }
        idx = np.argmax(failure_type) if len(failure_type) > 1 else int(failure_type[0])
        return label_map.get(idx, "none")
    return "none"

# Sample data
if len(df_labeled) >= NUM_SAMPLES:
    # Stratified sampling if possible
    sampled_df = df_labeled.sample(n=NUM_SAMPLES, random_state=42)
else:
    # Use all labeled + some unlabeled
    sampled_df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)

print(f"   Processing {len(sampled_df)} wafer maps...")

images = []
labels = []
valid_count = 0

for idx, row in sampled_df.iterrows():
    wafer_map = row.get('waferMap', None)
    failure_type = row.get('failureType', None)
    
    if wafer_map is not None and hasattr(wafer_map, 'shape') and wafer_map.size > 0:
        # Create image
        img = wafer_to_image(wafer_map, size=IMG_SIZE)
        if img is not None:
            images.append(img)
            labels.append(get_failure_label(failure_type))
            valid_count += 1
            
            if valid_count % 500 == 0:
                print(f"   Processed {valid_count} wafers...")
    
    if valid_count >= NUM_SAMPLES:
        break

images = np.array(images)
labels = np.array(labels)

print(f"\n✅ Extracted {len(images)} wafer images")
print(f"   Image shape: {images[0].shape}")
print(f"\n📊 Label Distribution:")
unique, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"   {label}: {count}")

# Save sample images
print("\n💾 Saving sample images...")
for i in range(min(10, len(images))):
    img = Image.fromarray(images[i])
    img.save(os.path.join(OUTPUT_DIR, f"wafer_{i}_{labels[i]}.png"))
print(f"   Saved 10 samples to {OUTPUT_DIR}/")

# ============================================================
# STEP 3: Train Vision CNN
# ============================================================
print("\n" + "=" * 60)
print("🧠 TRAINING VISION CNN ON REAL WAFERS")
print("=" * 60)

# Prepare data for CNN
# Resize images to CNN size
cnn_images = []
for img in images:
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((CNN_IMG_SIZE, CNN_IMG_SIZE), Image.Resampling.LANCZOS)
    cnn_images.append(np.array(pil_img))
cnn_images = np.array(cnn_images)

# Normalize
X_cnn = cnn_images.astype('float32') / 255.0

# Encode labels
# Map to 4 main categories for our app
def map_to_app_labels(label):
    """Map WM-811K labels to our app's 4 categories"""
    if label in ['none', 'Random']:
        return 'Clean'
    elif label in ['Scratch']:
        return 'Scratch'
    elif label in ['Edge-Ring', 'Edge-Loc', 'Donut']:
        return 'Edge Ring'
    elif label in ['Center', 'Loc', 'Near-full']:
        return 'Particle'
    else:
        return 'Clean'

app_labels = np.array([map_to_app_labels(l) for l in labels])
print(f"\n📊 Mapped Label Distribution:")
unique, counts = np.unique(app_labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"   {label}: {count}")

le = LabelEncoder()
y_encoded = le.fit_transform(app_labels)
y_cnn = keras.utils.to_categorical(y_encoded)

print(f"\n   Classes: {le.classes_}")
print(f"   X shape: {X_cnn.shape}")
print(f"   y shape: {y_cnn.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n   Train: {len(X_train)}, Test: {len(X_test)}")

# Build CNN model
print("\n🔨 Building CNN model...")

cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(CNN_IMG_SIZE, CNN_IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

# Train
print("\n🚀 Training CNN...")
history = cnn_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
loss, acc = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ CNN Test Accuracy: {acc:.1%}")

# Save CNN
cnn_model.save(os.path.join(MODEL_DIR, 'vision_model.h5'))
print(f"💾 Saved CNN to {MODEL_DIR}/vision_model.h5")

# Save label encoder classes
np.save(os.path.join(MODEL_DIR, 'vision_classes.npy'), le.classes_)

# ============================================================
# STEP 4: Train VAE
# ============================================================
print("\n" + "=" * 60)
print("🧬 TRAINING VAE ON REAL WAFERS")
print("=" * 60)

# Prepare data for VAE
X_vae = images.astype('float32') / 255.0
print(f"\n   VAE input shape: {X_vae.shape}")

# VAE Architecture
latent_dim = 64

# Encoder
encoder_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)

z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(4 * 4 * 256, activation='relu')(latent_inputs)
x = layers.Reshape((4, 4, 256))(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x)

decoder = keras.Model(latent_inputs, x, name='decoder')

# VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=[1, 2, 3])
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))

print("\n🚀 Training VAE...")
vae.fit(X_vae, epochs=50, batch_size=32, verbose=1)

# Save VAE components
encoder.save(os.path.join(MODEL_DIR, 'vae_encoder.h5'))
decoder.save(os.path.join(MODEL_DIR, 'vae_decoder.h5'))
print(f"\n💾 Saved VAE to {MODEL_DIR}/")

# Generate sample images
print("\n🎨 Generating sample wafer images...")
os.makedirs('assets/generated_wafers', exist_ok=True)

for i in range(10):
    z_sample = np.random.normal(0, 1, (1, latent_dim))
    generated = decoder.predict(z_sample, verbose=0)[0]
    generated = (generated * 255).astype(np.uint8)
    img = Image.fromarray(generated)
    img.save(f'assets/generated_wafers/vae_real_{i}.png')

print(f"   Saved 10 generated samples to assets/generated_wafers/")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"""
📊 Summary:
   - Extracted: {len(images)} real wafer images from WM-811K
   - CNN Accuracy: {acc:.1%}
   - VAE Trained: {len(X_vae)} images, 50 epochs
   
💾 Files Created:
   - models/vision_model.h5 (CNN for defect classification)
   - models/vae_encoder.h5 (VAE encoder)
   - models/vae_decoder.h5 (VAE decoder)
   - assets/real_wafers/ (sample images)
   - assets/generated_wafers/ (VAE generated images)

🚀 Now run: streamlit run app.py
""")
