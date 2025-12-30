"""
🧬 IMPROVED VAE - Generates Realistic Wafer Images
Fixed version with better training
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("🧬 IMPROVED VAE - Realistic Wafer Image Generator")
print("=" * 60)

# ============================================================
# GENERATE HIGH-QUALITY TRAINING DATA
# ============================================================

print("\n[1/5] Generating High-Quality Training Images...")

IMAGE_SIZE = 64
LATENT_DIM = 64  # Increased for better quality

def create_wafer_base(size=64):
    """Create realistic circular wafer"""
    img = np.zeros((size, size, 3), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < center - 3:
                # Gradient from center
                intensity = 0.5 + 0.3 * (1 - dist / center)
                img[i, j] = [intensity * 0.9, intensity * 0.95, intensity]
    
    return img

def add_scratch(img, size=64):
    """Add scratch defect"""
    center = size // 2
    angle = np.random.uniform(-0.3, 0.3)
    offset = np.random.randint(-10, 10)
    
    for i in range(10, size - 10):
        j = int(center + offset + (i - center) * angle)
        if 5 < j < size - 5:
            dist_from_center = np.sqrt((i - center)**2 + (j - center)**2)
            if dist_from_center < center - 5:
                for t in range(-1, 2):
                    if 0 <= j + t < size:
                        img[i, j + t] = [0.9, 0.15, 0.15]  # Red scratch
    return img

def add_edge_ring(img, size=64):
    """Add edge ring defect"""
    center = size // 2
    inner_r = np.random.randint(center - 12, center - 6)
    outer_r = inner_r + np.random.randint(3, 6)
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if inner_r < dist < outer_r:
                img[i, j] = [0.95, 0.5, 0.1]  # Orange ring
    return img

def add_particles(img, size=64):
    """Add particle contamination"""
    center = size // 2
    num_particles = np.random.randint(4, 10)
    
    for _ in range(num_particles):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(5, center - 8)
        px = int(center + dist * np.cos(angle))
        py = int(center + dist * np.sin(angle))
        
        radius = np.random.randint(2, 4)
        for i in range(max(0, px - radius), min(size, px + radius)):
            for j in range(max(0, py - radius), min(size, py + radius)):
                if (i - px)**2 + (j - py)**2 < radius**2:
                    img[i, j] = [0.2, 0.3, 0.9]  # Blue particles
    return img

def add_center_defect(img, size=64):
    """Add center defect"""
    center = size // 2
    radius = np.random.randint(8, 15)
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < radius:
                intensity = 0.3 + 0.4 * (dist / radius)
                img[i, j] = [0.9, intensity, intensity]
    return img

def generate_wafer_image(defect_type='random'):
    """Generate a wafer image with specified defect"""
    img = create_wafer_base(IMAGE_SIZE)
    
    if defect_type == 'random':
        defect_type = np.random.choice(['clean', 'scratch', 'edge_ring', 'particle', 'center'])
    
    if defect_type == 'scratch':
        img = add_scratch(img)
    elif defect_type == 'edge_ring':
        img = add_edge_ring(img)
    elif defect_type == 'particle':
        img = add_particles(img)
    elif defect_type == 'center':
        img = add_center_defect(img)
    # 'clean' - no defect added
    
    # Add slight noise for realism
    noise = np.random.randn(IMAGE_SIZE, IMAGE_SIZE, 3) * 0.02
    img = np.clip(img + noise, 0, 1)
    
    return img.astype(np.float32)

# Generate training data
X_train = []
defect_types = ['clean', 'scratch', 'edge_ring', 'particle', 'center']

for defect in defect_types:
    print(f"   Generating 200 '{defect}' images...")
    for _ in range(200):
        X_train.append(generate_wafer_image(defect))

X_train = np.array(X_train)
np.random.shuffle(X_train)

print(f"   ✅ Generated {len(X_train)} training images")

# ============================================================
# BUILD IMPROVED VAE
# ============================================================

print("\n[2/5] Building Improved VAE Architecture...")

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ENCODER
encoder_inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)

z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# DECODER
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(4 * 4 * 256, activation="relu")(latent_inputs)
x = layers.Reshape((4, 4, 256))(x)
x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

print("   ✅ Encoder: 64x64x3 → 64D latent space")
print("   ✅ Decoder: 64D → 64x64x3")

# VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction),
                    axis=(1, 2, 3)
                )
            )
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            total_loss = reconstruction_loss + 0.001 * kl_loss  # Reduced KL weight
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        
        return {"loss": self.total_loss_tracker.result()}

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))

# ============================================================
# TRAIN VAE
# ============================================================

print("\n[3/5] Training VAE (This takes ~2-3 minutes)...")

history = vae.fit(
    X_train, 
    epochs=50,  # More epochs for better quality
    batch_size=32, 
    verbose=1
)

# ============================================================
# GENERATE AND VERIFY IMAGES
# ============================================================

print("\n[4/5] Generating New Wafer Images...")

def generate_new_images(decoder, n_images=16):
    """Generate new images from random latent vectors"""
    random_latent = np.random.normal(0, 1, size=(n_images, LATENT_DIM))
    generated = decoder.predict(random_latent, verbose=0)
    return generated

generated_images = generate_new_images(decoder, 16)

print(f"   ✅ Generated {len(generated_images)} new images")
print(f"   Image stats: min={generated_images.min():.3f}, max={generated_images.max():.3f}")

# ============================================================
# SAVE EVERYTHING
# ============================================================

print("\n[5/5] Saving Models and Images...")

os.makedirs('models', exist_ok=True)
os.makedirs('assets/generated_wafers', exist_ok=True)
os.makedirs('assets/wafer_images', exist_ok=True)

# Save models
encoder.save('models/vae_encoder.h5')
decoder.save('models/vae_decoder.h5')

# Save latent dim for loading later
with open('models/vae_config.txt', 'w') as f:
    f.write(f"LATENT_DIM={LATENT_DIM}")

# Save generated images
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i])
    ax.set_title(f'Generated #{i+1}', fontsize=10)
    ax.axis('off')
    plt.imsave(f'assets/generated_wafers/gen_wafer_{i+1}.png', generated_images[i])

plt.suptitle('VAE Generated Wafer Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/generated_wafers/all_generated.png', dpi=150)
plt.close()

# Save comparison: Training vs Generated
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(X_train[i * 100])
    axes[0, i].set_title('Training', fontsize=9)
    axes[0, i].axis('off')
    
    axes[1, i].imshow(generated_images[i])
    axes[1, i].set_title('Generated', fontsize=9)
    axes[1, i].axis('off')

plt.suptitle('Training Data vs VAE Generated', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/generated_wafers/comparison.png', dpi=150)
plt.close()

# Save sample training images for reference (these are the GOOD ones)
print("   Saving sample defect images...")
for defect in defect_types:
    img = generate_wafer_image(defect)
    plt.imsave(f'assets/wafer_images/{defect}.png', img)

print("   ✅ Models saved to models/")
print("   ✅ Generated images saved to assets/generated_wafers/")
print("   ✅ Sample defect images saved to assets/wafer_images/")

print("\n" + "=" * 60)
print("✅ VAE TRAINING COMPLETE!")
print("=" * 60)
print(f"""
📊 Results:
   • Training Images: {len(X_train)}
   • Latent Dimension: {LATENT_DIM}
   • Epochs: 50
   • Generated Samples: 16

📁 Files Created:
   • models/vae_encoder.h5
   • models/vae_decoder.h5
   • assets/generated_wafers/*.png
   • assets/wafer_images/*.png (sample defects)
""")
