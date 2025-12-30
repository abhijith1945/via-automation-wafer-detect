# 🔬 Virtual Metrology System for Semiconductor Manufacturing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Enterprise-grade AI system for real-time wafer yield prediction, visual defect detection, and self-healing process control in semiconductor fabrication.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## 🎯 Overview

Virtual Metrology (VM) replaces slow, destructive physical measurements with fast, AI-powered predictions. This system demonstrates a **multimodal approach** combining:

1. **Sensor Analytics** - Random Forest on 590 process sensors
2. **Visual Inspection** - CNN-based defect classification
3. **Generative AI** - VAE for synthetic defect image generation
4. **Self-Healing Control** - Feed-forward parameter adjustment

### Business Value

| Metric | Physical Metrology | Virtual Metrology |
|--------|-------------------|-------------------|
| Time per wafer | ~30 minutes | ~0.3 seconds |
| Throughput | 2 wafers/hour | 12,000 wafers/hour |
| **Speedup** | - | **6,000x** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VIRTUAL METROLOGY SYSTEM v3.0                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   SENSOR     │    │   VISION     │    │  GENERATIVE  │           │
│  │   LAYER      │    │   LAYER      │    │     AI       │           │
│  │              │    │              │    │              │           │
│  │ Random Forest│───▶│     CNN      │───▶│     VAE      │           │
│  │ + SMOTE      │    │  Classifier  │    │   + NLG      │           │
│  │              │    │              │    │              │           │
│  │ 590 sensors  │    │ 4 defect     │    │ Image Gen    │           │
│  │ 93.3% acc    │    │ types        │    │ Report Gen   │           │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │
│         │                   │                   │                    │
│         └───────────────────┼───────────────────┘                    │
│                             │                                        │
│                             ▼                                        │
│                    ┌──────────────┐                                  │
│                    │ SELF-HEALING │                                  │
│                    │   CONTROL    │                                  │
│                    │              │                                  │
│                    │ Feed-forward │                                  │
│                    │ parameter    │                                  │
│                    │ adjustment   │                                  │
│                    └──────────────┘                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Wafer → 590 Sensors → PASS/FAIL prediction
                           ↓
                      If FAIL
                           ↓
                   Visual Inspection → Defect Type
                           ↓
                   Self-Healing → Parameter Correction
```

---

## ✨ Features

### 🔬 Single Wafer Analysis
- Real-time sensor data visualization
- Pass/Fail prediction with confidence scores
- Visual defect classification (Scratch, Edge Ring, Particle)
- Animated self-healing recommendations

### 📦 Batch Processing
- CSV upload for bulk analysis
- Progress tracking with yield statistics
- Defect distribution visualization

### 🧬 Generative AI Lab
- **VAE Image Generator**: Creates synthetic wafer defect images
- **NLG Report Generator**: AI-written defect analysis reports
- Data augmentation for rare defect types

### 📊 Analytics Dashboard
- SHAP-style feature importance
- Parameter distribution analysis
- Historical trend tracking

### 📈 Model Performance
- ROC and Precision-Recall curves
- Detailed confusion matrix
- Model comparison benchmarks

### 🔄 Auto-Simulation Mode
- Live demo with sensor drift simulation
- Continuous wafer generation for presentations

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/abhijith1945/via-automation-wafer-detect.git
cd via

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train Models (Optional)

```bash
# Train sensor model (Random Forest + SMOTE)
python train.py

# Train vision model (CNN)
python train_vision_real.py

# Train VAE for image generation
python train_vae.py
```

### Run the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 💻 Usage

### Single Wafer Analysis

1. Select **"🔬 Single Wafer"** mode
2. Adjust sensor parameters using sliders:
   - Chamber Pressure (90-110 Pa)
   - Etch Temperature (300-600°C)
   - Gas Flow Rate (40-60 sccm)
   - RF Power (800-1200 W)
3. Click **"🚀 ANALYZE WAFER"**
4. View prediction results and self-healing recommendations

### Batch Processing

1. Select **"📦 Batch Processing"** mode
2. Upload a CSV with columns: `pressure`, `temperature`, `flow_rate`, `rf_power`
3. Or click **"🎲 Generate Demo Batch"** for 20 sample wafers
4. View batch results and yield statistics

### Generative AI

1. Select **"🧬 Generative AI"** mode
2. Choose number of images to generate (1-16)
3. Click **"🎨 Generate Images"** to create synthetic defect images
4. Use **"📄 Generate AI Report"** for natural language analysis

### Auto-Simulation (Demo Mode)

1. In Single Wafer mode, check **"🔄 Auto-Simulate"**
2. Set simulation interval (2-10 seconds)
3. Watch continuous wafer analysis with random sensor drift

---

## 🧠 Model Details

### Sensor Model

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| Training Data | UCI SECOM (1,567 samples) |
| Features | 590 sensors |
| Balancing | SMOTE oversampling |
| Accuracy | 93.3% |
| AUC-ROC | 0.94 |

### Vision Model

| Parameter | Value |
|-----------|-------|
| Architecture | CNN (3 Conv layers) |
| Training Data | NEU Surface Defect Database |
| Classes | 4 (Clean, Scratch, Edge Ring, Particle) |
| Input Size | 128×128 pixels |

> ⚠️ **Note**: Vision model trained on NEU metal surface data as proxy. For production, retrain with actual semiconductor wafer images.

### VAE Generator

| Parameter | Value |
|-----------|-------|
| Architecture | Variational Autoencoder |
| Latent Dimension | 64 |
| Training Epochs | 50 |
| Input/Output | 64×64 RGB images |

---

## 📁 Project Structure

```
via/
├── app.py                  # Main Streamlit dashboard
├── train.py                # Sensor model training
├── train_vision.py         # Vision model training
├── train_vision_real.py    # Vision model (real images)
├── train_vae.py            # VAE training
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing
│   └── llm_reports.py      # NLG report generator
│
├── models/
│   ├── yield_model.pkl     # Trained Random Forest
│   ├── vision_model.h5     # Trained CNN
│   ├── vae_encoder.h5      # VAE encoder
│   └── vae_decoder.h5      # VAE decoder
│
├── data/
│   ├── raw/
│   │   └── uci-secom.csv   # UCI SECOM dataset
│   └── processed/          # Preprocessed data
│
└── assets/
    ├── wafer_images/       # Training images (.npy)
    └── generated_wafers/   # VAE generated images
```

---

## 📊 Performance Benchmarks

| Operation | Time |
|-----------|------|
| Sensor Prediction | < 20ms |
| Visual Analysis | < 300ms |
| VAE Generation | < 100ms |
| Full Pipeline | < 500ms |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** - SECOM Dataset
- **NEU Surface Defect Database** - Vision training images
- **Streamlit** - Dashboard framework
- **TensorFlow/Keras** - Deep learning models

---

<div align="center">

**Built with ❤️ for Semiconductor Manufacturing Excellence**

*Virtual Metrology System v3.0 | Enterprise Edition*

</div>
