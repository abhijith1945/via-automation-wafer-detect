# 🎛️ Virtual Metrology System

Real-time wafer yield prediction and process control for semiconductor manufacturing.

## 🎯 Overview

Virtual Metrology uses machine learning to predict wafer quality from process sensor data, replacing expensive and time-consuming physical measurements.

**Problem:** Physical metrology takes 30+ minutes per wafer
**Solution:** Instant prediction using existing sensor data

## 🏗️ Architecture

```
Raw Sensor Data → Cleaning → SMOTE Balancing → Random Forest → Streamlit Dashboard
     (590 sensors)   (NaN handling)   (Class balance)    (Prediction)     (Visualization)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# With real SECOM data (download from Kaggle first)
python train.py

# Or with synthetic data (for testing)
python train.py --synthetic
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

## 📁 Project Structure

```
via/
├── app.py                 # Streamlit dashboard
├── train.py               # Model training script
├── config.py              # Configuration constants
├── requirements.txt       # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # SECOM data fetching
│   └── preprocessing.py   # Data cleaning pipeline
│
├── data/
│   ├── raw/               # Original SECOM files
│   └── processed/         # Cleaned data
│
└── models/                # Trained model artifacts
    ├── yield_model.pkl
    ├── imputer.pkl
    ├── selector.pkl
    └── scaler.pkl
```

## 📊 Dataset

**UCI SECOM Dataset**
- 1,567 wafer samples
- 590 sensor features
- Highly imbalanced: only 6.6% failures
- Many missing values (NaN)

Download: [UCI SECOM on Kaggle](https://www.kaggle.com/datasets/paresh2047/uci-semcom)

Place files in `data/raw/`:
- `secom.data`
- `secom_labels.data`

## 🔧 Key Features

### Data Preprocessing
- Removes columns with >40% missing values
- Median imputation for remaining NaNs
- Variance threshold feature selection
- StandardScaler normalization

### Imbalanced Learning
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- Only applied to training data
- Balances 6.6% failure rate

### Model
- Random Forest Classifier
- 100 estimators, balanced class weights
- Evaluated on: Balanced Accuracy, F1-Score, ROC-AUC

### Dashboard
- Real-time sensor monitoring
- Pass/Fail prediction with confidence
- Process stability visualization
- Feature importance display

## 🎤 Demo Script

1. **Open Dashboard**: `streamlit run app.py`

2. **Normal Operation**:
   - Set Temperature to 450°C
   - Show green "WAFER PASS"
   - Say: *"All sensors nominal, wafer passes quality check"*

3. **Drift Detection**:
   - Slide Temperature to 580°C
   - Show red "WAFER FAIL"
   - Say: *"We caught this defect in 0.05 seconds. Physical measurement would take 30 minutes."*

## 📈 Innovation Points

1. **SMOTE for Class Imbalance**: Handles 6.6% defect rate without naive oversampling
2. **Robust Preprocessing**: Handles 590 messy sensor columns automatically
3. **Industrial Dashboard**: Control room style interface, not a toy demo

## 🏆 For Judges

**ROI Story:**
- Physical metrology: 30 min/wafer × $X/hour = expensive
- Virtual metrology: 0.05 sec/wafer = essentially free
- Early defect detection = reduced waste = millions saved

**Technical Merit:**
- Handles real industrial data challenges (NaN, imbalance)
- Uses proper ML practices (train/test split, stratified sampling)
- Production-ready architecture (saved models, preprocessors)

## 📝 License

MIT License - Built for educational/hackathon purposes.

---

Built with ❤️ using Python, Streamlit, scikit-learn, and imbalanced-learn
