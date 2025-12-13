"""
Configuration constants for the Virtual Metrology System.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data files
SECOM_DATA_FILE = os.path.join(DATA_RAW_DIR, "secom.data")
SECOM_LABELS_FILE = os.path.join(DATA_RAW_DIR, "secom_labels.data")
CLEAN_DATA_FILE = os.path.join(DATA_PROCESSED_DIR, "clean_data.csv")

# Model files
MODEL_FILE = os.path.join(MODELS_DIR, "yield_model.pkl")
IMPUTER_FILE = os.path.join(MODELS_DIR, "imputer.pkl")
SELECTOR_FILE = os.path.join(MODELS_DIR, "selector.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_FILE = os.path.join(MODELS_DIR, "feature_names.pkl")

# Preprocessing parameters
NAN_THRESHOLD = 0.4  # Drop columns with more than 40% missing data
VARIANCE_THRESHOLD = 0.01  # Minimum variance for feature selection

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# Labels mapping (SECOM uses -1 for Pass, 1 for Fail)
LABEL_PASS = -1
LABEL_FAIL = 1

# Dashboard sensor parameters (for demo/visualization)
SENSOR_RANGES = {
    "pressure": {"min": 90, "max": 110, "default": 100, "unit": "Pa"},
    "temperature": {"min": 300, "max": 600, "default": 450, "unit": "°C"},
    "flow_rate": {"min": 40, "max": 60, "default": 50, "unit": "sccm"},
}

# Alert thresholds (for demo)
ALERT_TEMP_HIGH = 550
ALERT_PRESSURE_LOW = 92
