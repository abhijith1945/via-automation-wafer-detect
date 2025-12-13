"""
Data Loader Module for Virtual Metrology System.

Handles fetching and loading the UCI SECOM dataset.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

def fetch_secom_from_uci():
    """
    Fetch SECOM dataset using ucimlrepo package.
    Returns features (X) and labels (y) as DataFrames.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        print("Fetching SECOM dataset from UCI ML Repository...")
        secom = fetch_ucirepo(id=179)  # SECOM dataset ID
        X = secom.data.features
        y = secom.data.targets
        print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    except ImportError:
        print("ucimlrepo not installed. Trying alternative method...")
        return None, None
    except Exception as e:
        print(f"Error fetching from UCI: {e}")
        return None, None


def load_secom_from_files(data_path: str, labels_path: str):
    """
    Load SECOM dataset from local files.
    
    Args:
        data_path: Path to secom.data file
        labels_path: Path to secom_labels.data file
    
    Returns:
        X: Features DataFrame (590 columns)
        y: Labels Series (-1 = Pass, 1 = Fail)
    """
    print(f"Loading SECOM data from local files...")
    
    # Load sensor data (space-separated, no headers)
    X = pd.read_csv(data_path, sep=r"\s+", header=None)
    
    # Load labels (space-separated: label, timestamp)
    labels_df = pd.read_csv(labels_path, sep=r"\s+", header=None)
    y = labels_df.iloc[:, 0]  # First column is the label
    
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: Pass={sum(y == -1)}, Fail={sum(y == 1)}")
    
    return X, y


def generate_synthetic_secom(n_samples: int = 600, n_features: int = 590):
    """
    Generate synthetic data mimicking SECOM dataset structure.
    Use this for testing when real data is unavailable.
    
    Args:
        n_samples: Number of wafers to simulate
        n_features: Number of sensor features
    
    Returns:
        X: Synthetic features DataFrame
        y: Synthetic labels Series
    """
    print(f"Generating synthetic SECOM-like data...")
    np.random.seed(42)
    
    # Create base data with some structure
    X = pd.DataFrame(np.random.rand(n_samples, n_features))
    
    # Add realistic NaN patterns (some columns have many NaNs)
    for col in np.random.choice(n_features, size=int(n_features * 0.1), replace=False):
        nan_mask = np.random.rand(n_samples) < 0.5
        X.iloc[nan_mask, col] = np.nan
    
    # Create imbalanced labels (~7% failures like real SECOM)
    y = pd.Series(np.random.choice([-1, 1], size=n_samples, p=[0.93, 0.07]))
    
    print(f"✓ Synthetic data created: {n_samples} samples, {n_features} features")
    print(f"  Class distribution: Pass={sum(y == -1)}, Fail={sum(y == 1)}")
    
    return X, y


def load_secom_from_csv(csv_path: str):
    """
    Load SECOM dataset from a combined CSV file (Kaggle format).
    
    Args:
        csv_path: Path to uci-secom.csv file
    
    Returns:
        X: Features DataFrame
        y: Labels Series (-1 = Pass, 1 = Fail)
    """
    print(f"Loading SECOM data from CSV: {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # The last column is Pass/Fail, first column is Time
    # Extract features (all columns except Time and Pass/Fail)
    feature_cols = [col for col in df.columns if col not in ['Time', 'Pass/Fail']]
    X = df[feature_cols]
    y = df['Pass/Fail']
    
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: Pass={sum(y == -1)}, Fail={sum(y == 1)}")
    
    return X, y


def load_data(use_synthetic: bool = False):
    """
    Main data loading function. Tries multiple methods in order:
    1. Combined CSV file (Kaggle format)
    2. UCI repository (if ucimlrepo installed)
    3. Local files (if present)
    4. Synthetic data (fallback for demo)
    
    Args:
        use_synthetic: Force use of synthetic data
    
    Returns:
        X: Features DataFrame
        y: Labels Series
    """
    from config import DATA_RAW_DIR, SECOM_DATA_FILE, SECOM_LABELS_FILE
    
    if use_synthetic:
        return generate_synthetic_secom()
    
    # Try combined CSV first (Kaggle format)
    csv_path = os.path.join(DATA_RAW_DIR, "uci-secom.csv")
    if os.path.exists(csv_path):
        return load_secom_from_csv(csv_path)
    
    # Try UCI repo
    X, y = fetch_secom_from_uci()
    if X is not None:
        return X, y
    
    # Try local files (original format)
    if os.path.exists(SECOM_DATA_FILE) and os.path.exists(SECOM_LABELS_FILE):
        return load_secom_from_files(SECOM_DATA_FILE, SECOM_LABELS_FILE)
    
    # Fallback to synthetic
    print("⚠ Real data not available. Using synthetic data for demo.")
    return generate_synthetic_secom()


if __name__ == "__main__":
    # Test the data loader
    X, y = load_data()
    print(f"\nData shape: {X.shape}")
    print(f"Missing values: {X.isna().sum().sum()}")
    print(f"Label distribution:\n{y.value_counts()}")
