"""
Preprocessing Module for Virtual Metrology System.

Handles data cleaning, imputation, and feature selection for SECOM data.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import joblib


def remove_high_nan_columns(X: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """
    Remove columns with too many missing values.
    
    Args:
        X: Input features DataFrame
        threshold: Maximum allowed proportion of NaN values (default 40%)
    
    Returns:
        DataFrame with high-NaN columns removed
    """
    initial_cols = X.shape[1]
    # Keep columns where non-NaN count >= (1 - threshold) * total rows
    min_valid = int(len(X) * (1 - threshold))
    X_clean = X.dropna(axis=1, thresh=min_valid)
    removed = initial_cols - X_clean.shape[1]
    print(f"  Removed {removed} columns with >{threshold*100:.0f}% missing values")
    return X_clean


def remove_constant_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with zero or near-zero variance.
    
    Args:
        X: Input features DataFrame
    
    Returns:
        DataFrame with constant columns removed
    """
    initial_cols = X.shape[1]
    # Calculate variance (ignoring NaNs)
    variances = X.var(skipna=True)
    non_constant_cols = variances[variances > 1e-10].index
    X_clean = X[non_constant_cols]
    removed = initial_cols - X_clean.shape[1]
    print(f"  Removed {removed} constant/near-constant columns")
    return X_clean


def impute_missing_values(X: pd.DataFrame, strategy: str = "median") -> tuple:
    """
    Fill missing values using specified strategy.
    
    Args:
        X: Input features DataFrame (may contain NaNs)
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')
    
    Returns:
        X_imputed: Array with no missing values
        imputer: Fitted imputer object (save for inference)
    """
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X)
    remaining_nans = np.isnan(X_imputed).sum()
    print(f"  Imputed missing values using {strategy}. Remaining NaNs: {remaining_nans}")
    return X_imputed, imputer


def apply_variance_threshold(X: np.ndarray, threshold: float = 0.01) -> tuple:
    """
    Remove low-variance features.
    
    Args:
        X: Input features array (already imputed)
        threshold: Minimum variance to keep a feature
    
    Returns:
        X_selected: Array with low-variance features removed
        selector: Fitted selector object (save for inference)
    """
    initial_features = X.shape[1]
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    removed = initial_features - X_selected.shape[1]
    print(f"  Removed {removed} low-variance features (threshold={threshold})")
    print(f"  Remaining features: {X_selected.shape[1]}")
    return X_selected, selector


def scale_features(X: np.ndarray) -> tuple:
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        X: Input features array
    
    Returns:
        X_scaled: Standardized features
        scaler: Fitted scaler object (save for inference)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Scaled features to zero mean and unit variance")
    return X_scaled, scaler


class DataPreprocessor:
    """
    Complete preprocessing pipeline for SECOM data.
    Encapsulates all steps and stores fitted transformers for inference.
    """
    
    def __init__(self, nan_threshold: float = 0.4, variance_threshold: float = 0.01,
                 impute_strategy: str = "median"):
        self.nan_threshold = nan_threshold
        self.variance_threshold = variance_threshold
        self.impute_strategy = impute_strategy
        
        # These will be fitted during preprocessing
        self.imputer = None
        self.selector = None
        self.scaler = None
        self.kept_columns = None
        self.feature_names = None
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X: Raw features DataFrame
        
        Returns:
            Preprocessed features array
        """
        print("\n📊 Preprocessing Pipeline")
        print("=" * 40)
        
        # Step 1: Remove high-NaN columns
        X_clean = remove_high_nan_columns(X, self.nan_threshold)
        self.kept_columns = X_clean.columns.tolist()
        
        # Step 2: Remove constant columns
        X_clean = remove_constant_columns(X_clean)
        self.kept_columns = X_clean.columns.tolist()
        
        # Step 3: Impute missing values
        X_imputed, self.imputer = impute_missing_values(X_clean, self.impute_strategy)
        
        # Step 4: Variance threshold feature selection
        X_selected, self.selector = apply_variance_threshold(X_imputed, self.variance_threshold)
        
        # Step 5: Scale features
        X_scaled, self.scaler = scale_features(X_selected)
        
        # Store feature names for later reference
        if self.selector is not None:
            selected_mask = self.selector.get_support()
            self.feature_names = [self.kept_columns[i] for i, keep in enumerate(selected_mask) if keep]
        
        print(f"\n✓ Final shape: {X_scaled.shape}")
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: New features DataFrame
        
        Returns:
            Preprocessed features array
        """
        if self.imputer is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Keep only the columns we used during training
        X_subset = X[self.kept_columns] if isinstance(X, pd.DataFrame) else X
        
        # Apply transformations
        X_imputed = self.imputer.transform(X_subset)
        X_selected = self.selector.transform(X_imputed)
        X_scaled = self.scaler.transform(X_selected)
        
        return X_scaled
    
    def save(self, path_prefix: str):
        """Save fitted transformers to disk."""
        joblib.dump(self.imputer, f"{path_prefix}_imputer.pkl")
        joblib.dump(self.selector, f"{path_prefix}_selector.pkl")
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
        joblib.dump(self.kept_columns, f"{path_prefix}_columns.pkl")
        joblib.dump(self.feature_names, f"{path_prefix}_feature_names.pkl")
        print(f"✓ Preprocessor saved to {path_prefix}_*.pkl")
    
    @classmethod
    def load(cls, path_prefix: str):
        """Load fitted transformers from disk."""
        preprocessor = cls()
        preprocessor.imputer = joblib.load(f"{path_prefix}_imputer.pkl")
        preprocessor.selector = joblib.load(f"{path_prefix}_selector.pkl")
        preprocessor.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        preprocessor.kept_columns = joblib.load(f"{path_prefix}_columns.pkl")
        preprocessor.feature_names = joblib.load(f"{path_prefix}_feature_names.pkl")
        print(f"✓ Preprocessor loaded from {path_prefix}_*.pkl")
        return preprocessor


if __name__ == "__main__":
    # Test preprocessing pipeline
    import sys
    sys.path.insert(0, "..")
    from src.data_loader import load_data
    
    X, y = load_data(use_synthetic=True)
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    print(f"\nProcessed data shape: {X_processed.shape}")
