"""
Training Script for Virtual Metrology System.

Trains a Random Forest model with SMOTE oversampling for wafer yield prediction.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODELS_DIR, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS,
    MODEL_FILE, IMPUTER_FILE, SELECTOR_FILE, SCALER_FILE, FEATURE_NAMES_FILE
)
from src.data_loader import load_data
from src.preprocessing import DataPreprocessor


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 50)
    print(f"  {text}")
    print("=" * 50)


def evaluate_model(model, X_test, y_test, label_mapping=None):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True labels
        label_mapping: Dict mapping numeric labels to names
    """
    print_header("Model Evaluation")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    print("\n📊 Classification Report:")
    target_names = ['Pass (-1)', 'Fail (1)'] if label_mapping is None else list(label_mapping.values())
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    print("\n📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Pass    Fail")
    print(f"  Actual Pass  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
    print(f"  Actual Fail  [{cm[1,0]:5d}  {cm[1,1]:5d}]")
    
    # Key metrics
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
    
    try:
        auc_roc = roc_auc_score(y_test, y_prob)
    except:
        auc_roc = 0.0
    
    print("\n🎯 Key Metrics:")
    print(f"  Balanced Accuracy: {balanced_acc:.3f}")
    print(f"  Precision (Fail):  {precision:.3f}")
    print(f"  Recall (Fail):     {recall:.3f}")
    print(f"  F1-Score (Fail):   {f1:.3f}")
    print(f"  ROC-AUC:           {auc_roc:.3f}")
    
    return {
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc_roc
    }


def get_feature_importance(model, feature_names):
    """Extract and display feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print("\n🔍 Top 10 Important Features:")
    for i, idx in enumerate(indices[:10]):
        name = feature_names[idx] if feature_names else f"Feature_{idx}"
        print(f"  {i+1}. {name}: {importance[idx]:.4f}")
    
    return importance


def train_model(use_synthetic: bool = False):
    """
    Main training pipeline.
    
    Args:
        use_synthetic: Use synthetic data instead of real SECOM data
    
    Returns:
        Trained model and metrics
    """
    print_header("Virtual Metrology Model Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =====================
    # 1. LOAD DATA
    # =====================
    print_header("Step 1: Loading Data")
    X, y = load_data(use_synthetic=use_synthetic)
    
    # Flatten y if it's a DataFrame
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()
    
    print(f"\n  Total samples: {len(y)}")
    print(f"  Pass (yield): {sum(y == -1)} ({100*sum(y == -1)/len(y):.1f}%)")
    print(f"  Fail (defect): {sum(y == 1)} ({100*sum(y == 1)/len(y):.1f}%)")
    
    # =====================
    # 2. PREPROCESS DATA
    # =====================
    print_header("Step 2: Preprocessing Data")
    preprocessor = DataPreprocessor(
        nan_threshold=0.4,
        variance_threshold=0.01,
        impute_strategy="median"
    )
    X_processed = preprocessor.fit_transform(X)
    
    # =====================
    # 3. TRAIN/TEST SPLIT
    # =====================
    print_header("Step 3: Train/Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y  # Important for imbalanced data
    )
    print(f"  Training samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")
    
    # =====================
    # 4. APPLY SMOTE
    # =====================
    print_header("Step 4: SMOTE Oversampling")
    print("  ⚠️ Applying SMOTE only to training data!")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n  Before SMOTE:")
    print(f"    Pass: {sum(y_train == -1)}, Fail: {sum(y_train == 1)}")
    print(f"\n  After SMOTE:")
    print(f"    Pass: {sum(y_train_resampled == -1)}, Fail: {sum(y_train_resampled == 1)}")
    
    # =====================
    # 5. TRAIN MODEL
    # =====================
    print_header("Step 5: Training Random Forest")
    
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n  Model: {model.__class__.__name__}")
    print(f"  Estimators: {N_ESTIMATORS}")
    print(f"  Training...")
    
    model.fit(X_train_resampled, y_train_resampled)
    print("  ✓ Training complete!")
    
    # =====================
    # 6. EVALUATE MODEL
    # =====================
    metrics = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feature_names = preprocessor.feature_names or [f"Sensor_{i}" for i in range(X_processed.shape[1])]
    importance = get_feature_importance(model, feature_names)
    
    # =====================
    # 7. SAVE ARTIFACTS
    # =====================
    print_header("Step 6: Saving Model Artifacts")
    
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"  ✓ Model saved: {MODEL_FILE}")
    
    # Save preprocessor components
    joblib.dump(preprocessor.imputer, IMPUTER_FILE)
    print(f"  ✓ Imputer saved: {IMPUTER_FILE}")
    
    joblib.dump(preprocessor.selector, SELECTOR_FILE)
    print(f"  ✓ Selector saved: {SELECTOR_FILE}")
    
    joblib.dump(preprocessor.scaler, SCALER_FILE)
    print(f"  ✓ Scaler saved: {SCALER_FILE}")
    
    joblib.dump({
        'feature_names': feature_names,
        'kept_columns': preprocessor.kept_columns,
        'importance': importance
    }, FEATURE_NAMES_FILE)
    print(f"  ✓ Feature info saved: {FEATURE_NAMES_FILE}")
    
    # =====================
    # SUMMARY
    # =====================
    print_header("Training Complete!")
    print(f"""
  📊 Model Performance:
     - Balanced Accuracy: {metrics['balanced_accuracy']:.1%}
     - Fail Detection Rate (Recall): {metrics['recall']:.1%}
     - F1-Score: {metrics['f1']:.3f}
     - ROC-AUC: {metrics['roc_auc']:.3f}
  
  📁 Saved Artifacts:
     - {MODEL_FILE}
     - {IMPUTER_FILE}
     - {SELECTOR_FILE}
     - {SCALER_FILE}
     - {FEATURE_NAMES_FILE}
  
  🚀 Next Step: Run the dashboard with `streamlit run app.py`
    """)
    
    return model, preprocessor, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Virtual Metrology Model")
    parser.add_argument("--synthetic", action="store_true", 
                        help="Use synthetic data for testing")
    args = parser.parse_args()
    
    train_model(use_synthetic=args.synthetic)
