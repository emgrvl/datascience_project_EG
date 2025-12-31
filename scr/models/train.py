"""
Model Training for DeepScent
Train Random Forest classifier to predict fragrance families
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import joblib
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, balanced_accuracy_score
)

# Handle class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠ imbalanced-learn not available. Install with: pip install imbalanced-learn")


def load_features(filepath: str = '/files/datascience_project_EG/data/molecules_with_features.csv') -> pd.DataFrame:
    """
    Load molecule dataset with features
    
    Args:
        filepath: Path to features CSV
        
    Returns:
        DataFrame with features
    """
    print(f"\n📂 Loading features from {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} molecules")
    return df


def prepare_data(df: pd.DataFrame, 
                 target_col: str = 'fragrance_family',
                 min_family_size: int = 10,
                 use_fingerprints: bool = True) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Prepare features and target for model training
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        min_family_size: Minimum samples per family
        use_fingerprints: Whether to use fingerprint features
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    print(f"\n🔧 Preparing data for training...")
    
    # Remove molecules without target
    df_clean = df[df[target_col].notna()].copy()
    print(f"  Removed {len(df) - len(df_clean)} molecules without fragrance family")
    
    # Filter small families
    family_counts = df_clean[target_col].value_counts()
    valid_families = family_counts[family_counts >= min_family_size].index
    df_filtered = df_clean[df_clean[target_col].isin(valid_families)].copy()
    
    removed_families = len(family_counts) - len(valid_families)
    if removed_families > 0:
        print(f"  Filtered {removed_families} families with <{min_family_size} samples")
    
    print(f"  Final dataset: {len(df_filtered)} molecules, {len(valid_families)} families")
    
    # Select feature columns
    metadata_cols = [
        'CAS number', 'Principal name', 'Primary descriptor', 
        'Descriptor 2', 'Descriptor 3', 'fragrance_family', 
        'all_descriptors', 'SMILES'
    ]
    
    # RDKit descriptors
    descriptor_cols = [col for col in df_filtered.columns 
                      if col.startswith(('Mol', 'Num', 'Log', 'TPSA', 
                                        'Heavy', 'Frac', 'Balaban', 'Bertz'))]
    
    # Fingerprints
    if use_fingerprints:
        fingerprint_cols = [col for col in df_filtered.columns 
                           if col.startswith(('MorganFP', 'MACCS'))]
        feature_cols = descriptor_cols + fingerprint_cols
    else:
        feature_cols = descriptor_cols
        print(f"  ⚠ Not using fingerprints (use_fingerprints=False)")
    
    print(f"  Features: {len(feature_cols)} total")
    print(f"    - RDKit descriptors: {len(descriptor_cols)}")
    if use_fingerprints:
        print(f"    - Fingerprint bits: {len(fingerprint_cols)}")
    
    # Prepare X and y
    X = df_filtered[feature_cols].copy()
    y = df_filtered[target_col].copy()
    
    # Check for missing values
    if X.isnull().any().any():
        print(f"  ⚠ Found missing values, filling with 0")
        X = X.fillna(0)
    
    print(f"\n✓ Data prepared: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, feature_cols


def build_model(use_smote: bool = True, 
                model_type: str = 'random_forest',
                random_state: int = 42) -> Pipeline:
    """
    Build ML pipeline with optional SMOTE
    
    Args:
        use_smote: Use SMOTE for class balancing
        model_type: 'random_forest' or 'gradient_boosting'
        random_state: Random seed
        
    Returns:
        sklearn Pipeline
    """
    print(f"\n🏗️  Building model pipeline...")
    print(f"  Model type: {model_type}")
    print(f"  SMOTE: {use_smote}")
    
    # Choose classifier
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle imbalance
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
    elif model_type == 'gradient_boosting':
        classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=random_state,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Build pipeline
    if use_smote and SMOTE_AVAILABLE:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('classifier', classifier)
        ])
    else:
        if use_smote and not SMOTE_AVAILABLE:
            print("  ⚠ SMOTE not available, using class_weight='balanced' instead")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
    
    print(f"✓ Pipeline built: {' -> '.join([step[0] for step in pipeline.steps])}")
    
    return pipeline


def train_model(X: pd.DataFrame, 
                y: pd.Series,
                test_size: float = 0.2,
                use_smote: bool = True,
                model_type: str = 'random_forest',
                random_state: int = 42,
                cv_folds: int = 5) -> Tuple[Pipeline, Dict]:
    """
    Train model with cross-validation
    
    Args:
        X: Features
        y: Target
        test_size: Test set proportion
        use_smote: Use SMOTE
        model_type: Model type
        random_state: Random seed
        cv_folds: Number of CV folds
        
    Returns:
        Tuple of (trained_model, results_dict)
    """
    print(f"\n🎯 Training model...")
    
    # Split data (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"  Train set: {len(X_train)} molecules")
    print(f"  Test set: {len(X_test)} molecules")
    
    # Check class distribution
    print(f"\n  Training set class distribution:")
    train_counts = y_train.value_counts().head(10)
    for family, count in train_counts.items():
        print(f"    {family:20s}: {count:4d}")
    
    # Build model
    model = build_model(use_smote=use_smote, model_type=model_type, random_state=random_state)
    
    # Train model
    print(f"\n  Training {model_type}...")
    model.fit(X_train, y_train)
    print(f"  ✓ Training complete!")
    
    # Cross-validation
    print(f"\n  Running {cv_folds}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    print(f"  CV F1-macro scores: {cv_scores}")
    print(f"  Mean CV F1-macro: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'model_type': model_type,
        'use_smote': use_smote,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'test_balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'train_f1_macro': f1_score(y_train, y_train_pred, average='macro'),
        'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
        'cv_f1_macro_mean': cv_scores.mean(),
        'cv_f1_macro_std': cv_scores.std(),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'classification_report': classification_report(y_test, y_test_pred),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'feature_names': X.columns.tolist(),
        'classes': model.classes_.tolist() if hasattr(model, 'classes_') else sorted(y.unique())
    }
    
    # Feature importance (if available)
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        results['feature_importances'] = model.named_steps['classifier'].feature_importances_
    
    # Print results
    print(f"\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Train Accuracy: {results['train_accuracy']:.3f}")
    print(f"Test Accuracy:  {results['test_accuracy']:.3f}")
    print(f"\nTrain Balanced Accuracy: {results['train_balanced_accuracy']:.3f}")
    print(f"Test Balanced Accuracy:  {results['test_balanced_accuracy']:.3f}")
    print(f"\nTrain F1-macro: {results['train_f1_macro']:.3f}")
    print(f"Test F1-macro:  {results['test_f1_macro']:.3f}")
    print(f"CV F1-macro:    {results['cv_f1_macro_mean']:.3f} (+/- {results['cv_f1_macro_std']:.3f})")
    
    # Check for overfitting
    if results['train_accuracy'] - results['test_accuracy'] > 0.15:
        print(f"\n⚠ Warning: Possible overfitting detected!")
        print(f"   Train-Test accuracy gap: {results['train_accuracy'] - results['test_accuracy']:.3f}")
    
    return model, results


def save_model(model: Pipeline, 
               results: Dict,
               output_dir: str = '/files/datascience_project_EG/results/models',
               save_name: str = None) -> str:
    """
    Save trained model and results
    
    Args:
        model: Trained pipeline
        results: Results dictionary
        output_dir: Output directory
        save_name: Custom save name
        
    Returns:
        Path to saved model
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_name is None:
        save_name = f"model_{results['model_type']}_{timestamp}"
    
    # Save model
    model_path = f"{output_dir}/{save_name}.joblib"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Save results (exclude large arrays)
    results_to_save = {
        k: v for k, v in results.items() 
        if k not in ['X_train', 'X_test', 'y_train', 'y_test', 
                     'y_train_pred', 'y_test_pred', 'confusion_matrix']
    }
    results_path = f"{output_dir}/{save_name}_results.joblib"
    joblib.dump(results_to_save, results_path)
    print(f"💾 Results saved: {results_path}")
    
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DeepScent fragrance family classifier')
    parser.add_argument('--input', type=str,
                       default='/files/datascience_project_EG/data/molecules_with_features.csv',
                       help='Input features CSV')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting'],
                       help='Model type')
    parser.add_argument('--no-smote', action='store_true',
                       help='Disable SMOTE (use class weights only)')
    parser.add_argument('--no-fingerprints', action='store_true',
                       help='Use only RDKit descriptors (no fingerprints)')
    parser.add_argument('--min-family-size', type=int, default=10,
                       help='Minimum samples per family')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--output-dir', type=str, default='/files/datascience_project_EG/results/models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DEEPSCENT MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_features(args.input)
    
    # Prepare data
    X, y, feature_names = prepare_data(
        df, 
        min_family_size=args.min_family_size,
        use_fingerprints=not args.no_fingerprints
    )
    
    # Train model
    model, results = train_model(
        X, y,
        test_size=args.test_size,
        use_smote=not args.no_smote,
        model_type=args.model,
        cv_folds=args.cv_folds
    )
    
    # Save model
    model_path = save_model(model, results, output_dir=args.output_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate model: python -m src.models.evaluate")
    print(f"  2. Make predictions: python -m src.models.predict <SMILES>")
    print(f"  3. Visualize results: python -m src.visualization")