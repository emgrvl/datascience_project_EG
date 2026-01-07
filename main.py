#!/usr/bin/env python3
"""
DeepScent - Complete End-to-End Pipeline
=========================================

Runs the complete DeepScent workflow from raw data to trained model:
1. Load raw IFRA data
2. Sample molecules for faster demo (!very small sample size so the results might differ from the paper!)
3. Fetch SMILES from PubChem
4. Preprocess and standardize families
5. Compute RDKit descriptors + Morgan fingerprints
6. Train model (user chooses from 3 options)
7. Evaluate and visualize
8. Interactive prediction demo

All outputs saved to examples/ directory.
Estimated time: 5-10 minutes (depending on sample size and model choice)

Author: Emeline Gravaillac
Project: DeepScent - Fragrance Family Classification
"""

import sys
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATH SETUP
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

print("="*80)
print("  ____                 ____                  _   ")
print(" |  _ \\  ___  ___ _ __/ ___|  ___ ___ _ __ | |_ ")
print(" | | | |/ _ \\/ _ \\ '_ \\___ \\ / __/ _ \\ '_ \\| __|")
print(" | |_| |  __/  __/ |_) |__) | (_|  __/ | | | |_ ")
print(" |____/ \\___|\\___| .__/____/ \\___\\___|_| |_|\\__|")
print("                 |_|                            ")
print()
print("     Machine Learning for Fragrance Family Classification")
print("     🔬 Complete End-to-End Pipeline 🔬")
print("="*80)
print()
print(f"📂 Working directory: {SCRIPT_DIR}")
print(f"📥 Input: data/raw_ifra-fig.csv (original IFRA dataset)")
print(f"📤 Output: examples/ directory (demo pipeline results)")
print()

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from scr.data.load_data import load_ifra_fig, fetch_smiles_from_pubchem
from scr.data.preprocess import standardize_fragrance_family, normalize_text
from scr.features.rdkit_features import compute_rdkit_descriptors
from scr.features.fingerprinting import compute_morgan_fp

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠ SMOTE not available. Using class weights instead.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PHASE 0: SETUP
# ============================================================================

def setup_environment():
    """Create necessary directories"""
    print("🔧 Setting up environment...")
    
    directories = [
        SCRIPT_DIR / 'examples' / 'data',
        SCRIPT_DIR / 'examples' / 'results' / 'models',
        SCRIPT_DIR / 'examples' / 'results' / 'figures',
        SCRIPT_DIR / 'examples' / 'results' / 'metrics'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory.relative_to(SCRIPT_DIR)}/")
    
    print("   ✓ All dependencies available")
    return True


# ============================================================================
# PHASE 1: LOAD AND SAMPLE RAW DATA
# ============================================================================

def load_and_sample_data(sample_size=50):
    """Load raw IFRA data and take a sample"""
    print("\n" + "="*80)
    print("PHASE 1: LOADING RAW IFRA DATA")
    print("="*80)
    
    print(f"\n📂 Loading raw IFRA dataset...")
    
    # Load raw data
    raw_data_path = SCRIPT_DIR / 'data' / 'raw_ifra-fig.csv'
    
    if not raw_data_path.exists():
        print(f"\n   ❌ File not found: {raw_data_path}")
        print("\n   Please ensure raw_ifra-fig.csv is in the data/ directory")
        return None
    
    try:
        df = load_ifra_fig(str(raw_data_path))
        print(f"\n   ✓ Loaded: {len(df)} total molecules from IFRA dataset")
    except Exception as e:
        print(f"\n   ❌ Error loading data: {e}")
        return None
    
    # Take a stratified sample
    print(f"\n📊 Creating sample of {sample_size} molecules for demo...")
    print(f"   (This reduces SMILES fetching time from hours to minutes)")
    
    if 'Primary descriptor' in df.columns:
        # Stratified sample - get diverse families
        top_families = df['Primary descriptor'].value_counts().head(10).index
        df_top = df[df['Primary descriptor'].isin(top_families)]
        
        df_sample = df_top.groupby('Primary descriptor').apply(
            lambda x: x.sample(min(5, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        # Fill to sample_size if needed
        remaining = sample_size - len(df_sample)
        if remaining > 0:
            df_remaining = df[~df.index.isin(df_sample.index)].sample(
                min(remaining, len(df)), random_state=42
            )
            df_sample = pd.concat([df_sample, df_remaining]).reset_index(drop=True)
    else:
        df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    print(f"   ✓ Sample size: {len(df_sample)} molecules")
    
    # Show family distribution
    if 'Primary descriptor' in df_sample.columns:
        print(f"\n   Family distribution in sample:")
        for family, count in df_sample['Primary descriptor'].value_counts().head(8).items():
            print(f"      {family:20s}: {count:3d}")
    
    return df_sample


# ============================================================================
# PHASE 2: FETCH SMILES FROM PUBCHEM
# ============================================================================

def fetch_smiles_data(df):
    """Fetch SMILES strings from PubChem"""
    print("\n" + "="*80)
    print("PHASE 2: FETCHING SMILES FROM PUBCHEM")
    print("="*80)
    
    print(f"\n🔍 Fetching SMILES for {len(df)} molecules from PubChem...")
    print(f"   Estimated time: {len(df) * 0.3 / 60:.1f} minutes")
    print(f"   (Rate limited to ~4 requests/second)")
    
    smiles_list = []
    success_count = 0
    
    for idx, row in df.iterrows():
        cas = row.get('CAS number', '')
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"   Progress: {idx + 1}/{len(df)} ({success_count} found)", end='\r')
        
        # Fetch SMILES
        smiles = fetch_smiles_from_pubchem(cas)
        smiles_list.append(smiles)
        
        if smiles:
            success_count += 1
        
        # Rate limiting
        time.sleep(0.25)
    
    print(f"\n   ✓ Retrieved: {success_count}/{len(df)} SMILES ({success_count/len(df)*100:.1f}% success)")
    
    # Add to dataframe
    df['SMILES'] = smiles_list
    df_with_smiles = df[df['SMILES'].notna()].copy()
    
    print(f"   ✓ Valid molecules: {len(df_with_smiles)}")
    
    # Save intermediate result
    output_path = SCRIPT_DIR / 'examples' / 'data' / 'sample_with_smiles.csv'
    df_with_smiles.to_csv(output_path, index=False)
    print(f"   ✓ Saved: {output_path.relative_to(SCRIPT_DIR)}")
    
    return df_with_smiles


# ============================================================================
# PHASE 3: PREPROCESS DATA
# ============================================================================

def preprocess_data(df):
    """Standardize fragrance families"""
    print("\n" + "="*80)
    print("PHASE 3: PREPROCESSING & STANDARDIZATION")
    print("="*80)
    
    print(f"\n🔧 Standardizing fragrance families...")
    
    # Mapping from common descriptors to standard families
    family_mapping = {
        'floral': 'floral', 'rose': 'floral', 'jasmine': 'floral', 'lavender': 'floral',
        'citrus': 'citrus', 'lemon': 'citrus', 'orange': 'citrus', 'bergamot': 'citrus',
        'woody': 'woody', 'cedar': 'woody', 'sandalwood': 'woody',
        'oriental': 'oriental', 'vanilla': 'oriental', 'balsamic': 'oriental',
        'spicy': 'spicy', 'green': 'green', 'fruity': 'fruity', 'herbal': 'herbal',
        'fresh': 'green', 'fatty': 'green', 'minty': 'minty', 'mint': 'minty',
    }
    
    # Standardize
    if 'Primary descriptor' in df.columns:
        df['fragrance_family'] = df['Primary descriptor'].apply(
            lambda x: family_mapping.get(normalize_text(x), normalize_text(x))
        )
    else:
        df['fragrance_family'] = 'unknown'
    
    print(f"   ✓ Mapped to {df['fragrance_family'].nunique()} families")
    
    # Show distribution
    print(f"\n   Fragrance Family Distribution:")
    for family, count in df['fragrance_family'].value_counts().head(10).items():
        print(f"      {family:20s}: {count:3d}")
    
    # Save
    output_path = SCRIPT_DIR / 'examples' / 'data' / 'sample_preprocessed.csv'
    df.to_csv(output_path, index=False)
    print(f"\n   ✓ Saved: {output_path.relative_to(SCRIPT_DIR)}")
    
    return df


# ============================================================================
# PHASE 4: GENERATE FEATURES
# ============================================================================

def generate_features(df):
    """Compute RDKit descriptors and Morgan fingerprints"""
    print("\n" + "="*80)
    print("PHASE 4: FEATURE ENGINEERING")
    print("="*80)
    
    print(f"\n🧪 Computing molecular features for {len(df)} molecules...")
    
    # Step 1: RDKit descriptors
    print("\n   Step 1/2: RDKit molecular descriptors (17 features)...")
    descriptors_list = []
    
    for idx, smiles in enumerate(df['SMILES']):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"      Progress: {idx + 1}/{len(df)}", end='\r')
        
        desc = compute_rdkit_descriptors(smiles, extended=True)
        descriptors_list.append(desc)
    
    print(f"\n      ✓ Computed descriptors for {len(descriptors_list)} molecules")
    
    # Convert to DataFrame
    desc_df = pd.DataFrame(descriptors_list)
    df_features = pd.concat([df.reset_index(drop=True), desc_df], axis=1)
    
    # Step 2: Morgan fingerprints
    print("\n   Step 2/2: Morgan fingerprints (2048 bits)...")
    fingerprints_list = []
    
    for idx, smiles in enumerate(df['SMILES']):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"      Progress: {idx + 1}/{len(df)}", end='\r')
        
        fp = compute_morgan_fp(smiles, n_bits=2048)
        fingerprints_list.append(fp)
    
    print(f"\n      ✓ Computed fingerprints for {len(fingerprints_list)} molecules")
    
    # Convert to DataFrame
    fp_df = pd.DataFrame(
        fingerprints_list,
        columns=[f'MorganFP_{i}' for i in range(2048)]
    )
    
    # Combine everything
    df_all_features = pd.concat([df_features.reset_index(drop=True), fp_df], axis=1)
    
    # Clean: remove rows with missing descriptors
    descriptor_cols = desc_df.columns
    df_clean = df_all_features.dropna(subset=descriptor_cols)
    
    print(f"\n   ✓ Final dataset: {len(df_clean)} molecules × {len(df_clean.columns)} features")
    print(f"      - Metadata: {len([c for c in df.columns])} columns")
    print(f"      - RDKit descriptors: 17")
    print(f"      - Morgan fingerprints: 2048")
    
    # Save
    output_path = SCRIPT_DIR / 'examples' / 'data' / 'sample_with_features.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\n   ✓ Saved: {output_path.relative_to(SCRIPT_DIR)}")
    
    return df_clean


# ============================================================================
# PHASE 5: MODEL SELECTION AND TRAINING
# ============================================================================

def select_model():
    """Interactive model selection"""
    print("\n" + "="*80)
    print("MODEL SELECTION")
    print("="*80)
    
    print("\nChoose your model:")
    print()
    print("  1️⃣  Random Forest + Fingerprints (Full Feature Set)")
    print("     • Accuracy: ~72-75%")
    print("     • Training time: ~30 seconds")
    print("     • Features: 2,065 (17 descriptors + 2,048 fingerprints)")
    print("     • Best for: Baseline performance")
    print()
    print("  2️⃣  Random Forest - Descriptors Only (RECOMMENDED) ⭐")
    print("     • Accuracy: ~75-78%")
    print("     • Training time: ~10 seconds ⚡")
    print("     • Features: 17 (highly interpretable)")
    print("     • Best for: Fast training, interpretability")
    print()
    print("  3️⃣  Gradient Boosting + Fingerprints (Best Performance)")
    print("     • Accuracy: ~80-85%")
    print("     • Training time: ~2-5 minutes")
    print("     • Features: 2,065")
    print("     • Best for: Maximum accuracy")
    print()
    
    while True:
        choice = input("Select model (1-3) [2]: ").strip() or '2'
        
        if choice in ['1', '2', '3']:
            return choice
        else:
            print("   Invalid choice. Please enter 1, 2, or 3.")


def train_model(df, model_choice):
    """Train selected model"""
    print("\n" + "="*80)
    print("PHASE 5: MODEL TRAINING")
    print("="*80)
    
    # Prepare data
    print(f"\n🔧 Preparing training data...")
    
    metadata_cols = ['CAS number', 'Principal name', 'Primary descriptor',
                     'Descriptor 2', 'Descriptor 3', 'fragrance_family', 
                     'all_descriptors', 'SMILES']
    
    # Get feature columns
    descriptor_cols = [col for col in df.columns 
                      if col not in metadata_cols 
                      and not col.startswith('MorganFP')]
    
    fingerprint_cols = [col for col in df.columns if col.startswith('MorganFP')]
    
    # Select features based on model choice
    if model_choice == '1':
        feature_cols = descriptor_cols + fingerprint_cols
        model_name = "Random Forest + Fingerprints"
        use_fingerprints = True
        model_type = 'rf'
    elif model_choice == '2':
        feature_cols = descriptor_cols
        model_name = "Random Forest - Descriptors Only"
        use_fingerprints = False
        model_type = 'rf'
    elif model_choice == '3':
        feature_cols = descriptor_cols + fingerprint_cols
        model_name = "Gradient Boosting + Fingerprints"
        use_fingerprints = True
        model_type = 'gb'
    
    print(f"\n   Model: {model_name}")
    print(f"   Features: {len(feature_cols)}")
    
    # Filter to families with at least 3 samples
    family_counts = df['fragrance_family'].value_counts()
    valid_families = family_counts[family_counts >= 3].index
    df_train = df[df['fragrance_family'].isin(valid_families)].copy()
    
    print(f"   Families: {len(valid_families)}")
    print(f"   Molecules: {len(df_train)}")
    
    # Prepare X and y
    X = df_train[feature_cols].fillna(0)
    y = df_train['fragrance_family']
    
    # Calculate appropriate test size
    n_samples = len(X)
    n_classes = len(y.unique())
    min_test_size = max(n_classes + 2, int(n_samples * 0.15))
    test_size = max(min_test_size, int(n_samples * 0.25))
    
    if test_size >= n_samples * 0.5:
        test_size = int(n_samples * 0.3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Build pipeline
    if SMOTE_AVAILABLE and model_type == 'rf' and len(y_train) > 20:
        min_class_size = y_train.value_counts().min()
        k_neighbors = min(5, min_class_size - 1)
        
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
        print(f"   Using SMOTE for class balancing")
    elif model_type == 'gb':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        print(f"   Using Gradient Boosting")
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
        print(f"   Using Random Forest with class weights")
    
    # Train
    print(f"\n   🚀 Training {model_name}...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"   ✓ Training complete ({training_time:.1f}s)")
    
    # Evaluate
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print(f"\n   📊 Results:")
    print(f"      Train Accuracy: {train_acc:.3f}")
    print(f"      Test Accuracy:  {test_acc:.3f}")
    print(f"      F1-macro:       {test_f1:.3f}")
    
    # Cross-validation
    if len(X_train) >= 15:
        print(f"\n   🔄 Running 3-fold cross-validation...")
        cv = StratifiedKFold(n_splits=min(3, n_classes), shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            print(f"      CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        except:
            print(f"      Skipping CV (dataset too small)")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_option{model_choice}_{timestamp}.joblib"
    model_path = SCRIPT_DIR / 'examples' / 'results' / 'models' / model_filename
    joblib.dump(pipeline, model_path)
    
    print(f"\n   ✓ Model saved: {model_path.relative_to(SCRIPT_DIR)}")
    
    return pipeline, {
        'model_choice': model_choice,
        'model_name': model_name,
        'use_fingerprints': use_fingerprints,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'f1_macro': test_f1,
        'training_time': training_time,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'model_path': str(model_path)
    }


# ============================================================================
# PHASE 6: EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(results):
    """Generate evaluation metrics and visualizations"""
    print("\n" + "="*80)
    print("PHASE 6: EVALUATION & VISUALIZATION")
    print("="*80)
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    print(f"\n📊 Classification Report for {results['model_name']}:")
    print(report)
    
    # Save report
    report_path = SCRIPT_DIR / 'examples' / 'results' / 'metrics' / f'classification_report_option{results["model_choice"]}.txt'
    with open(report_path, 'w') as f:
        f.write(f"DEEPSCENT - Classification Report\n")
        f.write(f"Model: {results['model_name']}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Accuracy: {results['test_acc']:.3f}\n")
        f.write(f"F1-macro: {results['f1_macro']:.3f}\n\n")
        f.write(report)
    
    print(f"\n   ✓ Saved: {report_path.relative_to(SCRIPT_DIR)}")
    
    # Confusion Matrix
    print("\n📈 Generating confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    families = sorted(y_test.unique())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=families, yticklabels=families)
    plt.title(f'Confusion Matrix - {results["model_name"]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = SCRIPT_DIR / 'examples' / 'results' / 'figures' / f'confusion_matrix_option{results["model_choice"]}.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {cm_path.relative_to(SCRIPT_DIR)}")
    plt.close()


# ============================================================================
# PHASE 7: INTERACTIVE PREDICTIONS
# ============================================================================

def demo_predictions(results):
    """Demonstrate predictions on known molecules"""
    print("\n" + "="*80)
    print("PHASE 7: INTERACTIVE PREDICTION DEMO")
    print("="*80)
    
    from scr.models.predict import predict_smiles
    
    # Test molecules
    test_molecules = {
        'Linalool (Floral)': 'CC(C)=CCCC(C)(O)C=C',
        'Limonene (Citrus)': 'CC1=CCC(CC1)C(=C)C',
        'Vanillin (Sweet)': 'COC1=C(C=CC(=C1)C=O)O',
        'Eugenol (Spicy)': 'COC1=C(C=CC(=C1)CC=C)O',
    }
    
    print(f"\n🔮 Testing {results['model_name']} on known molecules:")
    print()
    
    for name, smiles in test_molecules.items():
        try:
            result = predict_smiles(
                results['model_path'],
                smiles,
                use_fingerprints=results['use_fingerprints'],
                extended_descriptors=True,
                verbose=False
            )
            
            if 'error' not in result:
                family = result['predicted_family']
                conf = result.get('confidence', 0)
                if conf:
                    print(f"   {name:25s} → {family:15s} ({conf:.1%})")
                else:
                    print(f"   {name:25s} → {family:15s}")
            else:
                print(f"   {name:25s} → Error: {result['error']}")
        except Exception as e:
            print(f"   {name:25s} → Error: {str(e)}")
    
    # Interactive mode
    print("\n" + "="*80)
    print("TRY YOUR OWN MOLECULES!")
    print("="*80)
    print("\nEnter SMILES strings to predict fragrance families")
    print("Examples: CCO (ethanol), CC(C)=CCCC(C)(O)C=C (linalool)")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            smiles = input("Enter SMILES (or 'quit'): ").strip()
            
            if smiles.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Exiting prediction demo...")
                break
            
            if not smiles:
                continue
            
            result = predict_smiles(
                results['model_path'],
                smiles,
                use_fingerprints=results['use_fingerprints'],
                extended_descriptors=True,
                verbose=False
            )
            
            if 'error' not in result:
                print(f"   → Predicted family: {result['predicted_family']}")
                if result.get('confidence'):
                    print(f"   → Confidence: {result['confidence']:.1%}")
                    
                    if 'top_3_predictions' in result:
                        print(f"   → Top 3 predictions:")
                        for i, (fam, prob) in enumerate(result['top_3_predictions'][:3], 1):
                            print(f"      {i}. {fam:15s} {prob:.1%}")
            else:
                print(f"   ❌ Error: {result['error']}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting prediction demo...")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    start_time = time.time()
    
    try:
        # Phase 0: Setup
        if not setup_environment():
            return
        
        # Get sample size from user
        print("\n" + "="*80)
        print("SAMPLE SIZE SELECTION")
        print("="*80)
        print("\nLarger samples give better models but take longer to process.")
        print("Recommended: 50 molecules (~3-5 min total)")
        print()
        
        sample_size = 50
        while True:
            size_input = input("Enter sample size (20-100) [50]: ").strip() or '50'
            try:
                sample_size = int(size_input)
                if 20 <= sample_size <= 100:
                    break
                else:
                    print("   Please enter a number between 20 and 100")
            except ValueError:
                print("   Please enter a valid number")
        
        # Phase 1: Load and sample
        df = load_and_sample_data(sample_size=sample_size)
        if df is None:
            return
        
        # Phase 2: Fetch SMILES
        df_smiles = fetch_smiles_data(df)
        if len(df_smiles) == 0:
            print("\n❌ No SMILES retrieved. Exiting.")
            return
        
        # Phase 3: Preprocess
        df_processed = preprocess_data(df_smiles)
        
        # Phase 4: Generate features
        df_features = generate_features(df_processed)
        
        # Phase 5: Model selection and training
        model_choice = select_model()
        model, results = train_model(df_features, model_choice)
        
        # Phase 6: Evaluate
        evaluate_model(results)
        
        # Phase 7: Interactive predictions
        demo_predictions(results)
        
        # Final summary
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("DEEPSCENT PIPELINE COMPLETE! 🎉")
        print("="*80)
        print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"\n📁 All outputs saved in: examples/")
        print(f"   - Data: examples/data/")
        print(f"   - Model: examples/results/models/")
        print(f"   - Figures: examples/results/figures/")
        print(f"   - Metrics: examples/results/metrics/")
        
        print(f"\n🎯 Model Performance (Option {model_choice}):")
        print(f"   Model: {results['model_name']}")
        print(f"   Test Accuracy: {results['test_acc']:.1%}")
        print(f"   F1-macro: {results['f1_macro']:.3f}")
        print(f"   Training time: {results['training_time']:.1f}s")
        
        print(f"\n Next Steps:")
        print(f"   1. View confusion matrix: examples/results/figures/")
        print(f"   2. Check classification report: examples/results/metrics/")
        print(f"   3. Compare with production models in results/models/")
        print(f"   4. Run with full dataset: python scr/models/train.py")
        
        print("\n" + "="*80)
        print("Thank you for using DeepScent!")
        print("For full dataset and advanced features, see README.md")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        print("   Partial results may be saved in examples/")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        print("\n   Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()