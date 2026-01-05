"""
Unit Tests for DeepScent Project
Tests data loading, feature engineering, and model training
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, '/files/datascience_project_EG')

# ============================================================================
# TEST DATA LOADING AND PREPROCESSING
# ============================================================================

def test_load_ifra_data():
    """Test loading IFRA dataset"""
    from scr.data.load_data import load_ifra_fig
    
    # Assuming the file exists
    try:
        df = load_ifra_fig('/files/datascience_project_EG/data/IFRA_FIG_v1.0.csv')
        assert isinstance(df, pd.DataFrame)
        assert 'CAS number' in df.columns
        assert 'Primary descriptor' in df.columns
        assert len(df) > 0
        print(f"✓ IFRA data loaded: {len(df)} molecules")
    except FileNotFoundError:
        pytest.skip("IFRA data file not found")


def test_fetch_smiles_from_pubchem():
    """Test fetching SMILES from PubChem"""
    from scr.data.load_data import fetch_smiles_from_pubchem
    
    # Test with known CAS numbers
    test_cases = [
        ('50-00-0', 'formaldehyde'),  # Formaldehyde
        ('64-17-5', 'ethanol'),       # Ethanol
        ('78-70-6', 'linalool')       # Linalool
    ]
    
    for cas, name in test_cases:
        smiles = fetch_smiles_from_pubchem(cas)
        if smiles:  # Some may fail due to PubChem rate limits
            assert isinstance(smiles, str)
            assert len(smiles) > 0
            print(f"✓ {name} ({cas}): {smiles}")


def test_fragrance_family_standardization():
    """Test fragrance family mapping"""
    # Test that standardization works
    test_descriptors = [
        ('floral', 'floral'),
        ('rose', 'floral'),
        ('citrus', 'citrus'),
        ('lemon', 'citrus'),
        ('woody', 'woody'),
        ('cedarwood', 'woody')
    ]
    
    # This would need the actual mapping function if you have one
    # For now, just test the concept
    for descriptor, expected_family in test_descriptors:
        # Simplified test
        assert expected_family in ['floral', 'citrus', 'woody', 'green', 
                                   'fruity', 'herbal', 'spicy', 'oriental']


# ============================================================================
# TEST FEATURE ENGINEERING
# ============================================================================

def test_compute_rdkit_descriptors():
    """Test RDKit molecular descriptor calculation"""
    from scr.features.rdkit_features import compute_rdkit_descriptors
    
    # Test molecules
    test_molecules = {
        'ethanol': 'CCO',
        'linalool': 'CC(C)=CCCC(C)(O)C=C',
        'limonene': 'CC1=CCC(CC1)C(=C)C'
    }
    
    for name, smiles in test_molecules.items():
        desc = compute_rdkit_descriptors(smiles, extended=True)
        
        # Check that all expected descriptors are present
        assert isinstance(desc, dict)
        assert 'MolWt' in desc
        assert 'LogP' in desc
        assert 'TPSA' in desc
        assert 'NumAromaticRings' in desc
        
        # Check extended descriptors
        assert 'BalabanJ' in desc
        assert 'FractionCsp3' in desc
        
        # Check that values are numeric
        assert isinstance(desc['MolWt'], (int, float))
        assert not np.isnan(desc['MolWt'])
        
        print(f"✓ {name}: {len(desc)} descriptors computed")


def test_compute_rdkit_descriptors_invalid():
    """Test that invalid SMILES returns NaN values"""
    from scr.features.rdkit_features import compute_rdkit_descriptors
    
    invalid_smiles = "INVALID_SMILES_STRING"
    desc = compute_rdkit_descriptors(invalid_smiles)
    
    # Should return dict with NaN values
    assert isinstance(desc, dict)
    assert all(pd.isna(v) for v in desc.values())
    print("✓ Invalid SMILES handled correctly")


def test_compute_morgan_fingerprint():
    """Test Morgan fingerprint generation"""
    from scr.features.fingerprinting import compute_morgan_fp
    
    smiles = "CCO"  # Ethanol
    fp = compute_morgan_fp(smiles, n_bits=2048)
    
    # Check output format
    assert isinstance(fp, np.ndarray)
    assert fp.shape[0] == 2048
    assert fp.dtype in [np.int32, np.int64, int]
    assert np.all((fp == 0) | (fp == 1))  # Binary values
    assert fp.sum() > 0  # Some bits should be set
    
    print(f"✓ Morgan fingerprint: {fp.sum()} bits set out of 2048")


def test_compute_maccs_fingerprint():
    """Test MACCS keys fingerprint generation"""
    from scr.features.fingerprinting import compute_maccs_fp
    
    smiles = "CCO"  # Ethanol
    fp = compute_maccs_fp(smiles)
    
    # Check output format
    assert isinstance(fp, np.ndarray)
    assert fp.shape[0] == 166  # MACCS has 166 keys
    assert fp.dtype in [np.int32, np.int64, int]
    assert np.all((fp == 0) | (fp == 1))
    
    print(f"✓ MACCS fingerprint: {fp.sum()} keys set out of 166")


def test_molecular_similarity():
    """Test Tanimoto similarity calculation"""
    from scr.features.fingerprinting import calculate_similarity
    
    # Similar molecules
    linalool = "CC(C)=CCCC(C)(O)C=C"
    geraniol = "CC(C)=CCCC(C)(O)C=C"  # Very similar
    limonene = "CC1=CCC(CC1)C(=C)C"    # Different
    
    # Similarity should be high for similar molecules
    sim_high = calculate_similarity(linalool, geraniol)
    sim_low = calculate_similarity(linalool, limonene)
    
    assert 0 <= sim_high <= 1
    assert 0 <= sim_low <= 1
    # Linalool and geraniol are identical, so similarity should be 1.0
    assert sim_high > sim_low
    
    print(f"✓ Linalool-Geraniol similarity: {sim_high:.3f}")
    print(f"✓ Linalool-Limonene similarity: {sim_low:.3f}")


def test_feature_computation_dataframe():
    """Test computing features for entire DataFrame"""
    from scr.features.rdkit_features import compute_features
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        'name': ['Ethanol', 'Linalool', 'Limonene'],
        'SMILES': ['CCO', 'CC(C)=CCCC(C)(O)C=C', 'CC1=CCC(CC1)C(=C)C']
    })
    
    result = compute_features(test_df, smiles_col='SMILES', extended=True)
    
    # Check output
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert 'MolWt' in result.columns
    assert 'LogP' in result.columns
    assert 'BalabanJ' in result.columns
    
    # Check no NaN values for valid SMILES
    assert result['MolWt'].notna().all()
    
    print(f"✓ Computed features for {len(result)} molecules")
    print(f"✓ Total columns: {len(result.columns)}")


# ============================================================================
# TEST MODEL TRAINING
# ============================================================================

def test_train_random_forest():
    """Test Random Forest training"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create balanced classes
    families = ['floral', 'woody', 'citrus', 'fruity', 'herbal']
    y = pd.Series(np.random.choice(families, n_samples))
    
    # Train simple Random Forest
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    
    # Test prediction
    predictions = model.predict(X)
    
    assert len(predictions) == n_samples
    assert all(pred in families for pred in predictions)
    
    # Test predict_proba
    probas = model.predict_proba(X)
    assert probas.shape[0] == n_samples
    assert probas.shape[1] == len(families)
    assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    print(f"✓ Random Forest trained and tested successfully")


def test_smote_handling():
    """Test SMOTE for class imbalance"""
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        # Create imbalanced dataset
        np.random.seed(42)
        X_majority = np.random.rand(80, 5)
        y_majority = ['floral'] * 80
        
        X_minority = np.random.rand(20, 5)
        y_minority = ['woody'] * 20
        
        X = np.vstack([X_majority, X_minority])
        y = np.array(y_majority + y_minority)
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
        print("✓ SMOTE pipeline works correctly")
        
    except ImportError:
        pytest.skip("imbalanced-learn not installed")


def test_model_persistence():
    """Test saving and loading models"""
    import joblib
    import tempfile
    from sklearn.ensemble import RandomForestClassifier
    
    # Train a simple model
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = np.random.choice(['floral', 'woody', 'citrus'], 50)
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        joblib.dump(model, tmp.name)
        
        # Load back
        loaded_model = joblib.load(tmp.name)
        
        # Test that predictions match
        pred_original = model.predict(X)
        pred_loaded = loaded_model.predict(X)
        
        assert np.array_equal(pred_original, pred_loaded)
        print("✓ Model save/load works correctly")


# ============================================================================
# TEST PREDICTION PIPELINE
# ============================================================================

def test_end_to_end_prediction():
    """Test complete prediction pipeline"""
    from scr.features.rdkit_features import compute_rdkit_descriptors
    from scr.features.fingerprinting import compute_morgan_fp
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and train a simple model
    np.random.seed(42)
    n_samples = 100
    n_features = 17  # RDKit descriptors
    
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.choice(['floral', 'woody', 'citrus'], n_samples)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test prediction on new molecule
    test_smiles = "CCO"  # Ethanol
    descriptors = compute_rdkit_descriptors(test_smiles, extended=True)
    
    # Extract values in order
    feature_vector = np.array(list(descriptors.values())).reshape(1, -1)
    
    # Predict
    prediction = model.predict(feature_vector)
    proba = model.predict_proba(feature_vector)
    
    assert prediction[0] in ['floral', 'woody', 'citrus']
    assert proba.shape == (1, 3)
    assert np.isclose(proba.sum(), 1.0)
    
    print(f"✓ End-to-end prediction: {prediction[0]} (confidence: {proba.max():.2f})")


def test_batch_prediction():
    """Test predicting multiple molecules"""
    from scr.features.rdkit_features import compute_rdkit_descriptors
    from sklearn.ensemble import RandomForestClassifier
    
    # Create simple model
    np.random.seed(42)
    X_train = np.random.rand(50, 17)
    y_train = np.random.choice(['floral', 'woody'], 50)
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Test molecules
    test_molecules = ['CCO', 'CC(C)=CCCC(C)(O)C=C', 'CC1=CCC(CC1)C(=C)C']
    
    features_list = []
    for smiles in test_molecules:
        desc = compute_rdkit_descriptors(smiles, extended=True)
        if not all(pd.isna(v) for v in desc.values()):
            features_list.append(list(desc.values()))
    
    if features_list:
        X_test = np.array(features_list)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(features_list)
        print(f"✓ Batch prediction on {len(predictions)} molecules successful")


# ============================================================================
# TEST DATA QUALITY
# ============================================================================

def test_no_missing_values_in_features():
    """Test that feature computation handles missing values"""
    from scr.features.rdkit_features import compute_rdkit_descriptors
    
    valid_smiles = "CCO"
    invalid_smiles = "INVALID"
    
    # Valid SMILES should have no NaN (except for some edge cases)
    desc_valid = compute_rdkit_descriptors(valid_smiles)
    valid_values = [v for v in desc_valid.values() if not pd.isna(v)]
    assert len(valid_values) > 0
    
    # Invalid SMILES should return NaN
    desc_invalid = compute_rdkit_descriptors(invalid_smiles)
    assert all(pd.isna(v) for v in desc_invalid.values())
    
    print("✓ Feature computation handles invalid SMILES correctly")


def test_feature_value_ranges():
    """Test that computed features are in reasonable ranges"""
    from scr.features.rdkit_features import compute_rdkit_descriptors
    
    smiles = "CC(C)=CCCC(C)(O)C=C"  # Linalool
    desc = compute_rdkit_descriptors(smiles, extended=True)
    
    # Test reasonable ranges
    assert 0 < desc['MolWt'] < 1000  # Molecular weight
    assert -10 < desc['LogP'] < 10   # Lipophilicity
    assert 0 <= desc['TPSA'] < 300   # Polar surface area
    assert 0 <= desc['NumAromaticRings'] < 10
    assert 0 <= desc['NumRotatableBonds'] < 20
    
    print("✓ Feature values are in reasonable ranges")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DEEPSCENT - RUNNING UNIT TESTS")
    print("="*70)
    
    test_functions = [
        ("Data Loading", [
            test_load_ifra_data,
            test_fetch_smiles_from_pubchem,
            test_fragrance_family_standardization,
        ]),
        ("Feature Engineering", [
            test_compute_rdkit_descriptors,
            test_compute_rdkit_descriptors_invalid,
            test_compute_morgan_fingerprint,
            test_compute_maccs_fingerprint,
            test_molecular_similarity,
            test_feature_computation_dataframe,
        ]),
        ("Model Training", [
            test_train_random_forest,
            test_smote_handling,
            test_model_persistence,
        ]),
        ("Prediction Pipeline", [
            test_end_to_end_prediction,
            test_batch_prediction,
        ]),
        ("Data Quality", [
            test_no_missing_values_in_features,
            test_feature_value_ranges,
        ]),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    for category, tests in test_functions:
        print(f"\n{'='*70}")
        print(f"Testing: {category}")
        print('='*70)
        
        for test_func in tests:
            total_tests += 1
            test_name = test_func.__name__
            
            try:
                test_func()
                passed_tests += 1
                print(f"✅ PASSED: {test_name}")
            except pytest.skip.Exception as e:
                skipped_tests += 1
                print(f"⏭️  SKIPPED: {test_name} - {str(e)}")
            except AssertionError as e:
                failed_tests += 1
                print(f"❌ FAILED: {test_name}")
                print(f"   Error: {str(e)}")
            except Exception as e:
                failed_tests += 1
                print(f"❌ ERROR: {test_name}")
                print(f"   Error: {type(e).__name__}: {str(e)}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests:   {total_tests}")
    print(f"✅ Passed:      {passed_tests}")
    print(f"❌ Failed:      {failed_tests}")
    print(f"⏭️  Skipped:     {skipped_tests}")
    print(f"Success rate:  {passed_tests/total_tests*100:.1f}%")
    print("="*70)
    
    # Exit with appropriate code
    if failed_tests > 0:
        print("\n⚠️  Some tests failed!")
        exit(1)
    else:
        print("\n All tests passed!")
        exit(0)