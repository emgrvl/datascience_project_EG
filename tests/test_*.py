import pytest
import pandas as pd
from src.data.load_data import fetch_pubchem, load_ifra_fig, merge_smiles

#test data loading/fetching 
def test_fetch_pubchem():
    smiles_list = ['50-00-0', '64-17-5']  # CAS numbers for formaldehyde and ethanol
    df = fetch_pubchem(smiles_list)
    assert 'SMILES' in df.columns
    assert len(df) == 2

def test_load_ifra():
    df = load_ifra("data/sample_fig.csv")
    assert "IFRA_Family" in df.columns
    assert len(df) > 0

def test_merge_smiles():
    df1 = pd.DataFrame({'CAS': [1,2], 'Name':['A','B'], 'IFRA_Family':['floral','woody']})
    df2 = pd.DataFrame({'CAS':[1,2], 'SMILES':['C','CC']})
    df = merge_smiles(df1, df2)
    assert "SMILES" in df.columns
    assert len(df) == 2

#test features
from src.features.rdkit_features import compute_rdkit_descriptors
from src.features.fingerprinting import compute_morgan_fp
import numpy as np

def test_compute_rdkit_descriptors():
    desc = compute_rdkit_descriptors("CCO")
    assert "MolWt" in desc

def test_compute_morgan_fp():
    fp = compute_morgan_fp("CCO")
    assert isinstance(fp, np.ndarray)
    assert fp.shape[0] == 2048

#test model pipeline
from src.models.train import train_model
from sklearn.ensemble import RandomForestClassifier

def test_train_model():
    X = pd.DataFrame(np.random.rand(10,5), columns=list('ABCDE'))
    y = pd.Series(["floral","woody","citrus","floral","green","woody","amber","spicy","citrus","floral"])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier) or hasattr(model, "predict")