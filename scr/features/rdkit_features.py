#Compute RDKIT molecular descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

def compute_rdkit_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),# Molecular Weight
        'NumAromaticRings': Descriptors.NumAromaticRings(mol), # Number of Aromatic Rings
        'LogP': Descriptors.MolLogP(mol),  # Octanol-Water Partition Coefficient
        'NumHDonors': Descriptors.NumHDonors(mol), # Number of Hydrogen Bond Donors
        'NumHAcceptors': Descriptors.NumHAcceptors(mol), # Number of Hydrogen Bond Acceptors
        'TPSA': Descriptors.TPSA(mol), # Topological Polar Surface Area
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol) # Number of Rotatable Bonds

    }
    return descriptors

def compute_features(df: pd.DataFrame, smiles_col: str ='SMILES') -> pd.DataFrame:
    features = df[smiles_col].apply(compute_rdkit_descriptors)
    features_df = pd.DataFrame(features.tolist())
    return pd.concat([df.reset_index(drop=True), features_df], axis=1)