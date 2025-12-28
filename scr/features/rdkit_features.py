"""
Compute RDKit molecular descriptors for fragrance molecules
Enhanced version with error handling and extended features
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def compute_rdkit_descriptors(smiles: str, extended: bool = True) -> Dict[str, float]:
    """
    Compute molecular descriptors from SMILES string
    
    Args:
        smiles: SMILES string representation of molecule
        extended: If True, compute additional descriptors
        
    Returns:
        Dictionary of descriptor names and values
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        # Return empty dict or NaN values for invalid SMILES
        return _get_empty_descriptors(extended)
    
    # Core descriptors (always computed)
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),                    # Molecular Weight
        'LogP': Descriptors.MolLogP(mol),                   # Lipophilicity
        'TPSA': Descriptors.TPSA(mol),                      # Topological Polar Surface Area
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),  # Aromatic rings
        'NumHDonors': Descriptors.NumHDonors(mol),          # H-bond donors
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),    # H-bond acceptors
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),  # Flexibility
    }
    
    # Extended descriptors (optional - more features for better predictions)
    if extended:
        descriptors.update({
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),      # Non-carbon atoms
            'NumRings': Descriptors.RingCount(mol),                 # Total rings
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),      # Non-hydrogen atoms
            'FractionCsp3': Descriptors.FractionCSP3(mol),          # Sp3 carbon fraction
            'MolMR': Crippen.MolMR(mol),                            # Molar Refractivity
            'BalabanJ': Descriptors.BalabanJ(mol),                  # Molecular complexity
            'BertzCT': Descriptors.BertzCT(mol),                    # Complexity metric
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        })
    
    return descriptors


def _get_empty_descriptors(extended: bool = True) -> Dict[str, float]:
    """
    Get dictionary with NaN values for invalid molecules
    
    Args:
        extended: Whether to include extended descriptors
        
    Returns:
        Dictionary with NaN values
    """
    descriptors = {
        'MolWt': np.nan,
        'LogP': np.nan,
        'TPSA': np.nan,
        'NumAromaticRings': np.nan,
        'NumHDonors': np.nan,
        'NumHAcceptors': np.nan,
        'NumRotatableBonds': np.nan,
    }
    
    if extended:
        descriptors.update({
            'NumHeteroatoms': np.nan,
            'NumRings': np.nan,
            'NumSaturatedRings': np.nan,
            'NumAliphaticRings': np.nan,
            'HeavyAtomCount': np.nan,
            'FractionCsp3': np.nan,
            'MolMR': np.nan,
            'BalabanJ': np.nan,
            'BertzCT': np.nan,
            'NumValenceElectrons': np.nan,
        })
    
    return descriptors


def compute_features(df: pd.DataFrame, 
                     smiles_col: str = 'SMILES',
                     extended: bool = True,
                     drop_smiles: bool = False) -> pd.DataFrame:
    """
    Compute RDKit features for all molecules in DataFrame
    
    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of column containing SMILES strings
        extended: Compute extended descriptor set
        drop_smiles: Whether to drop the SMILES column after processing
        
    Returns:
        DataFrame with added descriptor columns
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame")
    
    print(f"Computing RDKit descriptors for {len(df)} molecules...")
    print(f"Extended features: {extended}")
    
    # Compute descriptors for all molecules
    features = df[smiles_col].apply(lambda x: compute_rdkit_descriptors(x, extended))
    features_df = pd.DataFrame(features.tolist())
    
    # Report statistics
    n_invalid = features_df.isnull().all(axis=1).sum()
    if n_invalid > 0:
        print(f"⚠ Warning: {n_invalid} molecules have invalid SMILES")
    
    # Combine with original dataframe
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    # Optionally drop SMILES column (not needed for modeling)
    if drop_smiles:
        result_df = result_df.drop(columns=[smiles_col])
    
    print(f"✓ Added {len(features_df.columns)} descriptor columns")
    print(f"  Columns: {list(features_df.columns)}")
    
    return result_df


def get_descriptor_info() -> pd.DataFrame:
    """
    Get information about available descriptors
    
    Returns:
        DataFrame with descriptor names and descriptions
    """
    info = {
        'Descriptor': [
            'MolWt', 'LogP', 'TPSA', 'NumAromaticRings', 'NumHDonors',
            'NumHAcceptors', 'NumRotatableBonds', 'NumHeteroatoms', 'NumRings',
            'NumSaturatedRings', 'NumAliphaticRings', 'HeavyAtomCount',
            'FractionCsp3', 'MolMR', 'BalabanJ', 'BertzCT', 'NumValenceElectrons'
        ],
        'Description': [
            'Molecular Weight',
            'Lipophilicity (octanol-water partition)',
            'Topological Polar Surface Area',
            'Number of Aromatic Rings',
            'Number of Hydrogen Bond Donors',
            'Number of Hydrogen Bond Acceptors',
            'Number of Rotatable Bonds',
            'Number of Heteroatoms (non-carbon)',
            'Total Number of Rings',
            'Number of Saturated Rings',
            'Number of Aliphatic Rings',
            'Number of Heavy Atoms (non-hydrogen)',
            'Fraction of sp3 Carbons',
            'Molar Refractivity',
            'Balaban J Index (molecular complexity)',
            'Bertz Complexity Index',
            'Number of Valence Electrons'
        ],
        'Relevance to Odor': [
            'Size influences volatility',
            'Affects membrane permeability',
            'Polarity affects receptor binding',
            'Shape and rigidity of molecule',
            'Hydrogen bonding capability',
            'Hydrogen bonding capability',
            'Molecular flexibility',
            'Chemical diversity',
            'Structural complexity',
            'Rigidity vs flexibility',
            'Structural features',
            'Molecular size',
            'Saturated vs unsaturated',
            'Volume and polarizability',
            'Structural uniqueness',
            'Overall complexity',
            'Electronic properties'
        ]
    }
    
    return pd.DataFrame(info)


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("RDKit Molecular Descriptors - Demo")
    print("=" * 70)
    
    # Example molecules
    test_molecules = {
        'Linalool (Floral)': 'CC(C)=CCCC(C)(O)C=C',
        'Limonene (Citrus)': 'CC1=CCC(CC1)C(=C)C',
        'Vanillin (Oriental)': 'COC1=C(C=CC(=C1)C=O)O',
        'Eugenol (Spicy)': 'COC1=C(C=CC(=C1)CC=C)O'
    }
    
    # Compute descriptors for each
    print("\nExample Descriptor Calculations:\n")
    for name, smiles in test_molecules.items():
        desc = compute_rdkit_descriptors(smiles, extended=False)
        print(f"{name}:")
        print(f"  SMILES: {smiles}")
        for key, value in desc.items():
            print(f"  {key:20s}: {value:.2f}")
        print()
    
    # Show descriptor info
    print("\n" + "=" * 70)
    print("Available Descriptors and Their Relevance to Odor Prediction")
    print("=" * 70)
    info_df = get_descriptor_info()
    print(info_df.to_string(index=False))
    
    # Test on DataFrame
    print("\n" + "=" * 70)
    print("Testing on DataFrame")
    print("=" * 70)
    test_df = pd.DataFrame({
        'name': list(test_molecules.keys()),
        'SMILES': list(test_molecules.values())
    })
    
    result = compute_features(test_df, extended=True)
    print(f"\nResult shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(f"\nFirst row descriptors:")
    print(result.iloc[0][['MolWt', 'LogP', 'TPSA', 'NumAromaticRings']])