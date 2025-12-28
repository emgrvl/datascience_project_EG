"""
Molecular fingerprints for fragrance molecules
Enhanced version with error handling and multiple fingerprint types
"""
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pandas as pd
from typing import Optional

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute Morgan (circular) fingerprint for a single molecule
    
    Args:
        smiles: SMILES string
        radius: Fingerprint radius (2 = ECFP4)
        n_bits: Number of bits in fingerprint
        
    Returns:
        Binary fingerprint as numpy array
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        # Return zeros for invalid SMILES
        return np.zeros(n_bits, dtype=int)
    
    # Generate Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    
    # Convert to numpy array
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr


def compute_maccs_fp(smiles: str) -> np.ndarray:
    """
    Compute MACCS keys fingerprint (166-bit structural keys)
    
    Args:
        smiles: SMILES string
        
    Returns:
        Binary fingerprint as numpy array
    """
    from rdkit.Chem import MACCSkeys
    
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return np.zeros(166, dtype=int)
    
    # Generate MACCS keys
    fp = MACCSkeys.GenMACCSKeys(mol)
    
    # Convert to numpy array
    arr = np.zeros((166,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr


def add_fingerprints(df: pd.DataFrame, 
                     smiles_col: str = 'SMILES',
                     fp_type: str = 'morgan',
                     radius: int = 2,
                     n_bits: int = 2048) -> pd.DataFrame:
    """
    Add molecular fingerprints to DataFrame
    
    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of SMILES column
        fp_type: Type of fingerprint ('morgan' or 'maccs')
        radius: Radius for Morgan fingerprints
        n_bits: Number of bits for Morgan fingerprints
        
    Returns:
        DataFrame with added fingerprint columns
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame")
    
    print(f"Computing {fp_type.upper()} fingerprints for {len(df)} molecules...")
    
    if fp_type.lower() == 'morgan':
        print(f"  Radius: {radius}, Bits: {n_bits}")
        fps = df[smiles_col].apply(lambda x: compute_morgan_fp(x, radius, n_bits))
        fp_prefix = 'MorganFP'
        n_features = n_bits
    elif fp_type.lower() == 'maccs':
        print(f"  MACCS keys: 166 structural features")
        fps = df[smiles_col].apply(compute_maccs_fp)
        fp_prefix = 'MACCS'
        n_features = 166
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}. Use 'morgan' or 'maccs'")
    
    # Convert to DataFrame
    fps_df = pd.DataFrame(fps.tolist(), index=df.index)
    fps_df.columns = [f'{fp_prefix}_{i}' for i in range(n_features)]
    
    # Combine with original dataframe
    result_df = pd.concat([df.reset_index(drop=True), fps_df], axis=1)
    
    # Statistics
    n_invalid = (fps_df.sum(axis=1) == 0).sum()
    if n_invalid > 0:
        print(f"⚠ Warning: {n_invalid} molecules produced empty fingerprints")
    
    print(f"✓ Added {n_features} fingerprint columns")
    
    return result_df


def calculate_similarity(smiles1: str, smiles2: str, fp_type: str = 'morgan') -> float:
    """
    Calculate Tanimoto similarity between two molecules
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        fp_type: Fingerprint type ('morgan' or 'maccs')
        
    Returns:
        Tanimoto similarity (0-1, where 1 = identical)
    """
    if fp_type.lower() == 'morgan':
        fp1 = compute_morgan_fp(smiles1)
        fp2 = compute_morgan_fp(smiles2)
    elif fp_type.lower() == 'maccs':
        fp1 = compute_maccs_fp(smiles1)
        fp2 = compute_maccs_fp(smiles2)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
    
    # Tanimoto coefficient
    intersection = np.logical_and(fp1, fp2).sum()
    union = np.logical_or(fp1, fp2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def find_similar_molecules(target_smiles: str,
                           df: pd.DataFrame,
                           smiles_col: str = 'SMILES',
                           top_n: int = 5,
                           fp_type: str = 'morgan') -> pd.DataFrame:
    """
    Find most similar molecules in dataset
    
    Args:
        target_smiles: SMILES of query molecule
        df: DataFrame with molecules
        smiles_col: SMILES column name
        top_n: Number of similar molecules to return
        fp_type: Fingerprint type
        
    Returns:
        DataFrame with top N similar molecules and similarity scores
    """
    similarities = df[smiles_col].apply(
        lambda x: calculate_similarity(target_smiles, x, fp_type)
    )
    
    df_with_sim = df.copy()
    df_with_sim['similarity'] = similarities
    
    # Sort by similarity (descending)
    df_sorted = df_with_sim.sort_values('similarity', ascending=False)
    
    return df_sorted.head(top_n)


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("Molecular Fingerprints - Demo")
    print("=" * 70)
    
    # Example molecules
    test_molecules = {
        'Linalool (Floral)': 'CC(C)=CCCC(C)(O)C=C',
        'Geraniol (Floral)': 'CC(C)=CCCC(C)(O)C=C',  # Very similar to Linalool
        'Limonene (Citrus)': 'CC1=CCC(CC1)C(=C)C',
        'Vanillin (Oriental)': 'COC1=C(C=CC(=C1)C=O)O',
    }
    
    # Test Morgan fingerprint
    print("\nMorgan Fingerprint Example:")
    smiles = test_molecules['Linalool (Floral)']
    fp = compute_morgan_fp(smiles)
    print(f"SMILES: {smiles}")
    print(f"Fingerprint shape: {fp.shape}")
    print(f"Number of bits set: {fp.sum()}")
    print(f"First 20 bits: {fp[:20]}")
    
    # Test MACCS fingerprint
    print("\n\nMACCS Keys Example:")
    fp_maccs = compute_maccs_fp(smiles)
    print(f"SMILES: {smiles}")
    print(f"Fingerprint shape: {fp_maccs.shape}")
    print(f"Number of keys set: {fp_maccs.sum()}")
    
    # Test similarity calculation
    print("\n\n" + "=" * 70)
    print("Similarity Calculation")
    print("=" * 70)
    
    smiles_list = list(test_molecules.values())
    names_list = list(test_molecules.keys())
    
    print("\nPairwise Tanimoto Similarities:")
    print(f"{'Molecule 1':<25} {'Molecule 2':<25} {'Similarity':>10}")
    print("-" * 70)
    
    for i, (name1, smiles1) in enumerate(test_molecules.items()):
        for j, (name2, smiles2) in enumerate(test_molecules.items()):
            if i < j:  # Only upper triangle
                sim = calculate_similarity(smiles1, smiles2)
                print(f"{name1:<25} {name2:<25} {sim:>10.3f}")
    
    # Test DataFrame processing
    print("\n\n" + "=" * 70)
    print("DataFrame Processing")
    print("=" * 70)
    
    test_df = pd.DataFrame({
        'name': names_list,
        'SMILES': smiles_list
    })
    
    print("\nOriginal DataFrame:")
    print(test_df)
    
    # Add fingerprints
    result = add_fingerprints(test_df.copy(), fp_type='morgan', n_bits=512)
    print(f"\nAfter adding fingerprints: {result.shape}")
    print(f"Columns: {len(result.columns)} total")
    
    # Test finding similar molecules
    print("\n\n" + "=" * 70)
    print("Finding Similar Molecules")
    print("=" * 70)
    
    target = test_molecules['Linalool (Floral)']
    print(f"\nTarget molecule: Linalool")
    print(f"SMILES: {target}")
    print("\nMost similar molecules:")
    
    similar = find_similar_molecules(target, test_df, top_n=3)
    print(similar[['name', 'similarity']])