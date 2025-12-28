"""
Feature Generation Pipeline for DeepScent
Generates RDKit descriptors and molecular fingerprints
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# FIXED: Correct import names and paths
sys.path.insert(0, '/files/datascience_project_EG')

from scr.features.rdkit_features import compute_features
from scr.features.fingerprinting import add_fingerprints


def generate_all_features(input_path: str,
                          output_path: str,
                          use_fingerprints: bool = True,
                          use_extended_descriptors: bool = True,
                          fingerprint_type: str = 'morgan',
                          n_bits: int = 2048):
    """
    Generate all molecular features for the dataset
    
    Args:
        input_path: Path to CSV with SMILES column
        output_path: Where to save features
        use_fingerprints: Whether to compute fingerprints
        use_extended_descriptors: Use extended RDKit descriptor set
        fingerprint_type: 'morgan' or 'maccs'
        n_bits: Number of bits for Morgan fingerprints
    """
    print("\n" + "="*70)
    print("DEEPSCENT FEATURE GENERATION PIPELINE")
    print("="*70)
    
    # Load data
    print(f"\n1. Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} molecules")
    
    # Check for SMILES column
    if 'SMILES' not in df.columns:
        raise ValueError("DataFrame must have 'SMILES' column. Run pipeline.py --fetch-smiles first!")
    
    # Filter out molecules without SMILES
    df_valid = df[df['SMILES'].notna()].copy()
    n_removed = len(df) - len(df_valid)
    
    if n_removed > 0:
        print(f"   ⚠ Filtered out {n_removed} molecules without SMILES")
        print(f"   Working with {len(df_valid)} molecules")
    
    # Generate RDKit descriptors
    print(f"\n2. Computing RDKit molecular descriptors")
    print(f"   Extended features: {use_extended_descriptors}")
    
    # FIXED: Use correct function name
    df_with_descriptors = compute_features(
        df_valid,
        smiles_col='SMILES',
        extended=use_extended_descriptors,
        drop_smiles=False
    )
    
    # Generate fingerprints (optional)
    if use_fingerprints:
        print(f"\n3. Computing molecular fingerprints")
        print(f"   Type: {fingerprint_type.upper()}")
        df_final = add_fingerprints(
            df_with_descriptors,
            smiles_col='SMILES',
            fp_type=fingerprint_type,
            n_bits=n_bits
        )
    else:
        print(f"\n3. Skipping fingerprints (use_fingerprints=False)")
        df_final = df_with_descriptors
    
    # Remove rows with missing descriptors
    print(f"\n4. Handling missing values")
    before = len(df_final)
    
    # Get descriptor columns (exclude metadata)
    metadata_cols = ['CAS number', 'Principal name', 'Primary descriptor', 
                     'Descriptor 2', 'Descriptor 3', 'fragrance_family', 
                     'all_descriptors', 'SMILES']
    descriptor_cols = [col for col in df_final.columns if col not in metadata_cols]
    
    # Remove rows with any NaN in descriptors
    df_final = df_final.dropna(subset=descriptor_cols)
    n_removed = before - len(df_final)
    
    if n_removed > 0:
        print(f"   Removed {n_removed} molecules with invalid descriptors")
    print(f"   Final dataset: {len(df_final)} molecules")
    
    # Save results
    print(f"\n5. Saving features to {output_path}")
    df_final.to_csv(output_path, index=False)
    
    # Generate summary
    print("\n" + "="*70)
    print("FEATURE GENERATION COMPLETE!")
    print("="*70)
    
    summary = {
        'input_molecules': len(df),
        'molecules_with_smiles': len(df_valid),
        'final_molecules': len(df_final),
        'total_features': len(descriptor_cols),
        'rdkit_descriptors': sum(1 for col in descriptor_cols if not col.startswith(('MorganFP', 'MACCS'))),
        'fingerprint_bits': sum(1 for col in descriptor_cols if col.startswith(('MorganFP', 'MACCS')))
    }
    
    print(f"\nDataset Summary:")
    print(f"  Input molecules:     {summary['input_molecules']}")
    print(f"  With SMILES:         {summary['molecules_with_smiles']}")
    print(f"  Final (valid):       {summary['final_molecules']}")
    print(f"  Success rate:        {summary['final_molecules']/summary['input_molecules']*100:.1f}%")
    print(f"\nFeature Summary:")
    print(f"  Total features:      {summary['total_features']}")
    print(f"  RDKit descriptors:   {summary['rdkit_descriptors']}")
    print(f"  Fingerprint bits:    {summary['fingerprint_bits']}")
    
    # Show fragrance family distribution
    if 'fragrance_family' in df_final.columns:
        print(f"\nFragrance Family Distribution (after filtering):")
        family_counts = df_final['fragrance_family'].value_counts()
        for family, count in family_counts.head(10).items():
            pct = count / len(df_final) * 100
            print(f"  {family:20s}: {count:4d} ({pct:5.1f}%)")
        
        if len(family_counts) > 10:
            print(f"  ... and {len(family_counts) - 10} more families")
    
    print(f"\n✓ Features saved to: {output_path}")
    print(f"\nNext step: Train models with:")
    print(f"  python train_model.py")
    
    return df_final, summary


def analyze_features(df: pd.DataFrame):
    """
    Generate feature correlation and importance analysis
    
    Args:
        df: DataFrame with features
    """
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)
    
    # Get only RDKit descriptor columns (not fingerprints - too many)
    metadata_cols = ['CAS number', 'Principal name', 'Primary descriptor', 
                     'Descriptor 2', 'Descriptor 3', 'fragrance_family', 
                     'all_descriptors', 'SMILES']
    
    descriptor_cols = [col for col in df.columns 
                      if col not in metadata_cols 
                      and not col.startswith(('MorganFP', 'MACCS'))]
    
    if len(descriptor_cols) == 0:
        print("No RDKit descriptors found to analyze")
        return
    
    print(f"\nAnalyzing {len(descriptor_cols)} RDKit descriptors")
    
    # Compute correlation matrix
    corr_matrix = df[descriptor_cols].corr()
    
    # Find highly correlated features
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr:
        print(f"\n⚠ Found {len(high_corr)} highly correlated feature pairs (|r| > 0.8):")
        for feat1, feat2, corr in high_corr[:10]:
            print(f"  {feat1:20s} <-> {feat2:20s}: {corr:6.3f}")
        if len(high_corr) > 10:
            print(f"  ... and {len(high_corr) - 10} more pairs")
        print("\n  → Consider feature selection or PCA")
    else:
        print("\n✓ No highly correlated features found")
    
    # Basic statistics
    print(f"\nDescriptor Statistics:")
    stats = df[descriptor_cols].describe()
    print(stats)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate molecular features for DeepScent')
    parser.add_argument('--input', type=str, 
                       default='/files/datascience_project_EG/data/ifra_with_smiles.csv',
                       help='Input CSV file with SMILES')
    parser.add_argument('--output', type=str,
                       default='/files/datascience_project_EG/data/molecules_with_features.csv',
                       help='Output CSV file')
    parser.add_argument('--no-fingerprints', action='store_true',
                       help='Skip fingerprint generation (faster)')
    parser.add_argument('--basic-descriptors', action='store_true',
                       help='Use basic descriptor set only')
    parser.add_argument('--fp-type', type=str, default='morgan',
                       choices=['morgan', 'maccs'],
                       help='Fingerprint type')
    parser.add_argument('--n-bits', type=int, default=2048,
                       help='Number of bits for Morgan fingerprints')
    parser.add_argument('--analyze', action='store_true',
                       help='Run feature analysis after generation')
    
    args = parser.parse_args()
    
    # Generate features
    df_features, summary = generate_all_features(
        input_path=args.input,
        output_path=args.output,
        use_fingerprints=not args.no_fingerprints,
        use_extended_descriptors=not args.basic_descriptors,
        fingerprint_type=args.fp_type,
        n_bits=args.n_bits
    )
    
    # Optional analysis
    if args.analyze:
        analyze_features(df_features)