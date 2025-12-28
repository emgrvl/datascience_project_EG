"""
DeepScent Main Pipeline
Complete workflow from raw data to trained model
"""
import os
import sys
from pathlib import Path
import pandas as pd
from scr.data.load_data import load_ifra_fig, add_smiles_to_fig
from scr.data.preprocess import preprocess_ifra_data, filter_by_smiles

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        '/files/datascience_project_EG/data',
        '/files/datascience_project_EG/results/models',
        '/files/datascience_project_EG/results/metrics',
        '/files/datascience_project_EG/results/figures'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories set up")


def run_pipeline(fetch_smiles: bool = False, use_sample: bool = False):
    """
    Run the complete DeepScent pipeline
    
    Args:
        fetch_smiles: Whether to fetch SMILES from PubChem (slow!)
        use_sample: Use only first 100 molecules for testing
    """
    print("\n" + "="*70)
    print(" "*20 + "DEEPSCENT PIPELINE")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # STEP 1: Load raw data
    print("\n" + "─"*70)
    print("STEP 1: Loading raw IFRA-FIG data")
    print("─"*70)
    df_raw = load_ifra_fig('/files/datascience_project_EG/data/raw_ifra-fig.csv')
    
    if use_sample:
        print(f"\n✓ Using sample of first 100 molecules for testing")
        df_raw = df_raw.head(100)
    
    # STEP 2: Preprocess
    print("\n" + "─"*70)
    print("STEP 2: Preprocessing data")
    print("─"*70)
    df_preprocessed = preprocess_ifra_data(df_raw)
    
    # Save preprocessed data
    output_path = '/files/datascience_project_EG/data/ifra_preprocessed.csv'
    df_preprocessed.to_csv(output_path, index=False)
    print(f"\n✓ Saved preprocessed data: {output_path}")
    
    # STEP 3: Fetch SMILES (optional - very slow!)
    if fetch_smiles:
        print("\n" + "─"*70)
        print("STEP 3: Fetching SMILES from PubChem")
        print("─"*70)
        print("⚠ This will take a LONG time (~1-2 hours for full dataset)")
        print("We can start with use_sample=True")
        
        df_with_smiles = add_smiles_to_fig(df_preprocessed)
        
        # Save with SMILES
        smiles_path = '/files/datascience_project_EG/data/ifra_with_smiles_pipeline.csv'
        df_with_smiles.to_csv(smiles_path, index=False)
        print(f"\n✓ Saved data with SMILES: {smiles_path}")
        
        # Filter valid molecules
        df_final = filter_by_smiles(df_with_smiles, min_family_size=5)
    else:
        print("\n" + "─"*70)
        print("STEP 3: Skipping SMILES fetching (use fetch_smiles=True to enable)")
        print("─"*70)
        print("💡 For next steps, you'll need SMILES strings")
        print("   Run: python pipeline.py --fetch-smiles")
        df_final = df_preprocessed
    
    # STEP 4: Generate analysis report
    print("\n" + "─"*70)
    print("STEP 4: Generating data analysis report")
    print("─"*70)
    
    report = generate_data_report(df_final, has_smiles=fetch_smiles)
    
    report_path = '/files/datascience_project_EG/results/data_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Saved analysis report: {report_path}")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  • Preprocessed data: /files/datascience_project_EG/data/ifra_preprocessed.csv")
    if fetch_smiles:
        print(f"  • Data with SMILES: /files/datascience_project_EG/data/ifra_with_smiles.csv")
    print(f"  • Analysis report: results/data_analysis_report.txt")
    
    print(f"\n📊 Final dataset: {len(df_final)} molecules")
    
    if not fetch_smiles:
        print("\n" + "─"*70)
        print("NEXT STEPS:")
        print("─"*70)
        print("1. Run with SMILES fetching:")
        print("   python pipeline.py --fetch-smiles")
        print("\n2. Or provide SMILES data manually and skip to feature generation")
        print("   python -m src.features.rdkit_features")
    
    return df_final


def generate_data_report(df: pd.DataFrame, has_smiles: bool = False) -> str:
    """Generate a comprehensive data analysis report"""
    
    report = []
    report.append("="*70)
    report.append(" "*20 + "DEEPSCENT DATA ANALYSIS REPORT")
    report.append("="*70)
    report.append("")
    
    # Basic statistics
    report.append("DATASET OVERVIEW")
    report.append("-"*70)
    report.append(f"Total molecules: {len(df)}")
    report.append(f"Columns: {list(df.columns)}")
    report.append("")
    
    # Fragrance families
    report.append("FRAGRANCE FAMILY DISTRIBUTION")
    report.append("-"*70)
    family_counts = df['fragrance_family'].value_counts()
    for family, count in family_counts.items():
        pct = count / len(df) * 100
        report.append(f"  {family:25s}: {count:4d} ({pct:5.1f}%)")
    report.append("")
    
    # SMILES availability
    if has_smiles and 'SMILES' in df.columns:
        smiles_count = df['SMILES'].notna().sum()
        smiles_pct = smiles_count / len(df) * 100
        report.append("SMILES AVAILABILITY")
        report.append("-"*70)
        report.append(f"Molecules with SMILES: {smiles_count}/{len(df)} ({smiles_pct:.1f}%)")
        report.append("")
    
    # Missing values
    report.append("MISSING VALUES")
    report.append("-"*70)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            pct = count / len(df) * 100
            report.append(f"  {col:25s}: {count:4d} ({pct:5.1f}%)")
    else:
        report.append("  No missing values")
    report.append("")
    
    # Class balance analysis
    report.append("CLASS BALANCE ANALYSIS")
    report.append("-"*70)
    min_count = family_counts.min()
    max_count = family_counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    report.append(f"Smallest class: {min_count} molecules")
    report.append(f"Largest class: {max_count} molecules")
    report.append(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        report.append("\n⚠ SEVERE class imbalance detected!")
        report.append("  Recommendation: Use SMOTE or class weights in training")
    elif imbalance_ratio > 5:
        report.append("\n⚠ Moderate class imbalance detected")
        report.append("  Recommendation: Consider class weights")
    else:
        report.append("\n✓ Classes are relatively balanced")
    report.append("")
    
    report.append("="*70)
    
    return "\n".join(report)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepScent Pipeline')
    parser.add_argument('--fetch-smiles', action='store_true',
                       help='Fetch SMILES from PubChem (slow!)')
    parser.add_argument('--sample', action='store_true',
                       help='Use only first 100 molecules for testing')
    
    args = parser.parse_args()
    
    run_pipeline(fetch_smiles=args.fetch_smiles, use_sample=args.sample)
