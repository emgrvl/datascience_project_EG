"""
Data loading utilities for DeepScent project
Improved version with proper error handling and SMILES fetching
"""
import re
import pandas as pd
import time
from typing import Optional, List

try:
    from pubchempy import get_compounds, Compound
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False
    print("Warning: pubchempy not available. Install with: pip install pubchempy")


def load_ifra_fig(file_path: str = '/files/datascience_project_EG/data/raw_ifra-fig.csv') -> pd.DataFrame:
    """
    Load IFRA FIG dataset
    
    Args:
        file_path: Path to the raw IFRA-FIG CSV file
        
    Returns:
        DataFrame with fragrance molecules
    """
    # The file uses semicolon as separator
    raw_fig = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')
    
    # Clean column names (remove BOM if present)
    raw_fig.columns = raw_fig.columns.str.strip()
    
    print(f"✓ Loaded IFRA-FIG dataset: {len(raw_fig)} molecules")
    print(f"✓ Columns: {list(raw_fig.columns)}")
    print(f"\nFirst few rows:")
    print(raw_fig.head())
    
    return raw_fig


def fetch_smiles_from_pubchem(cas_number: str, max_retries: int = 3) -> Optional[str]:
    """
    Fetch SMILES string from PubChem using CAS number
    
    Args:
        cas_number: CAS Registry Number
        max_retries: Number of retry attempts
        
    Returns:
        SMILES string or None if not found
    """
    if not PUBCHEM_AVAILABLE:
        return None
        
    for attempt in range(max_retries):
        try:
            # Search by name (CAS number)
            compounds = get_compounds(cas_number, 'name')
            
            if compounds:
                # Return canonical SMILES of first result
                return compounds[0].canonical_smiles
            
            # Small delay to avoid rate limiting
            time.sleep(0.2)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            else:
                print(f"  Failed to fetch {cas_number}: {str(e)}")
                return None
    
    return None


def add_smiles_to_fig(fig_df: pd.DataFrame, batch_size: int = 100, save_progress: bool = True) -> pd.DataFrame:
    """
    Add SMILES strings to FIG dataset by querying PubChem
    
    Args:
        fig_df: IFRA-FIG DataFrame
        batch_size: Save progress after this many molecules
        save_progress: Whether to save intermediate results
        
    Returns:
        DataFrame with added SMILES column
    """
    if not PUBCHEM_AVAILABLE:
        print("⚠ PubChem library not available. Skipping SMILES fetching.")
        fig_df['SMILES'] = None
        return fig_df
    
    print(f"\n🔍 Fetching SMILES from PubChem for {len(fig_df)} molecules...")
    print("This may take a while. Progress will be saved periodically.")
    
    smiles_list = [None]*len(fig_df)
    
    for idx, row in fig_df.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(fig_df)} molecules...", end='\r')
        
        cas = row['CAS number']
        smiles = fetch_smiles_from_pubchem(cas)
        smiles_list.append(smiles)
        
        # Save progress periodically
        if save_progress and (idx + 1) % batch_size == 0:
            temp_df = fig_df.iloc[:idx+1].copy()
            temp_df['SMILES'] = smiles_list
            temp_df.to_csv('/files/datascience_project_EG/data/fig_with_smiles_progress.csv', index=False)
            print(f"\n  💾 Progress saved: {idx+1}/{len(fig_df)} molecules")
    
    fig_df['SMILES'] = smiles_list
    
    # Statistics
    found = sum(1 for s in smiles_list if s is not None)
    print(f"\n✓ SMILES found for {found}/{len(fig_df)} molecules ({found/len(fig_df)*100:.1f}%)")
    
    return fig_df


def load_processed_data(filepath: str = '/files/datascience_project_EG/data/processed_molecules.csv') -> pd.DataFrame:
    """
    Load processed molecule dataset
    
    Args:
        filepath: Path to processed data file
        
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded processed dataset: {len(df)} molecules")
    return df


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("DeepScent - Data Loading Demo")
    print("=" * 60)
    
    # Load IFRA data
    fig_df = load_ifra_fig()
    
    #fetch SMILES 
    fig_with_smiles = add_smiles_to_fig(fig_df)  
    fig_with_smiles.to_csv('/files/datascience_project_EG/data/fig_with_smiles_sample.csv', index=False) #got 2149/3271 molecules with SMILES
