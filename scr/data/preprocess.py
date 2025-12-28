"""
Data preprocessing for fragrance molecules
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


# Official IFRA fragrance family taxonomy (27 families)
IFRA_FAMILIES = {
    'acidic', 'aldehydic', 'amber', 'animal like', 'anisic', 'aromatic',
    'balsamic', 'camphoraceous', 'citrus', 'earthy', 'floral', 'food like',
    'fruity', 'gourmand', 'green', 'herbal', 'honey', 'marine', 'minty',
    'musk like', 'ozonic', 'powdery', 'smoky', 'spicy', 'sulfurous',
    'tobacco like', 'woody'
}


def normalize_text(text: str) -> str:
    """
    Normalize text: lowercase, strip whitespace
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if pd.isna(text):
        return None
    return str(text).lower().strip()


def standardize_fragrance_family(descriptor: str) -> Optional[str]:
    """
    Map descriptor to standardized IFRA fragrance family
    
    Args:
        descriptor: Raw fragrance descriptor
        
    Returns:
        Standardized family name or None
    """
    if pd.isna(descriptor):
        return None
    
    normalized = normalize_text(descriptor)
    
    # Direct match
    if normalized in IFRA_FAMILIES:
        return normalized
    
    # Handle variations
    mappings = {
        'animal': 'animal like',
        'musk': 'musk like',
        'musky': 'musk like',
        'food': 'food like',
        'tobacco': 'tobacco like',
        'sulphur': 'sulfurous',
        'sulfur': 'sulfurous'
    }
    
    for key, value in mappings.items():
        if key in normalized:
            return value
    
    # If no match, keep original
    return normalized


def create_primary_label(row: pd.Series) -> str:
    """
    Create primary fragrance family label from descriptors
    Uses Primary descriptor, falls back to Descriptor 2
    
    Args:
        row: DataFrame row with descriptor columns
        
    Returns:
        Primary fragrance family
    """
    # Try Primary descriptor first
    primary = standardize_fragrance_family(row.get('Primary descriptor'))
    if primary and primary in IFRA_FAMILIES:
        return primary
    
    # Fall back to Descriptor 2
    desc2 = standardize_fragrance_family(row.get('Descriptor 2'))
    if desc2 and desc2 in IFRA_FAMILIES:
        return desc2
    
    # Fall back to Descriptor 3
    desc3 = standardize_fragrance_family(row.get('Descriptor 3'))
    if desc3 and desc3 in IFRA_FAMILIES:
        return desc3
    
    # Default if none match
    return 'other'


def preprocess_ifra_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive preprocessing of IFRA-FIG data
    
    Args:
        df: Raw IFRA DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    print("\n" + "="*60)
    print("PREPROCESSING IFRA-FIG DATA")
    print("="*60)
    
    df_clean = df.copy()
    original_count = len(df_clean)
    
    # 1. Clean column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # 2. Standardize text columns
    print("\n1. Standardizing text columns...")
    text_cols = ['Principal name', 'Primary descriptor', 'Descriptor 2', 'Descriptor 3']
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(normalize_text)
    
    # 3. Create standardized fragrance family
    print("2. Creating standardized fragrance families...")
    df_clean['fragrance_family'] = df_clean.apply(create_primary_label, axis=1)
    
    # 4. Create multi-label representation (all three descriptors)
    df_clean['all_descriptors'] = df_clean.apply(
        lambda row: [
            standardize_fragrance_family(row.get('Primary descriptor')),
            standardize_fragrance_family(row.get('Descriptor 2')),
            standardize_fragrance_family(row.get('Descriptor 3'))
        ],
        axis=1
    )
    
    # Remove None values from all_descriptors
    df_clean['all_descriptors'] = df_clean['all_descriptors'].apply(
        lambda x: [d for d in x if d is not None]
    )
    
    # 5. Remove rows without CAS number
    print("3. Removing rows without CAS numbers...")
    before = len(df_clean)
    df_clean = df_clean[df_clean['CAS number'].notna()]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} rows without CAS numbers")
    
    # 6. Remove duplicates based on CAS number
    print("4. Removing duplicate CAS numbers...")
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['CAS number'], keep='first')
    removed = before - len(df_clean)
    if removed > 0:
        print(f"   Removed {removed} duplicate entries")
    
    # 7. Statistics
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Original molecules: {original_count}")
    print(f"Final molecules: {len(df_clean)}")
    print(f"Removed: {original_count - len(df_clean)} ({(original_count - len(df_clean))/original_count*100:.1f}%)")
    
    # Fragrance family distribution
    print("\nFragrance Family Distribution:")
    family_counts = df_clean['fragrance_family'].value_counts()
    for family, count in family_counts.head(15).items():
        print(f"  {family:20s}: {count:4d} ({count/len(df_clean)*100:5.1f}%)")
    
    if len(family_counts) > 15:
        print(f"  ... and {len(family_counts) - 15} more families")
    
    return df_clean


def filter_by_smiles(df: pd.DataFrame, min_family_size: int = 10) -> pd.DataFrame:
    """
    Filter dataset to only include molecules with valid SMILES
    and fragrance families with sufficient samples
    
    Args:
        df: DataFrame with SMILES column
        min_family_size: Minimum number of molecules per family
        
    Returns:
        Filtered DataFrame
    """
    print("\n" + "="*60)
    print("FILTERING BY SMILES AND FAMILY SIZE")
    print("="*60)
    
    original_count = len(df)
    
    # Filter out rows without SMILES
    df_filtered = df[df['SMILES'].notna()].copy()
    smiles_removed = original_count - len(df_filtered)
    print(f"Removed {smiles_removed} molecules without SMILES")
    
    # Filter families with too few samples
    family_counts = df_filtered['fragrance_family'].value_counts()
    valid_families = family_counts[family_counts >= min_family_size].index
    
    df_filtered = df_filtered[df_filtered['fragrance_family'].isin(valid_families)]
    family_removed = original_count - smiles_removed - len(df_filtered)
    print(f"Removed {family_removed} molecules from families with <{min_family_size} samples")
    
    print(f"\nFinal dataset: {len(df_filtered)} molecules across {len(valid_families)} families")
    
    return df_filtered


if __name__ == "__main__":
    # Example usage
    from load_data import load_ifra_fig
    
    # Load data
    df = load_ifra_fig()
    
    # Preprocess
    df_clean = preprocess_ifra_data(df)
    
    # Save
    df_clean.to_csv('/files/datascience_project_EG/data/ifra_preprocessed.csv', index=False)
    print("\n✓ Saved preprocessed data to:/files/datascience_project_EG/data/ifra_preprocessed.csv")
