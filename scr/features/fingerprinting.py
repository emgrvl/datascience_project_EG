#Molecular fingerprints

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 

#for one molecule
def compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

#for dataframe
import pandas as pd
def add_fingerprints(df: pd.DataFrame, smiles_col: str ='SMILES', radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    fps = df[smiles_col].apply(lambda x: compute_morgan_fp(x, radius, n_bits))
    fps_df = pd.DataFrame(fps.tolist(), index=df.index)
    fps_df.columns = [f'FP_{i}' for i in range(n_bits)]
    return pd.concat([df.reset_index(drop=True), fps_df], axis=1)