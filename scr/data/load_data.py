import pandas as pd
from pubchempy import get_compounds

#Retrieve PubChem Data
def fetch_pubchem(smiles_list: list[str]) -> pd.DataFrame:
    """
        Fetch chemical properties from PubChem API.
        Args:
            smiles_list (list[str]): List of SMILES strings.
        Returns:
            pd.DataFrame: DataFrame with additional molecular info.
        """
    #PubChem API query

    pubchem_data = []
    for cas_number in smiles_list:
        try:
            compound = get_compounds(cas_number, 'name')[0]
            smiles = compound.canonical_smiles
            molecular_formula = compound.molecular_formula
            exact_mass = compound.exact_mass
            inchi_key = compound.inchi_key
            print(f"CAS Number: {cas_number}, SMILES: {smiles}, Molecular Formula: {molecular_formula}, Exact Mass: {exact_mass}, InChIKey: {inchi_key}")
            pubchem_data.append({
                'CAS_Number': cas_number,
                'SMILES': smiles,
                'Molecular_Formula': molecular_formula,
                'Exact_Mass': exact_mass,
                'InChIKey': inchi_key
            })

        except Exception as e:
            print(f"Error fetching data for CAS number {cas_number}: {e}")

    pubchem_df = pd.DataFrame(pubchem_data)
    return pubchem_df


#Load FIG
import pandas as pd

def load_ifra_fig(file_path = 'raw_ifra-fig.csv') -> pd.DataFrame:
    raw_fig = pd.read_csv(file_path, sep=';')
    raw_fig.head()
    return raw_fig

#merge FIG dataset with pubchem SMILES strings
def merge_smiles(df_fig: pd.DataFrame, df_pubchem: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(df_fig, df_pubchem, on='CAS_Number', how='left')
    return merged_df