import re
import pandas as pd
from pubchempy import get_compounds, get_synonyms


#Load FIG
def load_ifra_fig(file_path = '/files/datascience_project_EG/data/raw_ifra-fig.csv') -> pd.DataFrame:
    raw_fig = pd.read_csv(file_path, sep=';')
    raw_fig.head()
    print(raw_fig.columns)
    print(f"FIG dataset contains {len(raw_fig)} molecules.")
    print(raw_fig.head())
    return raw_fig


#CAS Registry Numbers are not officially supported by PubChem, 
# but they are often present in the synonyms associated with a compound. 
# Therefore it is straightforward to retrieve them by filtering the synonyms 
# to just those with the CAS Registry Number format:
#le pb c'est que certaines molecules dans pubchem sont formées à partir d'autres donc on peut
#faire face à des doublons ou des absences, le but est donc de lié fig et pubchem par le nom principal et vérifier
#si le cas number est bien le meme entre les deux bases de données

#Retrieve PubChem Data with CAS Numbers of FIG dataset
def fetch_pubchem(fig: pd.DataFrame) -> pd.DataFrame:
    smiles_list = []
    cas_rns = []
    compounds_list = []

    for index, molecule in fig.iterrows(): #on itère sur les elt de fig
        #print(molecule['CAS number'])
        results = get_synonyms(molecule['CAS number'], "name") #PB le nom principal n'est pas forcément le meme entre les deux bases meme si on retrouve celui du fig dans les synonymes de pubchem
        #results correspond à la liste de tous les noms associés à la molécule
        for result in results: 
            #cid = result["CID"] #this is a unique identifier for each compound in PubChem
            for syn in result.get("Synonym", []): #dans le cas de plusieurs syn
               match = re.match(r"(\d{2,7}-\d\d-\d)", syn) 
               if match and match.group(1) == molecule['CAS number']:
                   cas_rns.append(match.group(1))
                   #print(f"Found CAS {match.group(1)} for molecule {molecule['Principal name']}")
                   compounds = get_compounds(syn, 'name') #molecule from pubchem to get SMILES 
                   if compounds:
                        smiles_list.append(compounds[0].connectivity_smiles) #canonical_smiles is deprecated: Use connectivity_smiles instead
                        compounds_list.append(compounds) #toutes les infos de la molécule sont dans compounds ATTENTION ICI
                   else:
                        smiles_list.append(None)
                   break
    chem_df = pd.DataFrame({'CAS number': cas_rns, 'SMILES': smiles_list, 'Compounds': compounds_list})
    print(f"Retrieved {len(chem_df)} molecules from PubChem.")
    chem_df.head()
    return chem_df

#Simplified Molecular Input Line Entry System (SMILES)

#merge FIG dataset with pubchem SMILES strings and other info
def merge_smiles(df_fig: pd.DataFrame, df_pubchem: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(df_fig, df_pubchem, on='CAS number', how='left')
    return merged_df


if __name__ == "__main__":
    raw_fig = load_ifra_fig()
    pubchem_df = fetch_pubchem(raw_fig)
    fig_with_pubchem = merge_smiles(raw_fig, pubchem_df)
    fig_with_pubchem.to_csv('/files/datascience_project_EG/data/fig_with_pubchem.csv', index=False) #save the merged dataset to a csv file
