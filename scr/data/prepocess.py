import pandas as pd
import string
#standardize scent family names

def normalize_scent(merged_df: pd.DataFrame) -> pd.DataFrame:
    #verifier les noms des colonnes apres la fusion
    merged_df['Principal name'] = merged_df['Principal name'].str.lower().str.strip()
    return merged_df



#revoir ce bloc ---------------------------------------------------------------
def map_scent_family(scent: str) -> str:
    if pd.isna(scent):
        return None
    scent = scent.lower().translate(str.maketrans('', '', string.punctuation)) 

if __name__ == "__main__":
    scent_mappings = {
        'Acidic': 'acidic',
        'Aldehydic': 'aldehydic',
        'Amber': 'amber',
        'Animal Like': 'animal like',
        'Anisic': 'anisic',
        'Aromatic': 'aromatic',
        'Balsamic': 'balsamic',
        'Camphoraceous': 'camphoraceous',
        'Citrus': 'citrus',
        'Earthy': 'earthy',
        'Floral': 'floral',
        'Food Like': 'food like',
        'Fruity': 'fruity',
        'Gourmand': 'gourmand',
        'Green': 'green',
        'Herbal': 'herbal',
        'Honey': 'honey',
        'Marine': 'marine',
        'Minty': 'minty',
        'Musk Like': 'musk like',
        'Ozonic': 'ozonic',
        'Powdery': 'powdery',
        'Smoky': 'smoky',
        'Spicy': 'spicy',
        'Sulfurous': 'sulfurous',
        'Tobacco Like': 'tobacco like',
        'Woody': 'woody'
    }
    for key in scent_mappings:
        if key in scent:
            return scent_mappings[key]
    return 'other'