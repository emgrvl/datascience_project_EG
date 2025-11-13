#Single molecule prediction

import joblib
import numpy as np
from .features.rdkit_features import compute_descriptors

def predict_smiles(model_path: str, smiles: str) -> dict:
    """
    Predict odor family from SMILES.

    Args:
        model_path (str)
        smiles (str)

    Returns:
        dict
    """
    model = joblib.load(model_path)
    features = compute_descriptors(smiles)
    if not features:
        return {'error': 'Invalid SMILES'}
    X = np.array([list(features.values())])
    pred = model.predict(X)[0]
    return {'predicted_family': pred}