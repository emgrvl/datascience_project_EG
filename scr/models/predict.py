"""
Single molecule prediction for DeepScent
Predicts fragrance family from SMILES string
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, '/files/datascience_project_EG')

from scr.features.rdkit_features import compute_rdkit_descriptors
from scr.features.fingerprinting import compute_morgan_fp


def predict_smiles(model_path: str, 
                   smiles: str,
                   use_fingerprints: bool = True,
                   extended_descriptors: bool = True,
                   verbose: bool = True) -> dict:
    """
    Predict fragrance family from SMILES string
    
    Args:
        model_path: Path to trained model (.joblib)
        smiles: SMILES string of molecule
        use_fingerprints: Whether model was trained with fingerprints
        extended_descriptors: Whether model used extended descriptors
        verbose: Print prediction details
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load model
        if verbose:
            print(f"\n🔮 Predicting fragrance family...")
            print(f"   SMILES: {smiles}")
        
        model = joblib.load(model_path)
        
        # Compute molecular descriptors
        descriptors = compute_rdkit_descriptors(smiles, extended=extended_descriptors)
        
        if not descriptors or all(pd.isna(v) for v in descriptors.values()):
            return {
                'error': 'Invalid SMILES or failed to compute descriptors',
                'smiles': smiles
            }
        
        # Build feature vector
        features = list(descriptors.values())
        
        # Add fingerprints if needed
        if use_fingerprints:
            fp = compute_morgan_fp(smiles, n_bits=2048)
            features.extend(fp.tolist())
        
        # Reshape for prediction
        X = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)[0]
            
            # Get top 3 predictions
            classes = model.classes_
            top_indices = np.argsort(probas)[::-1][:3]
            top_predictions = [
                (classes[i], probas[i]) for i in top_indices
            ]
        else:
            probas = None
            top_predictions = [(prediction, 1.0)]
        
        result = {
            'predicted_family': prediction,
            'smiles': smiles,
            'confidence': probas[list(model.classes_).index(prediction)] if probas is not None else None,
            'top_3_predictions': top_predictions,
            'descriptors': descriptors
        }
        
        if verbose:
            print(f"\n   ✓ Prediction: {prediction}")
            if result['confidence']:
                print(f"   Confidence: {result['confidence']:.1%}")
            
            if len(top_predictions) > 1:
                print(f"\n   Top 3 Predictions:")
                for family, prob in top_predictions:
                    print(f"      {family:20s}: {prob:.1%}")
        
        return result
        
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'smiles': smiles
        }


def batch_predict(model_path: str, 
                  smiles_list: list,
                  use_fingerprints: bool = True,
                  extended_descriptors: bool = True) -> pd.DataFrame:
    """
    Predict fragrance families for multiple molecules
    
    Args:
        model_path: Path to trained model
        smiles_list: List of SMILES strings
        use_fingerprints: Whether model was trained with fingerprints
        extended_descriptors: Whether model used extended descriptors
        
    Returns:
        DataFrame with predictions
    """
    print(f"\n🔮 Predicting {len(smiles_list)} molecules...")
    
    results = []
    for i, smiles in enumerate(smiles_list):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(smiles_list)}", end='\r')
        
        result = predict_smiles(
            model_path, 
            smiles, 
            use_fingerprints=use_fingerprints,
            extended_descriptors=extended_descriptors,
            verbose=False
        )
        
        results.append({
            'smiles': smiles,
            'predicted_family': result.get('predicted_family', 'Error'),
            'confidence': result.get('confidence'),
            'error': result.get('error')
        })
    
    print(f"\n   ✓ Complete!")
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict fragrance family from SMILES')
    parser.add_argument('smiles', type=str, help='SMILES string')
    parser.add_argument('--model', type=str, 
                       default='/files/datascience_project_EG/results/models/latest_model.joblib',
                       help='Path to trained model')
    parser.add_argument('--no-fingerprints', action='store_true',
                       help='Model was trained without fingerprints')
    parser.add_argument('--basic-descriptors', action='store_true',
                       help='Model used basic descriptors only')
    
    args = parser.parse_args()
    
    # Make prediction
    result = predict_smiles(
        model_path=args.model,
        smiles=args.smiles,
        use_fingerprints=not args.no_fingerprints,
        extended_descriptors=not args.basic_descriptors
    )
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
    else:
        print(f"\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"SMILES: {result['smiles']}")
        print(f"Predicted Family: {result['predicted_family']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.1%}")
        
        print(f"\nMolecular Descriptors:")
        for desc, value in list(result['descriptors'].items())[:17]:
            print(f"  {desc:20s}: {value:.2f}")
        #print(f"  ... and {len(result['descriptors'])-17} more") # if in fututre I add more descriptors and need to limit output