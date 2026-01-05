#Command-line interface for DeepScent
import argparse
import sys
sys.path.insert(0, '/files/datascience_project_EG')

from scr.models.predict import predict_smiles

def main():
    parser = argparse.ArgumentParser(
        description="Predict fragrance family from SMILES string"
    )
    parser.add_argument("smiles", type=str, help="SMILES string of molecule")
    parser.add_argument(
        "--model", 
        type=str, 
        default="results/models/model_gradient_boosting_20251231_002855.joblib",
        help="Path to trained model"
    )
    parser.add_argument(
        "--no-fingerprints",
        action="store_true",
        help="Use descriptor-only model"
    )
    
    args = parser.parse_args()
    
    result = predict_smiles(
        args.model, 
        args.smiles,
        use_fingerprints=not args.no_fingerprints,
        verbose=True
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"\nPredicted Family: {result['predicted_family']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.1%}")

if __name__ == "__main__":
    main()