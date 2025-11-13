#Command-line interface for AI Nose.

import argparse
from .models.predict import predict_smiles

def main():
    parser = argparse.ArgumentParser(description="Predict fragrance family from SMILES.")
    parser.add_argument("smiles", type=str, help="Input SMILES string")
    parser.add_argument("--model", type=str, default="results/models/rf_model.joblib", help="Path to trained model")
    args = parser.parse_args()

    result = predict_smiles(args.model, args.smiles)
    print(result)

if __name__ == "__main__":
    main()