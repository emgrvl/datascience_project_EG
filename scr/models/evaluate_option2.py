import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load model (Option 2 - no fingerprints)
model = joblib.load('/files/datascience_project_EG/results/models/model_random_forest_20251229_182033.joblib')

# 2. Load data
df = pd.read_csv('/files/datascience_project_EG/data/molecules_with_features.csv')

# 3. Get ONLY RDKit descriptors (no fingerprints)
metadata_cols = ['CAS number', 'Principal name', 'SMILES', 'fragrance_family',
                 'Primary descriptor', 'Descriptor 2', 'Descriptor 3', 'all_descriptors']

# Select only descriptor columns (exclude fingerprints)
descriptor_cols = [col for col in df.columns 
                  if col not in metadata_cols 
                  and not col.startswith(('MorganFP', 'MACCS'))]

print(f"Using {len(descriptor_cols)} RDKit descriptors")
print(f"Features: {descriptor_cols}")

X = df[descriptor_cols].fillna(0)
y = df['fragrance_family']

# 4. Predict and evaluate
y_pred = model.predict(X)

print("\n" + "="*70)
print("OPTION 2 RESULTS (No Fingerprints)")
print("="*70)
print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
print(f"F1-macro: {f1_score(y, y_pred, average='macro'):.3f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))