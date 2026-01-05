"""
Feature Importance Analysis for Option 2 (Descriptors Only)
FIXED VERSION
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set style
sns.set_style("whitegrid")

# Load Option 2 model (descriptors only)
model_path = '/files/datascience_project_EG/results/models/model_random_forest_20251229_182033.joblib'
model = joblib.load(model_path)

# Check what steps are in the pipeline
print("Pipeline steps:", model.steps if hasattr(model, 'steps') else "Not a pipeline")

# Get the classifier - try different possible names
clf = None
if hasattr(model, 'named_steps'):
    # Try common names
    for name in ['clf', 'classifier', 'randomforestclassifier', 'model']:
        if name in model.named_steps:
            clf = model.named_steps[name]
            print(f"Found classifier with name: '{name}'")
            break
    
    # If not found, just get the last step
    if clf is None:
        clf = model.steps[-1][1]
        print(f"Using last step: {model.steps[-1][0]}")
else:
    clf = model
    print("Model is not a pipeline, using directly")

# Check if it has feature_importances_
if not hasattr(clf, 'feature_importances_'):
    print("ERROR: Model doesn't have feature_importances_ attribute")
    print(f"Model type: {type(clf)}")
    exit(1)

# Get feature importance
importances = clf.feature_importances_

# Feature names (17 RDKit descriptors)
feature_names = [
    'MolWt', 'LogP', 'TPSA', 'NumAromaticRings', 'NumHDonors',
    'NumHAcceptors', 'NumRotatableBonds', 'NumHeteroatoms',
    'NumRings', 'NumSaturatedRings', 'NumAliphaticRings',
    'HeavyAtomCount', 'FractionCsp3', 'MolMR', 'BalabanJ',
    'BertzCT', 'NumValenceElectrons'
]

# Create DataFrame
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

# Full names for better readability
full_names = {
    'MolWt': 'Molecular Weight',
    'LogP': 'Lipophilicity (LogP)',
    'TPSA': 'Polar Surface Area',
    'NumAromaticRings': 'Aromatic Rings',
    'NumHDonors': 'H-Bond Donors',
    'NumHAcceptors': 'H-Bond Acceptors',
    'NumRotatableBonds': 'Rotatable Bonds',
    'NumHeteroatoms': 'Heteroatoms',
    'NumRings': 'Total Rings',
    'NumSaturatedRings': 'Saturated Rings',
    'NumAliphaticRings': 'Aliphatic Rings',
    'HeavyAtomCount': 'Heavy Atoms',
    'FractionCsp3': 'Fraction sp³ Carbons',
    'MolMR': 'Molar Refractivity',
    'BalabanJ': 'Balaban Index',
    'BertzCT': 'Bertz Complexity',
    'NumValenceElectrons': 'Valence Electrons'
}

feature_df['Full Name'] = feature_df['Feature'].map(full_names)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.viridis(feature_df['Importance'] / feature_df['Importance'].max())
bars = ax.barh(range(len(feature_df)), feature_df['Importance'], color=colors, edgecolor='black')

ax.set_yticks(range(len(feature_df)))
ax.set_yticklabels(feature_df['Full Name'])
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance - Molecular Descriptors\n(Random Forest - Option 2)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(feature_df.iterrows()):
    ax.text(row['Importance'] + 0.002, i, f"{row['Importance']:.3f}", 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('/files/datascience_project_EG/results/figures/feature_importance.png', 
            dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/figures/feature_importance.png")
plt.close()

# Print top 5
print("\n" + "="*60)
print("TOP 5 MOST IMPORTANT FEATURES")
print("="*60)
top5 = feature_df.tail(5)
for idx, row in top5.iterrows():
    print(f"{row['Full Name']:30s}: {row['Importance']:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print(f"Most important: {feature_df.iloc[-1]['Full Name']}")
print(f"Least important: {feature_df.iloc[0]['Full Name']}")
print("\nThese features drive fragrance family predictions!")
