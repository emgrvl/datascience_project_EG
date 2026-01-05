"""
Plot confidence distribution for all three models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

sys.path.insert(0, '/files/datascience_project_EG')

from scr.features.rdkit_features import compute_rdkit_descriptors
from scr.features.fingerprinting import compute_morgan_fp

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load data
df = pd.read_csv('/files/datascience_project_EG/data/molecules_with_features.csv')

# Load models
models = {
    'Option 1\n(RF + FP)': {
        'path': '/files/datascience_project_EG/results/models/model_random_forest_20251229_180742.joblib',
        'use_fp': True
    },
    'Option 2\n(RF only)': {
        'path': '/files/datascience_project_EG/results/models/model_random_forest_20251229_182033.joblib',
        'use_fp': False
    },
    'Option 3\n(GB + FP)': {
        'path': '/files/datascience_project_EG/results/models/model_gradient_boosting_20251231_002855.joblib',
        'use_fp': True
    }
}

# Prepare features for each model type
def prepare_features(df, use_fingerprints=True):
    metadata_cols = ['CAS number', 'Principal name', 'SMILES', 'fragrance_family',
                     'Primary descriptor', 'Descriptor 2', 'Descriptor 3', 'all_descriptors']
    
    if use_fingerprints:
        feature_cols = [col for col in df.columns if col not in metadata_cols]
    else:
        feature_cols = [col for col in df.columns 
                       if col not in metadata_cols 
                       and not col.startswith(('MorganFP', 'MACCS'))]
    
    return df[feature_cols].fillna(0)

# Get confidence scores for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, model_info) in enumerate(models.items()):
    print(f"Processing {model_name}...")
    
    # Load model
    model = joblib.load(model_info['path'])
    
    # Prepare features
    X = prepare_features(df, use_fingerprints=model_info['use_fp'])
    
    # Get predictions and confidence
    y_pred_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    
    # Get max confidence for each prediction
    max_confidences = np.max(y_pred_proba, axis=1)
    
    # Check if prediction is correct
    y_true = df['fragrance_family']
    correct = (y_pred == y_true).values
    
    # Plot
    ax = axes[idx]
    
    # Histogram for correct and incorrect
    ax.hist(max_confidences[correct], bins=30, alpha=0.7, label='Correct', 
            color='green', edgecolor='black')
    ax.hist(max_confidences[~correct], bins=30, alpha=0.7, label='Incorrect', 
            color='red', edgecolor='black')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{model_name}\nMean: {max_confidences.mean():.3f}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.grid(alpha=0.3)
    
    # Add stats
    textstr = f'Correct: {correct.sum()}\nIncorrect: {(~correct).sum()}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Prediction Confidence Distribution by Model', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/files/datascience_project_EG/results/figures/confidence_distribution.png', 
            dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/figures/confidence_distribution.png")
plt.close()

# Create box plot comparison
fig, ax = plt.subplots(figsize=(10, 6))

confidence_data = []
model_labels = []

for model_name, model_info in models.items():
    model = joblib.load(model_info['path'])
    X = prepare_features(df, use_fingerprints=model_info['use_fp'])
    y_pred_proba = model.predict_proba(X)
    max_confidences = np.max(y_pred_proba, axis=1)
    
    confidence_data.append(max_confidences)
    model_labels.append(model_name.replace('\n', ' '))

bp = ax.boxplot(confidence_data, labels=model_labels, patch_artist=True)

# Color the boxes
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Confidence', fontsize=12)
ax.set_title('Confidence Score Distribution Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('/files/datascience_project_EG/results/figures/confidence_boxplot.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/confidence_boxplot.png")
plt.close()

print("\n" + "="*60)
print("CONFIDENCE ANALYSIS COMPLETE!")
print("="*60)
