"""
Per-Family Accuracy Comparison Across Models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

sys.path.insert(0, '/files/datascience_project_EG')

# Set style
sns.set_style("whitegrid")

# Load data
df = pd.read_csv('/files/datascience_project_EG/data/molecules_with_features.csv')

# Prepare features function
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

# Models
models = {
    'Option 1': {
        'path': '/files/datascience_project_EG/results/models/model_random_forest_20251229_180742.joblib',
        'use_fp': True
    },
    'Option 2': {
        'path': '/files/datascience_project_EG/results/models/model_random_forest_20251229_182033.joblib',
        'use_fp': False
    },
    'Option 3': {
        'path': '/files/datascience_project_EG/results/models/model_gradient_boosting_20251231_002855.joblib',
        'use_fp': True
    }
}

# Calculate per-family accuracy for each model
results = []

for model_name, model_info in models.items():
    print(f"Analyzing {model_name}...")
    
    model = joblib.load(model_info['path'])
    X = prepare_features(df, use_fingerprints=model_info['use_fp'])
    y_true = df['fragrance_family']
    y_pred = model.predict(X)
    
    # Calculate accuracy per family
    for family in y_true.unique():
        mask = y_true == family
        if mask.sum() > 0:
            family_acc = (y_pred[mask] == y_true[mask]).mean()
            results.append({
                'Model': model_name,
                'Family': family,
                'Accuracy': family_acc,
                'Count': mask.sum()
            })

results_df = pd.DataFrame(results)

# Filter to families with at least 10 samples
family_counts = results_df.groupby('Family')['Count'].first()
large_families = family_counts[family_counts >= 10].index
results_df = results_df[results_df['Family'].isin(large_families)]

# Get top 12 families by sample count
top_families = results_df.groupby('Family')['Count'].first().nlargest(12).index
results_df_top = results_df[results_df['Family'].isin(top_families)]

# Pivot for grouped bar chart
pivot_df = results_df_top.pivot(index='Family', columns='Model', values='Accuracy')
pivot_df = pivot_df.sort_values('Option 3', ascending=True)

# Plot
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(pivot_df))
width = 0.25

bars1 = ax.barh(x - width, pivot_df['Option 1'], width, label='Option 1 (RF+FP)', 
                color='steelblue', edgecolor='black', alpha=0.8)
bars2 = ax.barh(x, pivot_df['Option 2'], width, label='Option 2 (RF only)', 
                color='seagreen', edgecolor='black', alpha=0.8)
bars3 = ax.barh(x + width, pivot_df['Option 3'], width, label='Option 3 (GB+FP)', 
                color='coral', edgecolor='black', alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels(pivot_df.index)
ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Per-Family Accuracy Comparison\n(Top 12 Families by Sample Size)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1.05])
ax.grid(axis='x', alpha=0.3)

# Add vertical line at 0.7 (your goal)
ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Goal (70%)')

plt.tight_layout()
plt.savefig('/files/datascience_project_EG/results/figures/per_family_accuracy.png', 
            dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/figures/per_family_accuracy.png")
plt.close()

# Summary statistics
print("\n" + "="*60)
print("PER-FAMILY ACCURACY SUMMARY")
print("="*60)

for model in ['Option 1', 'Option 2', 'Option 3']:
    model_data = results_df[results_df['Model'] == model]
    print(f"\n{model}:")
    print(f"  Mean accuracy: {model_data['Accuracy'].mean():.3f}")
    print(f"  Median accuracy: {model_data['Accuracy'].median():.3f}")
    print(f"  Families >80%: {(model_data['Accuracy'] > 0.8).sum()}/{len(model_data)}")
    print(f"  Families >90%: {(model_data['Accuracy'] > 0.9).sum()}/{len(model_data)}")

print("\n" + "="*60)
