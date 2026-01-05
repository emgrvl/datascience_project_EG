"""
Overall Model Comparison - Summary Visualization
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Model performance data (from your results)
models = ['Option 1\n(RF + FP)', 'Option 2\n(RF only)', 'Option 3\n(GB + FP)']

metrics = {
    'Accuracy': [0.728, 0.777, 0.850],
    'F1-macro': [0.604, 0.621, 0.684],
    'F1-weighted': [0.721, 0.770, 0.844],
    'Balanced Accuracy': [0.623, 0.623, 0.683]
}

# Additional info
features = [2065, 17, 2065]
training_time = [8, 2, 360]  # minutes (6 hours = 360 min)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. Main metrics comparison
ax1 = fig.add_subplot(gs[0, :])

x = np.arange(len(models))
width = 0.2

colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, (metric, values) in enumerate(metrics.items()):
    offset = width * (i - 1.5)
    bars = ax1.bar(x + offset, values, width, label=metric, color=colors[i], 
                   edgecolor='black', alpha=0.8)
    
    # Add value labels
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{values[j]:.3f}', ha='center', va='bottom', fontsize=9)

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_ylim([0, 1])
ax1.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Goal (70%)')
ax1.grid(axis='y', alpha=0.3)

# 2. Feature count
ax2 = fig.add_subplot(gs[1, 0])

bars = ax2.bar(models, features, color=['steelblue', 'seagreen', 'coral'], 
               edgecolor='black', alpha=0.8)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{int(features[i])}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax2.set_title('Feature Complexity', fontsize=13, fontweight='bold', pad=15)
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)

# 3. Training time
ax3 = fig.add_subplot(gs[1, 1])

bars = ax3.bar(models, training_time, color=['steelblue', 'seagreen', 'coral'], 
               edgecolor='black', alpha=0.8)

for i, bar in enumerate(bars):
    height = bar.get_height()
    if training_time[i] >= 60:
        label = f'{training_time[i]/60:.1f}h'
    else:
        label = f'{int(training_time[i])}m'
    ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
            label, ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
ax3.set_title('Computational Cost', fontsize=13, fontweight='bold', pad=15)
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3)

plt.suptitle('DeepScent Model Comparison Summary', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/files/datascience_project_EG/results/figures/model_comparison_summary.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/model_comparison_summary.png")
plt.close()

# Create a simple table visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Model', 'Accuracy', 'F1-macro', 'Features', 'Training Time', 'Best For'],
    ['Option 1\n(RF + FP)', '72.8%', '0.604', '2,065', '8 min', 'Baseline'],
    ['Option 2\n(RF only)', '77.7%', '0.621', '17', '2 min', 'Interpretability ⭐'],
    ['Option 3\n(GB + FP)', '85.0%', '0.684', '2,065', '6 hours', 'Max Performance 🏆'],
]

colors = [['lightgray']*6] + [['white']*6]*3
colors[2] = ['lightgreen'] * 6  # Highlight Option 2
colors[3] = ['lightcoral'] * 6  # Highlight Option 3

table = ax.table(cellText=table_data, cellColours=colors, loc='center',
                cellLoc='center', colWidths=[0.2, 0.12, 0.12, 0.12, 0.15, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('navy')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('DeepScent Model Comparison Table', fontsize=14, fontweight='bold', pad=20)

plt.savefig('/files/datascience_project_EG/results/figures/model_comparison_table.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/model_comparison_table.png")
plt.close()

print("\n" + "="*60)
print("MODEL COMPARISON VISUALIZATIONS COMPLETE!")
print("="*60)
