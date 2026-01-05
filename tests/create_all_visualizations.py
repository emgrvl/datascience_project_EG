"""
Master Visualization Script
Runs all visualization scripts and creates a complete set of figures
"""
import subprocess
import os
import sys

print("="*70)
print("DEEPSCENT - CREATING ALL VISUALIZATIONS")
print("="*70)

# Change to project directory
os.chdir('/files/datascience_project_EG')

# Create figures directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)

scripts = [
    ('Confidence Distribution', 'plot_confidence_distribution.py'),
    ('Feature Importance', 'plot_feature_importance.py'),
    ('Per-Family Accuracy', 'plot_per_family_accuracy.py'),
    ('Model Comparison', 'plot_model_comparison.py'),
]

print("\nRunning visualization scripts...\n")

for name, script in scripts:
    print(f"📊 Creating: {name}")
    print("-" * 70)
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, f'tests/{script}'],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"✅ {name} - SUCCESS\n")
        else:
            print(f"❌ {name} - FAILED")
            print(result.stderr)
            print()
    
    except subprocess.TimeoutExpired:
        print(f"⏱️ {name} - TIMEOUT (>2 minutes)")
        print()
    except Exception as e:
        print(f"❌ {name} - ERROR: {str(e)}")
        print()

# List all generated figures
print("\n" + "="*70)
print("GENERATED VISUALIZATIONS")
print("="*70)

figures_dir = 'results/figures'
if os.path.exists(figures_dir):
    figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    
    if figures:
        print(f"\n✅ Found {len(figures)} visualizations:\n")
        for i, fig in enumerate(sorted(figures), 1):
            size = os.path.getsize(os.path.join(figures_dir, fig)) / 1024
            print(f"  {i}. {fig:<40s} ({size:.1f} KB)")
    else:
        print("\n⚠️ No PNG files found in results/figures/")
else:
    print(f"\n❌ Directory not found: {figures_dir}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\n📁 All figures saved in: {figures_dir}/")
print("\nVisuals:")
print("  1. confusion_matrix_GB.png and confusion_matrix.png - Shows predictions patterns")
print("  2. model_comparison_summary.png - Overall performance")
print("  3. feature_importance.png - What drives predictions")
print("  4. confidence_distribution.png - Model certainty")
print("  5. per_family_accuracy.png - Family-specific performance")
