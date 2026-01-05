import sys

sys.path.insert(0, '/files/datascience_project_EG')
from scr.models.predict import predict_smiles

# Test molecules
test_molecules = {
    'Linalool (Floral)': 'CC(C)=CCCC(C)(O)C=C',
    'Limonene (Citrus)': 'CC1=CCC(CC1)C(=C)C',
    'Vanillin (Balsamic)': 'COC1=C(C=CC(=C1)C=O)O',
    'Eugenol (Spicy)': 'COC1=C(C=CC(=C1)CC=C)O',
    'Menthol (Minty)': 'CC(C)C1CCC(C)CC1O',
    'Coumarin (Herbal)': 'O=C1OC2=CC=CC=C2C=C1',
}

# Available models
models = {
    '1': {
        'name': 'Random Forest + Fingerprints (Option 1)',
        'path': 'results/models/model_random_forest_20251229_180742.joblib',  
        'use_fp': True,
        'accuracy': '72.8%'
    },
    '2': {
        'name': 'Random Forest - Descriptors Only (Option 2)',
        'path': 'results/models/model_random_forest_20251229_182033.joblib',  
        'use_fp': False,
        'accuracy': '77.7%'
    },
    '3': {
        'name': 'Gradient Boosting + Fingerprints (Option 3)',
        'path': 'results/models/model_gradient_boosting_20251231_002855.joblib',
        'use_fp': True,
        'accuracy': '85.0%'
    }
}

print("="*70)
print("DEEPSCENT - PREDICTION DEMO")
print("="*70)

# Model selection
print("\nAvailable Models:")
for key, info in models.items():
    print(f"  {key}. {info['name']} (Accuracy: {info['accuracy']})")

choice = input("\nSelect model (1-3) [3]: ").strip() or '3'

if choice not in models:
    print("Invalid choice. Using default (Option 3)")
    choice = '3'

selected_model = models[choice]
model_path = selected_model['path']
use_fingerprints = selected_model['use_fp']

print(f"\n✓ Using: {selected_model['name']}")
print(f"  Model: {model_path}")
print(f"  Features: {'Descriptors + Fingerprints' if use_fingerprints else 'Descriptors Only'}")

print("\n" + "="*70)
print("PREDICTIONS")
print("="*70)

# Track results
correct = 0
total = 0

for name, smiles in test_molecules.items():
    print(f"\n{name}")
    print(f"SMILES: {smiles}")
    
    result = predict_smiles(
        model_path, 
        smiles, 
        use_fingerprints=use_fingerprints,
        extended_descriptors=True,
        verbose=False
    )
    
    if 'error' in result:
        print(f"  ❌ Error: {result['error']}")
    else:
        pred = result['predicted_family']
        conf = result.get('confidence')
        
        if conf:
            print(f"  → Predicted: {pred} ({conf:.1%} confidence)")
        else:
            print(f"  → Predicted: {pred}")
        
        # Show top 3
        if 'top_3_predictions' in result and len(result['top_3_predictions']) > 1:
            print("  Top 3:")
            for i, (family, prob) in enumerate(result['top_3_predictions'][:3], 1):
                print(f"    {i}. {family:15s} {prob:.1%}")
        
        # Check if correct
        expected = name.split('(')[1].split(')')[0].lower()
        total += 1
        if pred.lower() == expected:
            print(f"  ✅ Correct!")
            correct += 1
        else:
            print(f"  ⚠ Expected: {expected}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Model: {selected_model['name']}")
print(f"Test Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"Overall Model Accuracy: {selected_model['accuracy']}")