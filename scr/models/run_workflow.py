"""
DeepScent Complete Workflow
From SMILES to Trained Model
"""
import sys
from pathlib import Path
import os

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ Found: {description}")
        return True
    else:
        print(f"❌ Missing: {description}")
        return False

def main():
    print("="*70)
    print("DEEPSCENT - COMPLETE WORKFLOW")
    print("="*70)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...\n")
    
    files_needed = {
        '/files/datascience_project_EG/data/ifra_with_smiles.csv': 'SMILES data',
        '/files/datascience_project_EG/data/molecules_with_features.csv': 'Features (RDKit + Fingerprints)',
    }
    
    all_present = True
    for filepath, desc in files_needed.items():
        if not check_file_exists(filepath, desc):
            all_present = False
    
    if not all_present:
        print("\n⚠️ Missing files! Here's what to do:\n")
        
        if not Path('/files/datascience_project_EG/data/ifra_with_smiles.csv').exists():
            print("1. Fetch SMILES:")
            print("   python pipeline.py --fetch-smiles")
            print()
        
        if not Path('/files/datascience_project_EG/data/molecules_with_features.csv').exists():
            print("2. Generate features:")
            print("   python generate_features.py")
            print()
        
        print("Then run this script again!")
        return
    
    print("\n✅ All prerequisites met!\n")
    
    # Training workflow
    print("="*70)
    print("TRAINING WORKFLOW")
    print("="*70)
    
    print("\n📊 Step 1: Quick Data Check")
    print("-"*70)
    
    import pandas as pd
    df = pd.read_csv('/files/datascience_project_EG/data/molecules_with_features.csv')
    
    print(f"Total molecules: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"With SMILES: {df['SMILES'].notna().sum()}")
    
    # Check family distribution
    print(f"\nTop 10 Fragrance Families:")
    family_counts = df['fragrance_family'].value_counts().head(10)
    for family, count in family_counts.items():
        print(f"  {family:20s}: {count:4d} molecules")
    
    # Recommendation
    total_families = df['fragrance_family'].nunique()
    small_families = (df['fragrance_family'].value_counts() < 10).sum()
    
    print(f"\nTotal families: {total_families}")
    print(f"Families with <10 samples: {small_families}")
    
    if small_families > 5:
        print(f"\n💡 Recommendation: Use --min-family-size 15 to filter small families")
        recommended_min = 15
    else:
        recommended_min = 10
    
    # Ask user to proceed
    print("\n" + "="*70)
    print("READY TO TRAIN MODEL")
    print("="*70)
    
    print("\nRecommended command:")
    print(f"  python /files/datascience_project_EG/scr/models/train.py --min-family-size {recommended_min}")
    
    print("\nOptions:")
    print("  1. Train with default settings (Random Forest + SMOTE)")
    print("  2. Train without fingerprints (faster, interpretable)")
    print("  3. Train with Gradient Boosting")
    print("  4. Custom settings")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        cmd = f"python /files/datascience_project_EG/scr/models/train.py --min-family-size {recommended_min}"
    elif choice == '2':
        cmd = f"python /files/datascience_project_EG/scr/models/train.py --no-fingerprints --min-family-size {recommended_min}"
    elif choice == '3':
        cmd = f"python /files/datascience_project_EG/scr/models/train.py --model gradient_boosting --min-family-size {recommended_min}"
    elif choice == '4':
        print("\nCustomize your training:")
        model = input("  Model (random_forest/gradient_boosting) [random_forest]: ").strip() or "random_forest"
        use_fp = input("  Use fingerprints? (y/n) [y]: ").strip().lower() or "y"
        min_size = input(f"  Min family size [{recommended_min}]: ").strip() or str(recommended_min)
        
        cmd = f"python /files/datascience_project_EG/scr/models/train.py --model {model} --min-family-size {min_size}"
        if use_fp == 'n':
            cmd += " --no-fingerprints"
    elif choice == '5':
        print("\nExiting. To train later, run:")
        print(f"  python /files/datascience_project_EG/scr/models/train.py --min-family-size {recommended_min}")
        return
    else:
        print("\nInvalid choice. Exiting.")
        return
    
    # Run training
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"\nCommand: {cmd}\n")
    
    proceed = input("Proceed? (y/n) [y]: ").strip().lower() or 'y'
    
    if proceed == 'y':
        print("\n🚀 Training model...\n")
        os.system(cmd)
        
        print("\n" + "="*70)
        print("WORKFLOW COMPLETE!")
        print("="*70)
        
        print("\n📊 Next Steps:")
        print("  1. Check results in: results/models/")
        print("  2. Evaluate model: python -m src.models.evaluate")
        print("  3. Make predictions: python -m src.models.predict <SMILES>")
        print("  4. Create visualizations")
        
    else:
        print("\nTraining cancelled.")

if __name__ == "__main__":
    main()
