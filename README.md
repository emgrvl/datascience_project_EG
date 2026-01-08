# **DeepScent – Predicting fragrance families from molecular architectures**

## **Overview**

**DeepScent** is a data science and machine learning project that explores the intersection of **chemistry and perfumery**.
The goal is to develop a model capable of predicting a molecule’s **fragrance family** (e.g., *floral, woody, fruity, green, oriental*) based on its **molecular descriptors**.

By leveraging cheminformatics data and statistical learning, this project seeks to reveal which **chemical features** most strongly influence scent perception — a step toward computational perfume design.

---

## **Objectives**

* Collect and preprocess fragrance molecule data from open databases
* Generate chemical descriptors using **RDKit**
* Train machine learning models to classify molecules into fragrance families
* Visualize relationships between molecular structure and olfactory characteristics
* Evaluate and interpret model performance

## **Project Highlights**

### **Scientific Insights**
- **Polar Surface Area (TPSA)** is the strongest predictor (9.5% importance)
- **Lipophilicity (LogP)** controls scent persistence (7.8% importance)
- Chemical properties outperform structural patterns for fragrance classification
- Interpretable model (77.7% accuracy) vs high-performance model (85% accuracy) trade-off

### **Technical Achievements**
- End-to-end pipeline from SMILES retrieval to deployment
- SMOTE-based class imbalance handling (464:3 ratio)
- Comprehensive evaluation with confusion matrices and confidence analysis
- 20 unit tests ensuring code quality

---

## **Tech Stack**

| Component | Purpose | Version |
|-----------|---------|---------|
| **Python** | Core programming language | 3.12+ |
| **RDKit** | Molecular descriptor computation | 2023.09.5 |
| **scikit-learn** | Machine learning & evaluation | 1.4.2 |
| **imbalanced-learn** | SMOTE for class imbalance | 0.12.0 |
| **pandas, NumPy** | Data wrangling & numerical analysis | Latest |
| **matplotlib, seaborn** | Data visualization | Latest |
| **joblib** | Model persistence | Latest |

---

## **Repository Structure**

```
datascience_project_EG/
├── README.md                          # Project documentation
├── AI_USAGE.md
├── PROPOSAL.md
├── requirements.txt 
├── main.py                            # Python dependencies
├── setup.py
├── pipeline.py
├── scr/
│   ├── __init__.py                    #Command-line interface for DeepScent
│   ├── data/  
│        ├── load_data.py
│        ├── preprocess.py
│   ├── features/   
│        ├── fingerprinting.py
│        ├── rdkit_features.py          # RDKit descriptor calculations   
│        ├── generate_features.py
│   ├── models/ 
│        ├── evaluate.py          
│        ├── evaluate_option2.py        # Evaluation of Option 2
│        ├── predict.py
│        ├── train.py                   # ML (RF, SVM, etc.)
│        ├── run_workflow.py            # Model pipeline
│   ├── utils/
│        ├── io.py
├── data/
│   ├── raw_ifra-fig.csv                # Unprocessed datasets (FIG - IFRA)
│   ├── fig_with_smiles_progress.csv 
│   ├── fig_with_smiles_sample.csv   
│   ├── ifra_preprocesseded.csv      
│   ├── ifra_with_smiles.csv            # Cleaned and standardized datasets
│   └── molecules_with_features.csv     # Final modeling dataset
├── results/
│   ├── data_analysis_report.txt        # summary of stats
│   ├── metrics/                        # Accuracy, confusion matrices, reports
│   ├── figures/                        # Plots, confusion matrices, feature importance
│   └── models/                         # Saved trained models (.joblib)
├── tests/
│   ├── create_all_visualizations.py    
│   ├── plot_confidence_distribution.py 
│   ├── plot_feature_importance.py      
│   ├── plot_model_comparison.py        
│   ├── plot_per_family_accuracy.py 
│   ├── test_predictions.py            
│   └── test_*.py                       # Unit and integration tests
---

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/emgrvl/datascience_project_EG.git
cd datascience_project_EG
```

### **2. Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
rdkit>=2023.09.5
scikit-learn>=1.4.2
imbalanced-learn>=0.12.0
pandas>=2.1.4
numpy>=1.26.3
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.3.0
```

### **4. Verify Installation**

```bash
python -c "from rdkit import Chem; print('RDKit installed successfully!')"
```

---

## **Quick Start**

### **Option A: Use Pre-trained Models** (Fastest)

```python
from scr.models.predict import predict_smiles

# Load best model (85% accuracy)
model_path = "results/models/model_gradient_boosting_20251231_002855.joblib"

# Predict fragrance family
result = predict_smiles(model_path, "CC(C)=CCCC(C)(O)C=C")  # Linalool

print(f"Predicted: {result['predicted_family']}")
print(f"Confidence: {result['confidence']:.1%}")
# Output: Predicted: floral, Confidence: 99.0%
```

### **Option B: Complete Pipeline** (From Scratch)

```bash
# 1. Preprocess data (if starting fresh)
python scr/data/load_data.py

# 2. Generate features
python scr/features/generate_features.py

# 3. Train models
python scr/models/run_workflow.py  # Interactive
# OR
python scr/models/train.py --model gradient_boosting

# 4. Evaluate
python scr/models/evaluate.py \
    --model results/models/model_gradient_boosting_*.joblib \
    --features data/molecules_with_features.csv

# 5. Generate visualizations
python tests/create_all_visualizations.py
```

---

## **Usage Examples**

### **1. Predict Single Molecule**

```python
from scr.models.predict import predict_smiles

# Example molecules
test_molecules = {
    'Linalool (Floral)': 'CC(C)=CCCC(C)(O)C=C',
    'Limonene (Citrus)': 'CC1=CCC(CC1)C(=C)C',
    'Vanillin (Balsamic)': 'COC1=C(C=CC(=C1)C=O)O',
}

model = "results/models/model_gradient_boosting_*.joblib"

for name, smiles in test_molecules.items():
    result = predict_smiles(model, smiles, verbose=True)
    print(f"\n{name}")
    print(f"  → {result['predicted_family']} ({result['confidence']:.1%})")
```

### **2. Batch Predictions**

```python
from scr.models.predict import batch_predict
import pandas as pd

# Load molecules
df = pd.read_csv("data/new_molecules.csv")  # Must have 'SMILES' column

# Predict all
results = batch_predict(
    model_path="results/models/model_gradient_boosting_*.joblib",
    smiles_list=df['SMILES'].tolist()
)

# Save predictions
results.to_csv("predictions.csv", index=False)
```

### **3. Compare All Models**

```bash
# Interactive prediction demo
python tests/test_predictions.py

# Select model: 1, 2, or 3
# See predictions with confidence scores
```

### **4. Generate All Visualizations**

```bash
python tests/create_all_visualizations.py

# Creates 8+ publication-ready figures in results/figures/
```

---

## **Model Performance**

### **Summary Comparison**

| Model | Accuracy | F1-macro | Features | Training Time | Best For |
|-------|----------|----------|----------|---------------|----------|
| **Option 1** (RF + FP) | 72.8% | 0.604 | 2,065 | 8 min | Baseline |
| **Option 2** (RF Desc) | 77.7% | 0.621 | **17** | 2 min | **Interpretability** ⭐ |
| **Option 3** (GB + FP) | **85.0%** | **0.684** | 2,065 | 6 hours | **Max Performance** 🏆 |

### **Key Findings**

**Best Performing Families:**
- Musk-like: 89.3% F1-score
- Acidic: 90.9% F1-score
- Sulfurous: 89.3% F1-score
- Woody: 88.1% F1-score

**Most Important Features (Option 2):**
1. **Polar Surface Area (9.5%)** - Polarity and receptor binding
2. **Lipophilicity (7.8%)** - Membrane permeability and persistence
3. **Balaban Index (7.8%)** - Molecular complexity
4. **Fraction sp³ Carbons (7.3%)** - Saturation level
5. **Rotatable Bonds (7.2%)** - Molecular flexibility

**Challenging Cases:**
- Vanillin: Predicted as "gourmand" (plausible overlap with balsamic)
- Coumarin: Predicted as "powdery" (subjective boundary with herbal)

---

## **Dataset**

### **Source**
- **IFRA Fragrance Ingredient Glossary** (v1.0)
  - 3,119 total molecules
  - 2,146 with valid SMILES (68.8% success rate)
  - 22 fragrance families after filtering

### **Data Pipeline**
1. **Raw data**: IFRA CSV with CAS numbers and descriptors
2. **SMILES retrieval**: PubChem API lookup (70-90% success rate)
3. **Preprocessing**: Standardize 100+ descriptors → 27 families
4. **Filtering**: Remove families with <10 samples → 22 families
5. **Feature engineering**: Compute 17 descriptors + 2,048 fingerprints
6. **Final dataset**: 2,146 molecules × 2,065 features

### **Class Distribution**
- **Largest**: Floral (464), Fruity (428), Green (223)
- **Smallest**: Honey (3), Camphoraceous (4), Ozonic (7)
- **Imbalance ratio**: 464:3 (handled with SMOTE)

---

## **Testing**

### **Run Unit Tests**

```bash
# Run all 20 tests
python tests/test_deepscent.py

# Or use pytest
pytest tests/test_deepscent.py -v
```

**Test Coverage:**
- Data loading (IFRA, PubChem API)
- Feature engineering (RDKit descriptors, fingerprints)
- Model training (RF, GB, SMOTE)
- Prediction pipeline (single, batch)
- Data quality (invalid SMILES, value ranges)

**Expected Output:**
```
======================================================================
DEEPSCENT - RUNNING UNIT TESTS
======================================================================
...
======================================================================
TEST SUMMARY
======================================================================
Total tests:   20
✅ Passed:      18
❌ Failed:      0
⏭️  Skipped:     2
Success rate:  90.0%
All tests passed!
```

---

### **Code Documentation**
All modules include comprehensive docstrings:
```python
def predict_smiles(model_path, smiles, use_fingerprints=True, verbose=True):
    """
    Predict fragrance family from SMILES string
    
    Args:
        model_path: Path to trained model (.joblib)
        smiles: SMILES string of molecule
        use_fingerprints: Whether model uses fingerprints
        verbose: Print prediction details
        
    Returns:
        dict: Prediction results with confidence
    """
```

---

## **Future Work**

### **Methodological Improvements**
- [ ] Graph Neural Networks for molecular representation
- [ ] Multi-label classification (molecules can have multiple families)
- [ ] Hierarchical classification (floral → rose vs jasmine)
- [ ] Active learning for uncertain predictions
- [ ] Ensemble methods combining Options 2 & 3

### **Data Enhancements**
- [ ] Expand to 8,500+ molecules (Good Scents Company)
- [ ] Include 3D molecular conformations
- [ ] Model concentration-dependent effects
- [ ] Predict blended fragrance characteristics
- [ ] Expert validation of predictions

### **Deployment**
- [ ] Web interface with molecular drawing
- [ ] REST API for high-throughput screening
- [ ] Mobile application for on-site classification
- [ ] Explainable AI with natural language explanations
- [ ] Bayesian uncertainty quantification

---


## **Citation**

If you use this project in your research, please cite:

```bibtex
@software{deepscent2025,
  author = {Your Name},
  title = {DeepScent: Predicting fragrance families from molecular architectures},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/emgrvl/datascience_project_EG}
}
```
## **Deliverables**

* Clean, labeled dataset of ≥ 500 fragrance molecules
* Trained classification models with evaluation metrics
* Visualizations of molecular odor clustering
* Technical report (10 pages) and final presentation

---

## **Future Work**

* Multi-label classification for complex scent profiles
* Deep learning for molecule-to-scent prediction
* Generative modeling for AI-based fragrance design

---

## **Acknowledgements**

### **Data Sources**

Data sourced from:
- [IFRA Fragrance Ingredient Glossary](https://ifrafragrance.org/about-fragrance/fragrance-ingredients-glossary/) - "Information derived from the IFRA Fragrance Ingredient Glossary, developed by The International Fragrance Association"
- [PubChem](https://pubchem.ncbi.nlm.nih.gov) - SMILES strings and molecular data
- [The Good Scents Company](http://www.thegoodscentscompany.com) - Additional fragrance data

### **Libraries & Tools**
- [RDKit](https://www.rdkit.org) - Open-source cheminformatics
- [scikit-learn](https://scikit-learn.org) - Machine learning in Python
- [imbalanced-learn](https://imbalanced-learn.org) - SMOTE implementation

### **Inspiration**
- Keller et al. (2017) - "Predicting human olfactory perception from chemical features of odor molecules"
- Rogers & Hahn (2010) - "Extended-connectivity fingerprints"



