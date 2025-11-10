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

---

## **Tech Stack**

| Component                      | Purpose                                            |
| ------------------------------ | -------------------------------------------------- |
| **Python 3.10+**               | Core programming language                          |
| **pandas, NumPy**              | Data wrangling & numerical analysis                |
| **RDKit**                      | Molecular descriptor computation                   |
| **scikit-learn**               | Machine learning & evaluation                      |
| **matplotlib, seaborn**        | Data visualization                                 |
| **t-SNE / PCA**                | Dimensionality reduction for odor-space mapping    |
| **SMOTE (imbalanced-learn)**   | Handling class imbalance                           |
| **Streamlit** *(stretch goal)* | Interactive user interface for molecule prediction |

---

## **Repository Structure**

```
ai-nose/
├── README.md
├── PROPOSAL.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Cleaning, merging, and feature extraction
│   ├── descriptor_generation.py     # RDKit descriptor calculations
│   ├── model_training.py            # ML pipeline (RF, SVM, etc.)
│   ├── evaluation.py                # Metrics and visualizations
│   └── visualization.py             # PCA, t-SNE, and feature plots
├── data/
│   ├── raw_ifra-fig.csv/                         # Unprocessed datasets (FIG - IFRA)
│   ├── processed/                   # Cleaned and standardized datasets
│   └── molecules.csv                # Final modeling dataset
├── results/
│   ├── metrics/                     # Accuracy, confusion matrices, reports
│   ├── figures/                     # PCA/t-SNE plots, feature importance
│   └── models/                      # Saved trained models (.pkl)
├── tests/
│   └── test_*.py                    # Unit and integration tests
└── docs/
    └── methodology.md               # Extended explanation of methods
```

---

## **Setup Instructions**

1. **Clone the repository**

   ```bash
   git clone https://github.com/emgrvl/datascience_project_EG.git
   cd datascience_project_EG
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # or `venv\Scripts\activate` on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**

   ```bash
   python src/data_preprocessing.py
   python src/descriptor_generation.py
   python src/model_training.py
   ```

5. **(Optional)** Launch the Streamlit app (stretch goal)

   ```bash
   streamlit run src/app.py
   ```

---

## **Example Usage**

Predict the fragrance family of a molecule based on its SMILES representation:

```python
from src.model_training import load_model, predict_family

model = load_model("results/models/final_rf_model.pkl")
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"   # Example: benzyl acetate
prediction = predict_family(model, smiles)
print(prediction)
# Output: "Floral / Fruity"
```

---

## **Dataset Plan**

| Source                           | Description                           | Notes                             |
| -------------------------------- | ------------------------------------- | --------------------------------- |
| **FIG IFRA Database**            | Aroma molecules with scent descriptors| Downloaded on CSV                 |
| **GoodScents Company Database**  | Aroma molecules with scent descriptors| Scraped or downloaded in CSV      |
| **PubChem**                      | Molecular structure and SMILES        | API-based retrieval               |
| **Synthetic Dataset (if needed)**| Augmented via SMILES sampling         | Balances underrepresented classes |

Each record will include:
`[Molecule Name, SMILES, MW, TPSA, Aromatic Rings, LogP, Fragrance Family, Odor Terms]`

---

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

Data sourced from:

* [International Fragrance Association - Fragrance Ingredients Glossary](https://ifrafragrance.org/about-fragrance/fragrance-ingredients-glossary/) - "Information derived from the IFRA Fragrance Ingredient Glossary, developed by The International Fragrance Association"
* [PubChem](https://pubchem.ncbi.nlm.nih.gov)
* [The GoodScents Company](http://www.thegoodscentscompany.com)