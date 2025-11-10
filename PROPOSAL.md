Title:
DeepScent – Predicting fragrance families from molecular architectures

Category: 
Scientific Computing, Machine Learning

Problem Statement/Motivation:
The art of perfumery begins at the molecular level: each scent we perceive comes from specific combinations of aromatic molecules interacting with human olfactory receptors. Despite centuries of empirical knowledge, the process of predicting a molecule’s scent profile — such as whether it smells floral, citrus, woody, or ambery - remains very qualitative and subjective.

With advances in cheminformatics, we now have the opportunity to bridge chemistry and sensory science. DeepScent aims to build a machine learning model that predicts fragrance family classifications from molecular structures. 

The project will use information derived from the IFRA Fragrance Ingredient Glossary (FIG), developed by The International Fragrance Association — a curated dataset that classifies aroma molecules into 27 standardized odor families. By linking structural chemistry with olfactory classification, this project will explore whether measurable chemical properties can explain — or even forecast — a molecule’s olfactory character, therefore providing a foundation for computational perfume design and molecule screening.


Planned Approach & Technologies:
Data Collection:
- Use the IFRA Fragrance Ingredient Glossary (FIG) as the primary dataset: Each molecule includes its CAS number, chemical name, and expert-assigned odor family label (e.g., amber, citrus, floral, green, spicy).
- Supplement FIG entries with molecular structure data (SMILES strings) and physicochemical properties from PubChem and The Good Scents Company databases.
- Extract molecular data including SMILES strings, chemical properties (e.g. molecular weight, functional groups), and associated odor descriptors or fragrance families.

Feature Engineering:
- Use RDKit to calculate molecular descriptors like LogP (hydrophobicity),  molecular weight, topological polar surface area, aromatic ring count, and functional group presence.
- Generate molecular fingerprints (Morgan or MACCS) for structural similarity representation.


Model Development:
- Split data into training and test sets, with balanced class distribution.
- Train several classification models — Random Forest, Support Vector Machine, and Gradient Boosting — using scikit-learn.
- Evaluate models using accuracy, macro-F1, and confusion matrices to assess classification across the 27 odor families.
- Apply PCA and t-SNE for dimensionality reduction and visualization of clusters in “odor space.”

Visualization & Interpretation:
- Apply Principal Component Analysis or t-SNE (interactive 2D maps) to visualize the “odor space”, showing how molecular similarities cluster within fragrance families.
- Use feature importance plots to highlight which molecular properties most influence scent classification.


Expected Challenges & Mitigation Strategies:
- Data quality & quantity: Fragrance molecule data is scattered and sometimes incomplete. I will consolidate multiple sources and perform deduplication and normalization.

- Class imbalance: Some fragrance families may be underrepresented. I will address this using the Synthetic Minority Over-sampling Technique and weighted loss functions to address this.

- Ambiguous Labels: Some molecules belong to multiple odor families. To simplify, I’ll start with single-label classification, later extending to multi-label prediction as a stretch goal.

- Chemical descriptor correlation: Use feature selection and PCA to prevent overfitting.


Success Criteria:
- Curated dataset of ≥500 FIG-validated fragrance ingredients with complete molecular descriptors.
- A trained model achieving ≥70% accuracy on test data when predicting fragrance families.
- Clear visualizations illustrating odor clustering and key molecular predictors.


Stretch Goals:
- Extend to multi-label classification, where molecules can belong to more than one fragrance family.
- Develop a Streamlit web app where users can input a molecule (via SMILES string) and receive a predicted olfactory family.
- Explore generative modeling (using variational autoencoders) to propose new molecular structures within a desired scent profile.
- Explore unsupervised clustering to discover emergent “hidden families” beyond IFRA’s taxonomy.

Impact & vision:
DeepScent merges chemistry, data science, and perfumery to create a scientific framework that tries to understand how molecular structures relate to scent perception. The project pushes beyond standard data analysis to explore machine learning’s role in a creative and sensory domain. Its results could inspire future research in computational fragrance design, virtual molecule screening, and AI-assisted perfumery innovation.

