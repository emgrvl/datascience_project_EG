# AI Usage Disclosure

## Project: DeepScent

This document outlines how AI assistance was used during the development of this project, in accordance with academic integrity and transparency guidelines.

---

## Summary of AI Assistance

AI tools were used as a coding assistant and learning resource throughout the development of this project. I developped independently the core scientific methodology, data analysis decisions, and interpretations.


AI assistance was instrumental in identifying and resolving bugs, particularly:

- Import path errors: Correcting `src` vs `scr` directory naming inconsistencies
- Train/test split issues: Fixing the `ValueError: test_size error by implementing dynamic test size calculation
- File path resolution: Debugging issues with relative vs absolute paths when accessing data files
- RDKit compatibility: Resolving deprecated function calls and version-specific syntax
- DataFrame operations: Fixing `FutureWarning` issues with append operations (migrated to list-based concatenation)
- SMOTE integration: Troubleshooting class imbalance handling with `imbalanced-learn` pipeline syntax

Example: The AI helped identify that using `df.append()` in a loop was causing performance issues and warnings, suggesting the more efficient pattern of building a list and converting to DataFrame once.

AI assisted in improving code quality and consistency:

- Variable naming conventions: Standardizing variable names across modules (e.g., `df_features`, `descriptor_cols`, `family_counts`)
- Function signatures: Ensuring consistent parameter naming and type hints
- Code formatting: Applying PEP 8 style guidelines consistently for user's better understanding
- Documentation strings: Standardizing docstring format across all functions
- Error messages: Making error messages more descriptive and actionable

Example: Refactoring inconsistent naming like `desc`, `descriptors`, `mol_desc` to a standardized `descriptors` pattern throughout the codebase.


AI significantly aided in creating comprehensive documentation:

- Code comments: Adding inline comments explaining complex operations (e.g., SMOTE k-neighbors calculation, fingerprint generation)

AI served as a learning resource for unfamiliar libraries and techniques:

- RDKit: Understanding molecular descriptor calculation, fingerprint generation, and SMILES processing
- imbalanced-learn: Learning SMOTE implementation and integration with sklearn pipelines
- scikit-learn pipelines: Understanding Pipeline vs ImbPipeline and proper transformer chaining
- File format libraries: Learning python-docx, python-pptx, and openpyxl for artifact generation

Example: When implementing Morgan fingerprints, the AI provided explanations of radius parameters, bit length considerations, and best practices for molecular similarity calculations.


