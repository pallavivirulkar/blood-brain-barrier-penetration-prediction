# BBBP Penetration ML

An end-to-end cheminformatics + machine-learning project for predicting whether a small molecule is likely to exhibit blood-brain barrier (BBB) penetration from its SMILES representation.

## Why this project?

The project turns molecular structure into a reproducible ML prediction pipeline:

**SMILES → RDKit descriptors + Morgan fingerprints → preprocessing → model comparison → evaluation → interpretable prediction**

It is designed as a portfolio project for drug discovery / computational biology / biomedical engineering roles.

> **Scientific scope:** this is a dataset-level binary classifier. It is not a clinical decision tool and does not establish whether a drug will cross the BBB in a patient.

## Dataset

The project uses the MoleculeNet BBBP dataset. The DeepChem BBBP loader documents the dataset as containing over 2,000 compounds with columns `name`, `smiles`, and `p_np`, where `p_np` is the binary penetration label. DeepChem recommends scaffold splitting for BBBP.

Dataset source:
- GLambard Molecules Dataset Collection: https://github.com/GLambard/Molecules_Dataset_Collection
- DeepChem BBBP loader / download source: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv

The repository intentionally does **not** commit the raw dataset; `src/data.py` downloads it automatically.

## Project structure

```text
bbbp-penetration-ml/
├── app/
│   └── streamlit_app.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── BBBP_ML_End_to_End.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Download data and train

```bash
python -m src.train
```

This:
- downloads BBBP if absent
- validates SMILES
- removes missing/duplicate records
- generates RDKit descriptors and Morgan fingerprints
- creates a scaffold-based train/test split
- trains Logistic Regression, Random Forest and Extra Trees
- reports Accuracy, Precision, Recall, F1 and ROC-AUC
- saves the best model to `models/bbbp_model.joblib`
- saves preprocessing metadata and model-comparison results

### 3. Run the interactive predictor

```bash
streamlit run app/streamlit_app.py
```

Enter a SMILES string and the app returns:
- predicted BBB penetration class
- model probability
- molecular descriptors
- a 2D molecule depiction

## Example SMILES

```text
CCO
```

This is intentionally a simple demonstration input, not a claim about BBB penetration of ethanol.

## Modeling choices

### Molecular representation

The pipeline combines:
- Molecular Weight
- LogP
- Topological Polar Surface Area (TPSA)
- H-bond donors
- H-bond acceptors
- Rotatable bonds
- Ring counts
- Fraction C-sp3
- Morgan circular fingerprint bits

This is more informative than using only five physicochemical descriptors.

### Data split

A **scaffold split** is used for the main evaluation because random molecule-level splitting can place structurally similar compounds in both train and test sets. Scaffold splitting provides a more chemically meaningful estimate of generalization to different molecular scaffolds.

For a simple teaching comparison, the notebook also contains an optional stratified 80/20 random split.

## Evaluation

The project reports:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix
- ROC curve
- model comparison

Accuracy alone is not treated as the primary metric.

## Limitations

- BBBP is a relatively small benchmark dataset.
- Labels reflect experimental permeability classification and may contain measurement/annotation noise.
- Molecular descriptors and fingerprints do not encode every biological mechanism involved in BBB transport.
- The model is not validated for clinical or prospective drug-development use.
- External validation on an independent BBB dataset would be required before making stronger claims.

## Future extensions

1. Hyperparameter optimization with nested cross-validation.
2. XGBoost / LightGBM comparison.
3. SHAP or permutation-based explainability.
4. Independent validation using B3DB.
5. Graph neural networks / message-passing neural networks.
6. Applicability-domain / uncertainty estimation.
7. Model card documenting intended use and failure modes.

## Resume-ready line

> Developed an RDKit-based ML pipeline to predict blood-brain barrier penetration from molecular structure, benchmarking descriptor and fingerprint-based classifiers using scaffold-aware evaluation.

## Citation

Wu, Z. et al. MoleculeNet: A Benchmark for Molecular Machine Learning. Chemical Science, 2018.
