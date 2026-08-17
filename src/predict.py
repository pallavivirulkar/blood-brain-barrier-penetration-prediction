import json
from pathlib import Path
import joblib
from src.features import featurize_smiles


MODEL_PATH = Path("models/bbbp_model.joblib")


def predict_smiles(smiles: str):
    model = joblib.load(MODEL_PATH)
    vector, descriptors, mol = featurize_smiles(smiles)

    if vector is None:
        raise ValueError("Invalid SMILES. RDKit could not parse the molecular structure.")

    probability = float(model.predict_proba([vector])[0, 1])
    label = int(probability >= 0.5)

    return {
        "label": label,
        "class": "BBB+" if label == 1 else "BBB-",
        "probability_BBB_positive": probability,
        "descriptors": descriptors,
    }


if __name__ == "__main__":
    import sys
    smiles = sys.argv[1] if len(sys.argv) > 1 else "CCO"
    print(predict_smiles(smiles))
