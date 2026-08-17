from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


DESCRIPTOR_NAMES = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
]

FP_SIZE = 256


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def descriptor_dict(mol) -> dict:
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
    }


def morgan_bits(mol, n_bits: int = FP_SIZE) -> np.ndarray:
    fp = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=n_bits).GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_smiles(smiles: str, n_bits: int = FP_SIZE):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, None, None

    desc = descriptor_dict(mol)
    fp = morgan_bits(mol, n_bits)
    vector = np.concatenate([np.array(list(desc.values()), dtype=float), fp.astype(float)])
    return vector, desc, mol


def build_feature_table(df: pd.DataFrame, n_bits: int = FP_SIZE):
    rows = []
    valid_indices = []

    for idx, smiles in df["smiles"].items():
        vector, desc, mol = featurize_smiles(smiles, n_bits)
        if vector is not None:
            rows.append(vector)
            valid_indices.append(idx)

    X = np.vstack(rows)
    clean = df.loc[valid_indices].copy().reset_index(drop=True)
    y = clean["p_np"].astype(int).to_numpy()
    return clean, X, y


def scaffold(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Approximate DeepChem-style scaffold split using randomized scaffold groups."""
    rng = np.random.default_rng(seed)
    work = df.copy()
    work["_scaffold"] = work["smiles"].map(scaffold)

    groups = list(work.groupby("_scaffold").groups.values())
    rng.shuffle(groups)

    target_test = int(np.ceil(len(work) * test_size))
    test_indices = []
    count = 0

    for group in groups:
        if count >= target_test:
            break
        test_indices.extend(list(group))
        count += len(group)

    test_set = set(test_indices)
    train_indices = [i for i in work.index if i not in test_set]

    return train_indices, test_indices
