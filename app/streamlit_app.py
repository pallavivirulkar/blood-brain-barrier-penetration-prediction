import sys
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import predict_smiles


st.set_page_config(
    page_title="BBBP Molecular Predictor",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 BBBP Molecular Predictor")
st.caption("Predictive cheminformatics demo: SMILES → molecular features → ML classifier")

smiles = st.text_input(
    "Enter a SMILES string",
    value="CCO",
    help="Example: CCO. The output is a model prediction, not a clinical recommendation.",
)

if st.button("Predict BBB penetration", type="primary"):
    try:
        result = predict_smiles(smiles)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction")
            if result["class"] == "BBB+":
                st.success("Predicted class: BBB+")
            else:
                st.warning("Predicted class: BBB-")

            st.metric(
                "Model probability of BBB+",
                f"{result['probability_BBB_positive']:.1%}",
            )

        with col2:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.image(Draw.MolToImage(mol, size=(420, 300)), caption="2D molecular structure")

        st.subheader("Molecular descriptors")
        st.dataframe(result["descriptors"], use_container_width=True)

        st.info(
            "Interpret the probability as a model output on the benchmark dataset. "
            "It should not be treated as evidence of clinical BBB permeability."
        )

    except Exception as exc:
        st.error(str(exc))

st.divider()
st.caption("Dataset: MoleculeNet BBBP. Model: descriptor + Morgan fingerprint classifier.")
