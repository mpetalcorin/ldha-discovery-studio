import streamlit as st
import pandas as pd
import plotly.express as px

from app.chem_utils import (
    validate_smiles,
    basic_properties,
    mol_to_image,
)

from app.model_utils import (
    load_or_train_model,
    build_feature_frame,
    predict_ldha_probability,
)


st.set_page_config(
    page_title="LDHA Inhibitor Discovery Studio",
    page_icon="🧬",
    layout="wide",
)

st.title("aAidea LDHA Discovery Studio")
st.caption(
    "AI-guided screening of candidate LDHA inhibitors for cancer metabolism research."
)

st.markdown(
    """
This app is inspired by an end-to-end LDHA inhibitor discovery workflow using molecular descriptors,
machine learning, drug-likeness filtering, model interpretation, and generative chemistry.

It is intended for research and educational use only. Computational predictions require experimental validation.
"""
)

model, feature_names = load_or_train_model()

with st.sidebar:
    st.header("Example SMILES")

    examples = {
        "Oxamate": "NC(=O)C(=O)O",
        "Gossypol-like fragment": "COc1cc(O)c(C=O)c(O)c1",
        "Small aromatic acid": "O=C(O)c1ccccc1O",
        "Custom": "",
    }

    selected = st.selectbox("Choose example", list(examples.keys()))

    st.markdown("---")
    st.markdown("### Workflow")
    st.markdown(
        """
1. Enter SMILES
2. Validate molecule
3. Calculate properties
4. Predict LDHA inhibition
5. Rank candidate potential
"""
    )

default_smiles = examples[selected]

smiles = st.text_input(
    "Enter SMILES string",
    value=default_smiles,
    placeholder="Example: NC(=O)C(=O)O",
)

if smiles:
    mol = validate_smiles(smiles)

    if mol is None:
        st.error("Invalid SMILES. Please enter a valid chemical structure.")
    else:
        props = basic_properties(smiles)
        X = build_feature_frame(props, feature_names)
        probability = predict_ldha_probability(model, X)

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("Molecular Structure")
            img = mol_to_image(smiles)
            st.image(img)

        with col2:
            st.subheader("LDHA Inhibition Prediction")

            st.metric(
                label="Predicted LDHA inhibition probability",
                value=f"{probability:.3f}",
            )

            if probability >= 0.80:
                st.success("High-priority predicted LDHA inhibitor candidate.")
            elif probability >= 0.50:
                st.warning("Moderate-priority candidate. Further filtering is recommended.")
            else:
                st.info("Low predicted LDHA inhibition probability.")

            st.markdown("### Drug-likeness Summary")

            prop_df = pd.DataFrame(
                [{"Property": k, "Value": v} for k, v in props.items()]
            )

            st.dataframe(prop_df, use_container_width=True)

        st.markdown("---")

        st.subheader("Property Profile")

        numeric_props = {
            k: v for k, v in props.items()
            if isinstance(v, (int, float)) and k != "Lipinski Pass"
        }

        chart_df = pd.DataFrame({
            "Property": list(numeric_props.keys()),
            "Value": list(numeric_props.values()),
        })

        fig = px.bar(
            chart_df,
            x="Property",
            y="Value",
            title="Calculated Molecular Properties",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("Candidate Decision")

        if props["Lipinski Pass"] and props["QED"] >= 0.5 and probability >= 0.75:
            st.success(
                "This molecule passes basic drug-likeness filters and has a strong predicted LDHA inhibition score."
            )
        elif probability >= 0.75 and not props["Lipinski Pass"]:
            st.warning(
                "This molecule has a strong predicted score but fails one or more Lipinski filters."
            )
        elif props["Lipinski Pass"] and probability < 0.75:
            st.info(
                "This molecule appears drug-like but has a lower predicted LDHA inhibition score."
            )
        else:
            st.error(
                "This molecule is not currently a strong candidate based on the demo model."
            )

else:
    st.info("Enter a SMILES string to begin.")

st.markdown("---")
st.caption(
    "Disclaimer: This app uses a local demo model unless replaced with a trained production model. "
    "Predictions are not medical advice and must be experimentally validated."
)
