import math
import streamlit as st


st.set_page_config(
    page_title="aAidea LDHA Discovery Studio",
    page_icon="🧬",
    layout="wide",
)

st.title("aAidea LDHA Discovery Studio")
st.caption("AI-guided screening of candidate LDHA inhibitors for cancer metabolism research.")

st.info(
    "This is a stable public MVP for Streamlit Cloud. "
    "It uses lightweight SMILES-based property estimation so the app can run without RDKit, Mordred, LightGBM, or SHAP."
)

st.markdown(
    """
Enter a SMILES string below. The app estimates simple molecular features, evaluates basic drug-likeness,
and produces a research-use LDHA inhibitor prioritisation score.

This is not a medical or clinical tool. Computational predictions require experimental validation.
"""
)

examples = [
    ("Oxamate", "NC(=O)C(=O)O"),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Nicotinamide-like", "NC(=O)c1ccncc1"),
]

st.markdown("### Example SMILES")
for name, smi in examples:
    st.code(f"{name}: {smi}", language="text")


def validate_smiles_text(smiles):
    if not smiles:
        return False

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#@+-\\/%.")
    return all(ch in allowed for ch in smiles.strip()) and len(smiles.strip()) >= 2


def estimate_properties(smiles):
    s = smiles.strip()

    c = s.count("C") + s.count("c")
    n = s.count("N") + s.count("n")
    o = s.count("O") + s.count("o")
    sulfur = s.count("S") + s.count("s")
    fluorine = s.count("F")
    chlorine = s.count("Cl")
    bromine = s.count("Br")
    aromatic = sum(1 for ch in s if ch in "cnops")
    rings = sum(1 for ch in s if ch.isdigit())

    mw = (
        c * 12.01
        + n * 14.01
        + o * 16.00
        + sulfur * 32.06
        + fluorine * 19.00
        + chlorine * 35.45
        + bromine * 79.90
        + max(4.0, c * 1.5)
    )

    hbd = n + o
    hba = n + o + sulfur
    logp = 0.54 * c + 0.28 * aromatic + 0.5 * chlorine + 0.7 * bromine - 1.5 * (n + o)
    tpsa = 12.0 * n + 17.0 * o + 25.0 * sulfur
    rotatable = max(0, s.count("C") - rings)

    qed = 0.8 - abs(logp - 2.5) * 0.08 - max(0, mw - 500) * 0.001
    qed = max(0.05, min(0.95, qed))

    lipinski = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10

    return {
        "Molecular Weight": round(mw, 2),
        "LogP": round(logp, 2),
        "H-Bond Donors": int(hbd),
        "H-Bond Acceptors": int(hba),
        "TPSA": round(tpsa, 2),
        "Rotatable Bonds": int(rotatable),
        "QED-like Score": round(qed, 3),
        "Lipinski Pass": lipinski,
        "Carbon Count": c,
        "Nitrogen Count": n,
        "Oxygen Count": o,
        "Aromatic Character": aromatic,
    }


def ldha_score(props):
    """
    Lightweight heuristic score for MVP deployment.
    The full scientific version should replace this with the trained LightGBM model.
    """
    score = 0.15

    if props["Lipinski Pass"]:
        score += 0.20

    score += min(props["QED-like Score"], 1.0) * 0.25

    if 150 <= props["Molecular Weight"] <= 500:
        score += 0.12

    if 1.0 <= props["LogP"] <= 4.5:
        score += 0.12

    if 30 <= props["TPSA"] <= 140:
        score += 0.08

    if props["Aromatic Character"] >= 3:
        score += 0.05

    if props["H-Bond Acceptors"] >= 2:
        score += 0.03

    return max(0.0, min(0.99, score))


with st.form("smiles_form", clear_on_submit=False):
    smiles = st.text_area(
        "Paste SMILES here",
        value="NC(=O)C(=O)O",
        height=90,
        help="Example: CC(=O)Oc1ccccc1C(=O)O",
    )

    submitted = st.form_submit_button("Analyse molecule")

if submitted:
    smiles = smiles.strip()

    if not validate_smiles_text(smiles):
        st.error("Invalid SMILES-like text. Please check the characters and try again.")
        st.stop()

    props = estimate_properties(smiles)
    probability = ldha_score(props)

    st.markdown("---")
    st.subheader("Input Molecule")
    st.code(smiles, language="text")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted LDHA Prioritisation Score")
        st.metric("LDHA inhibition probability, MVP estimate", f"{probability:.3f}")

        if probability >= 0.80:
            st.success("High-priority candidate for follow-up screening.")
        elif probability >= 0.50:
            st.warning("Moderate-priority candidate. Further filtering is recommended.")
        else:
            st.info("Low-priority candidate in this lightweight MVP model.")

    with col2:
        st.subheader("Decision Summary")

        if props["Lipinski Pass"] and props["QED-like Score"] >= 0.5 and probability >= 0.75:
            st.success("Drug-like profile with strong predicted LDHA prioritisation.")
        elif props["Lipinski Pass"] and probability >= 0.5:
            st.warning("Drug-like profile with moderate predicted LDHA prioritisation.")
        elif not props["Lipinski Pass"]:
            st.error("Fails one or more basic Lipinski-style filters.")
        else:
            st.info("Acceptable simple properties, but weak predicted LDHA prioritisation.")

    st.markdown("### Estimated Molecular Properties")

    rows = []
    for k, v in props.items():
        rows.append(f"| {k} | {v} |")

    table = "\n".join(["| Property | Value |", "|---|---|"] + rows)
    st.markdown(table)

else:
    st.markdown("Click **Analyse molecule** to run the MVP prediction.")

st.markdown("---")
st.caption(
    "Research-use MVP only. The production version should use RDKit, Mordred descriptors, "
    "a trained LightGBM model, SHAP interpretation, and experimental validation."
)
