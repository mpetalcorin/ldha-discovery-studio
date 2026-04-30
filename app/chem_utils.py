import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, Draw
from mordred import Calculator, descriptors


calc = Calculator(descriptors, ignore_3D=True)


def validate_smiles(smiles: str):
    """Return RDKit molecule if SMILES is valid."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol


def basic_properties(smiles: str):
    """Calculate basic drug-likeness properties."""
    mol = validate_smiles(smiles)

    if mol is None:
        return None

    props = {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "QED": QED.qed(mol),
    }

    props["Lipinski Pass"] = (
        props["Molecular Weight"] <= 500
        and props["LogP"] <= 5
        and props["H-Bond Donors"] <= 5
        and props["H-Bond Acceptors"] <= 10
    )

    return props


def calculate_mordred_descriptors(smiles: str):
    """Calculate Mordred descriptors for a single molecule."""
    mol = validate_smiles(smiles)

    if mol is None:
        return None

    desc = calc(mol)
    df = pd.DataFrame([desc.asdict()])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def mol_to_image(smiles: str):
    """Generate molecule image."""
    mol = validate_smiles(smiles)

    if mol is None:
        return None

    return Draw.MolToImage(mol, size=(400, 300))
