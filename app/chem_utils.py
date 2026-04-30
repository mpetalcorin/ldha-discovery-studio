import re
from PIL import Image, ImageDraw, ImageFont


def validate_smiles(smiles: str):
    """
    Lightweight SMILES validation for deployment MVP.
    This does not replace RDKit validation, but avoids deployment failure on Python 3.14.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#@+-\\/%.")
    if not all(ch in allowed for ch in smiles):
        return None

    if len(smiles.strip()) < 2:
        return None

    return smiles.strip()


def count_atoms(smiles: str, atom: str):
    pattern = rf"(?<![a-z]){atom}(?![a-z])"
    return len(re.findall(pattern, smiles))


def basic_properties(smiles: str):
    """
    Approximate molecular properties from SMILES text.
    This is a deployable MVP fallback. Replace with RDKit properties for the full scientific version.
    """
    clean = validate_smiles(smiles)

    if clean is None:
        return None

    c_count = clean.count("C") + clean.count("c")
    n_count = clean.count("N") + clean.count("n")
    o_count = clean.count("O") + clean.count("o")
    s_count = clean.count("S") + clean.count("s")
    f_count = clean.count("F")
    cl_count = clean.count("Cl")
    br_count = clean.count("Br")
    ring_count = sum(ch.isdigit() for ch in clean)
    aromatic_count = sum(1 for ch in clean if ch in "cnops")

    molecular_weight = (
        c_count * 12.01
        + n_count * 14.01
        + o_count * 16.00
        + s_count * 32.06
        + f_count * 19.00
        + cl_count * 35.45
        + br_count * 79.90
        + max(4, c_count * 1.5)
    )

    h_donors = n_count + o_count
    h_acceptors = n_count + o_count + s_count
    logp = 0.54 * c_count + 0.28 * aromatic_count + 0.5 * cl_count + 0.7 * br_count - 1.5 * (n_count + o_count)
    tpsa = 12.0 * n_count + 17.0 * o_count + 25.0 * s_count
    rotatable_bonds = max(0, clean.count("C") - ring_count)
    qed = max(0.05, min(0.95, 0.8 - abs(logp - 2.5) * 0.08 - max(0, molecular_weight - 500) * 0.001))

    props = {
        "Molecular Weight": round(molecular_weight, 2),
        "LogP": round(logp, 2),
        "H-Bond Donors": int(h_donors),
        "H-Bond Acceptors": int(h_acceptors),
        "TPSA": round(tpsa, 2),
        "Rotatable Bonds": int(rotatable_bonds),
        "QED": round(qed, 3),
    }

    props["Lipinski Pass"] = (
        props["Molecular Weight"] <= 500
        and props["LogP"] <= 5
        and props["H-Bond Donors"] <= 5
        and props["H-Bond Acceptors"] <= 10
    )

    return props


def calculate_mordred_descriptors(smiles: str):
    return None


def mol_to_image(smiles: str):
    """
    Create a simple placeholder molecule card.
    Full molecular rendering requires RDKit, which is disabled in this Streamlit Cloud MVP.
    """
    img = Image.new("RGB", (700, 320), "white")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((20, 20, 680, 300), radius=20, outline=(40, 40, 40), width=3)
    draw.text((50, 55), "SMILES structure", fill=(0, 0, 0))
    draw.text((50, 105), smiles[:80], fill=(0, 0, 0))
    draw.text((50, 170), "RDKit molecular drawing disabled in Streamlit Cloud MVP", fill=(80, 80, 80))
    draw.text((50, 215), "Use the Render/Docker version for full cheminformatics rendering.", fill=(80, 80, 80))

    return img
