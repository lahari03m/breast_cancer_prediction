import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

data = load_breast_cancer()
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

names = list(data.feature_names)
X = data.data
y = data.target                      # 0 = malignant, 1 = benign
ben_avg = X[y == 1].mean(axis=0)
mal_avg = X[y == 0].mean(axis=0)
all_std = X.std(axis=0)

# ---------- Plain-English meaning of each measurement ----------
MEANING = {
    "radius": "Average distance from the centre of the cell nucleus to its edge (size).",
    "texture": "Variation in gray-scale pixel values (how uneven the nucleus looks).",
    "perimeter": "Length of the boundary around the nucleus.",
    "area": "Area covered by the nucleus.",
    "smoothness": "Local variation in radius lengths (how smooth or bumpy the edge is).",
    "compactness": "Perimeter² / area − 1 (how tightly packed the shape is).",
    "concavity": "Severity of the inward dents in the nucleus outline.",
    "concave points": "Number of inward dents in the outline.",
    "symmetry": "How symmetrical the nucleus shape is.",
    "fractal dimension": "'Coastline approximation': how rough or jagged the outline is.",
}
GROUP = {"mean": "Average over all cells in the image",
         "error": "Variation (standard error) across the cells",
         "worst": "Average of the 3 largest / most extreme cells"}

def describe(feat):
    if feat.startswith("mean "):
        base, grp = feat[5:], "mean"
    elif feat.startswith("worst "):
        base, grp = feat[6:], "worst"
    else:
        base, grp = feat.replace(" error", ""), "error"
    return f"{MEANING[base]}  [{grp}: {GROUP[grp]}]"

# ---------- Header ----------
st.title("🎗️ Breast Cancer Prediction")
st.caption("Educational demo – NOT a medical diagnostic tool.")

with st.expander("ℹ️ How this app works (read first)", expanded=True):
    st.markdown("""
**Where the numbers come from.** A doctor takes a tiny sample of a breast lump with a fine needle,
photographs the cells under a microscope, and software measures the **cell nuclei**.
Each patient therefore has **10 measurements**, each summarised in **3 ways**, giving **30 inputs**:

| Group | Meaning |
|---|---|
| **mean** | the average value across all nuclei in the image |
| **error** | how much that value varies from cell to cell |
| **worst** | the average of the 3 most extreme (largest) nuclei |

**What the model learned.** It was trained on 569 real patients (357 benign, 212 malignant) and learned that
**malignant cells tend to be bigger, more irregular and more dented** than benign ones.
So to move the prediction towards **Malignant**, increase *radius, perimeter, area, concavity and concave points*
(especially the **worst** values). To move it towards **Benign**, decrease them.

**How to use it.**
1. Pick a starting point from the dropdown (typical benign, typical malignant, or the default average).
2. Change any values. Hover over the **?** next to a box to see the benign and malignant averages.
3. Click **Predict**, then read the explanation under the result.
""")

# ---------- Presets ----------
PRESETS = {
    "Dataset average (default)": X.mean(axis=0),
    "Typical BENIGN patient (average of benign group)": ben_avg,
    "Typical MALIGNANT patient (average of malignant group)": mal_avg,
    "Real malignant example (patient #1 of dataset)": X[0],
    "Real benign example (patient #20 of dataset)": X[np.where(y == 1)[0][0]],
}

def apply_preset():
    vals = PRESETS[st.session_state["preset"]]
    for i in range(30):
        st.session_state[f"f{i}"] = float(vals[i])

if "f0" not in st.session_state:
    for i in range(30):
        st.session_state[f"f{i}"] = float(X[:, i].mean())

st.selectbox("Starting point", list(PRESETS), key="preset", on_change=apply_preset)

# ---------- Reference table ----------
with st.expander("📋 Reference: what each feature means and typical values"):
    ref = pd.DataFrame({
        "Feature": names,
        "Meaning": [describe(n) for n in names],
        "Benign avg": np.round(ben_avg, 4),
        "Malignant avg": np.round(mal_avg, 4),
        "Higher value pushes towards": ["Malignant" if m > b else "Benign"
                                        for m, b in zip(mal_avg, ben_avg)],
    })
    st.dataframe(ref, hide_index=True)

# ---------- Inputs ----------
st.subheader("Enter the measurements")
tabs = st.tabs(["Mean values", "Error values", "Worst values"])
groups = {0: range(0, 10), 1: range(10, 20), 2: range(20, 30)}
for t, idx in zip(tabs, groups.values()):
    with t:
        cols = st.columns(3)
        for k, i in enumerate(idx):
            cols[k % 3].number_input(
                names[i], key=f"f{i}", format="%.4f", step=float(all_std[i]) / 10,
                help=(f"{describe(names[i])}\n\n"
                      f"Benign average: {ben_avg[i]:.4f}\n\n"
                      f"Malignant average: {mal_avg[i]:.4f}"))

# ---------- Predict ----------
if st.button("Predict", type="primary"):
    vals = np.array([st.session_state[f"f{i}"] for i in range(30)])
    pred = model.predict(scaler.transform(vals.reshape(1, -1)))[0]

    if pred == 0:
        st.error("### Prediction: MALIGNANT")
    else:
        st.success("### Prediction: BENIGN")

    # Which side is each value closer to? (distance measured in standard deviations)
    d_ben = np.abs(vals - ben_avg) / all_std
    d_mal = np.abs(vals - mal_avg) / all_std
    closer_mal = d_mal < d_ben
    n_mal = int(closer_mal.sum())
    st.write(f"**{n_mal} of 30** values are closer to the typical malignant value, "
             f"and **{30 - n_mal} of 30** are closer to the typical benign value.")

    strength = (d_ben - d_mal)                       # + = malignant-like, - = benign-like
    order = np.argsort(-np.abs(strength))[:8]
    expl = pd.DataFrame({
        "Feature": [names[i] for i in order],
        "Your value": [round(vals[i], 4) for i in order],
        "Benign avg": [round(ben_avg[i], 4) for i in order],
        "Malignant avg": [round(mal_avg[i], 4) for i in order],
        "Looks like": ["Malignant" if strength[i] > 0 else "Benign" for i in order],
    })
    st.write("**The 8 values that look most clearly malignant or benign:**")
    st.dataframe(expl, hide_index=True)

    st.info("This is a simple comparison with group averages to help you understand the inputs. "
            "The actual model (SVM) combines all 30 values together, so it can disagree with "
            "this summary when the values are mixed.")