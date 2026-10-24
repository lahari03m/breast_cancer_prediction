import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

data = load_breast_cancer()
names = list(data.feature_names)
x = data.data
y = data.target                  # 0 = malignant, 1 = benign
ben_avg = x[y == 1].mean(axis=0)
mal_avg = x[y == 0].mean(axis=0)

st.title("Breast Cancer Prediction")
st.write("Enter the 30 tumor measurements and click **Predict**. "
         "Malignant tumors usually have larger radius, perimeter, area, concavity and concave points.")

# starting values for the input boxes
presets = {
    "Average values": x.mean(axis=0),
    "Typical benign patient": ben_avg,
    "Typical malignant patient": mal_avg,
}

def set_values():
    for i in range(30):
        st.session_state[f"f{i}"] = float(presets[st.session_state["preset"]][i])

if "f0" not in st.session_state:
    for i in range(30):
        st.session_state[f"f{i}"] = float(x[:, i].mean())

st.selectbox("Start with:", list(presets), key="preset", on_change=set_values)

# input boxes: mean / error / worst in 3 tabs
tabs = st.tabs(["Mean", "Error", "Worst"])
for t, start in zip(tabs, [0, 10, 20]):
    with t:
        cols = st.columns(3)
        for k, i in enumerate(range(start, start + 10)):
            cols[k % 3].number_input(
                names[i], key=f"f{i}", format="%.4f",
                help=f"Benign avg: {ben_avg[i]:.4f} | Malignant avg: {mal_avg[i]:.4f}")

# prediction
if st.button("Predict", type="primary"):
    values = np.array([st.session_state[f"f{i}"] for i in range(30)])
    pred = model.predict(scaler.transform(values.reshape(1, -1)))[0]

    if pred == 0:
        st.error("# Malignant")
    else:
        st.success("# Benign")

    # a few key features compared with the group averages
    key = ["mean radius", "mean area", "mean concavity", "worst radius", "worst concave points"]
    idx = [names.index(k) for k in key]
    st.write("Key features compared with the dataset averages:")
    st.dataframe(pd.DataFrame({
        "Feature": key,
        "Your value": [round(values[i], 4) for i in idx],
        "Benign avg": [round(ben_avg[i], 4) for i in idx],
        "Malignant avg": [round(mal_avg[i], 4) for i in idx],
    }), hide_index=True)

st.caption("Educational project only, not for medical use.")
