import pickle
import numpy as np
import streamlit as st
from sklearn.datasets import load_breast_cancer

# load the saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

data = load_breast_cancer()
x = data.data
y = data.target

st.title('Breast Cancer Prediction')
st.write('Enter the tumor measurements below to predict whether it is benign or malignant.')

# choose what the input boxes start with
sample = st.sidebar.radio('Fill the boxes with:',
                          ['Average values', 'Malignant sample', 'Benign sample'])

if sample == 'Malignant sample':
    default = x[y == 0][0]
elif sample == 'Benign sample':
    default = x[y == 1][0]
else:
    default = x.mean(axis=0)

# one input box for each of the 30 features
inputs = []
cols = st.columns(3)
for i, name in enumerate(data.feature_names):
    value = cols[i % 3].number_input(name, value=float(default[i]), format='%.4f')
    inputs.append(value)

if st.button('Predict'):
    features = np.array(inputs).reshape(1, -1)
    features = scaler.transform(features)      # scale the same way as training
    pred = model.predict(features)[0]

    if pred == 0:
        st.error('Prediction: Malignant')
    else:
        st.success('Prediction: Benign')

st.caption('Educational project only, not for medical use.')
