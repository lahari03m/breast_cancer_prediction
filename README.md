# Breast Cancer Prediction (ML + Streamlit)

A machine learning web app that predicts whether a breast tumor is **benign** or **malignant**
from 30 cell-nucleus measurements, using the Wisconsin Diagnostic Breast Cancer dataset.

**Live app:** (https://breastcancerprediction-prj201.streamlit.app/)

## Dataset
- 569 patients (357 benign, 212 malignant)
- 30 numeric features: 10 measurements (radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave points, symmetry, fractal dimension), each given as
  mean, error and worst values
- Source: `sklearn.datasets.load_breast_cancer`
- Labels: `0 = malignant`, `1 = benign`
- No missing values

## Method
1. Loaded the data and split it 80/20 into train (455) and test (114) sets with `random_state=42`
2. Scaled the features with `StandardScaler` (fitted on the training data only)
3. Trained and compared six classification algorithms
4. Saved the best model and the scaler as pickle files
5. Built a Streamlit app that loads them and predicts on new input

## Results
| Algorithm | Accuracy |
|---|---|
| **SVM** | **0.982** |
| Logistic Regression | 0.974 |
| Random Forest | 0.965 |
| Naive Bayes | 0.965 |
| KNN | 0.947 |
| Decision Tree | 0.947 |

Best model: **SVM**, saved as `model.pkl`.

SVM confusion matrix on the test set (114 patients):

|  | Predicted malignant | Predicted benign |
|---|---|---|
| **Actually malignant** | 41 | 2 |
| **Actually benign** | 0 | 71 |

It caught 41 of 43 malignant cases (recall 0.95) with no false alarms. The 2 missed malignant cases
are the more serious kind of error in a medical setting.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## Files
- `app.py` – Streamlit app
- `breast_cancer_prediction_.ipynb` – notebook with data exploration, model comparison and training
- `model.pkl` – trained SVM model
- `requirements.txt` – dependencies
- `scaler.pkl` – fitted StandardScaler (applied to new input before predicting)
  

## How to use the app
1. Choose a starting point from the dropdown (typical benign or malignant patient)
2. Change any of the 30 values (hover over the ? icon for the meaning and typical values)
3. Click **Predict**

Malignant cells tend to be larger and more irregular, so higher radius, perimeter, area,
concavity and concave points (especially the "worst" values) push the prediction towards malignant.

## Disclaimer
Educational project only. This is **not** a medical diagnostic tool and must not be used
for real medical decisions.
