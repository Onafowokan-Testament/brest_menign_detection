import streamlit as st
import joblib
from data_utils import load_data
from model_utils import get_model
import numpy as np

# Load pre-trained model (train.py should dump best model, e.g. Random Forest)
MODEL_PATH = 'best_model.pkl'
model = joblib.load(MODEL_PATH)

st.title("Breast Cancer Prediction")
st.write("Enter patient features below:")

# Get feature names and defaults
_, _, _, _, feature_names = load_data('breast_cancer')
inputs = []
for name in feature_names:
    val = st.number_input(name, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    features = np.array(inputs).reshape(1, -1)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0, pred]
    label = 'Malignant' if pred == 0 else 'Benign'
    st.success(f"Prediction: {label} (confidence: {proba:.2f})")