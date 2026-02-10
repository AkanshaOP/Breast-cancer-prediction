
import joblib
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

# Load data to check ground truth
data = load_breast_cancer()
print("Target Names:", data.target_names)
# Usually ['malignant', 'benign']
# 0 = malignant, 1 = benign

# Load saved artifacts
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

print("\nFeatures used:", features)

# Test with a known Malignant case (Target = 0)
# Find first index where target is 0
malignant_idx = np.where(data.target == 0)[0][0]
malignant_sample = data.data[malignant_idx]
malignant_df = pd.DataFrame([malignant_sample], columns=data.feature_names)
# Filter to only used features
malignant_input = malignant_df[features]

print(f"\nTesting Known Malignant Case (Index {malignant_idx}):")
print("Input Values:", malignant_input.iloc[0].to_dict())
# Scale
malignant_scaled = scaler.transform(malignant_input)
# Predict
pred = model.predict(malignant_scaled)
print(f"Prediction: {pred[0]} (Should be 0 for Malignant)")

# Test with a known Benign case (Target = 1)
benign_idx = np.where(data.target == 1)[0][0]
benign_sample = data.data[benign_idx]
benign_df = pd.DataFrame([benign_sample], columns=data.feature_names)
benign_input = benign_df[features]

print(f"\nTesting Known Benign Case (Index {benign_idx}):")
print("Input Values:", benign_input.iloc[0].to_dict())
benign_scaled = scaler.transform(benign_input)
pred_benign = model.predict(benign_scaled)
print(f"Prediction: {pred_benign[0]} (Should be 1 for Benign)")
