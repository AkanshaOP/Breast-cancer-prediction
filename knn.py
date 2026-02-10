
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Feature Selection: Use only 'mean' features for simplicity in the web app
mean_features = [col for col in data.feature_names if 'mean' in col]
X = df[mean_features]
y = df['target']

# Split data with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (Random Forest doesn't strictly need it, but good practice and keeps app structure)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model - Using Random Forest for better robustness than KNN
# n_estimators=100 is standard
# class_weight='balanced' helps with the specific issue of missing malignant cases
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy (Random Forest): {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model and Scaler
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(mean_features, 'features.pkl')

print("Model (Random Forest), Scaler, and Feature names saved successfully.")