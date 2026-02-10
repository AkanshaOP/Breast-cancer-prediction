
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Feature Selection: Use only 'mean' features for simplicity in the web app
# This reduces inputs from 30 to 10, making the form user-friendly while maintaining good accuracy.
mean_features = [col for col in data.feature_names if 'mean' in col]
X = df[mean_features]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy (using mean features): {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model and Scaler
joblib.dump(knn, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(mean_features, 'features.pkl')

print("Model, Scaler, and Feature names saved successfully.")