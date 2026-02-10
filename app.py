
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
features_list = joblib.load('features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from JSON request
        data = request.json
        
        # Ensure we have all features
        input_data = []
        for feature in features_list:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            try:
                input_data.append(float(data[feature]))
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}'}), 400
        
        # Create DataFrame (single sample)
        df_input = pd.DataFrame([input_data], columns=features_list)
        
        # Scale input
        scaled_input = scaler.transform(df_input)
        
        # Predict
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        
        # Benign (0) or Malignant (1) - Check sklearn target names
        # Usually target 0 is Malignant and 1 is Benign in some datasets, let's verify.
        # However, in sklearn load_breast_cancer:
        # 'malignant' is 0, 'benign' is 1 ? Or vice versa?
        # Let's verify target names
        # target_names: array(['malignant', 'benign'], dtype='<U9')
        # So 0 -> Malignant, 1 -> Benign.
        
        result_text = "Benign (Non-Cancerous)" if prediction == 1 else "Malignant (Cancerous)"
        probability = prediction_proba[prediction] * 100
        
        return jsonify({
            'prediction': result_text,
            'probability': f"{probability:.2f}%",
            'is_malignant': int(prediction) == 0
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
