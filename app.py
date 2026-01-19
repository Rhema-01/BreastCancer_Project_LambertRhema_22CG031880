import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'breast_cancer_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

# --- INITIALIZE VARIABLES ---
model = None
scaler = None

# --- LOAD ASSETS ---
try:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ Success: Model and Scaler loaded!")
    else:
        print("❌ Error: .pkl files missing from the 'model' folder!")
except Exception as e:
    print(f"❌ Load Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # This check prevents the 'not defined' error if loading failed
    if model is None or scaler is None:
        return jsonify({'error': 'Model or Scaler not initialized. Check server logs.'}), 500

    try:
        data = request.json.get('data')
        features = np.array([data])
        
        # This is where the error was happening
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        
        result = "Malignant" if prediction[0] == 0 else "Benign"
        color = "#e11d48" if prediction[0] == 0 else "#059669"
        
        return jsonify({'prediction': result, 'color': color})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)