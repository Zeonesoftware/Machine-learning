from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# ADD THIS ROOT ROUTE
@app.route('/')
def home():
    return jsonify({
        "message": "Machine Learning API is running!",
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Health check",
            "/predict": "POST - Make predictions"
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "features": [1.0, 2.0, 3.0, 4.0]
            }
        }
    })

# Health check endpoint
@app.route('/health')
def health():
    status = "healthy" if model is not None and scaler is not None else "unhealthy"
    return jsonify({"status": status}), 200

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Please provide features in the request body'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
