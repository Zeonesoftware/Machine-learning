from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = None
scaler = None

try:
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    else:
        print("❌ Model file not found")
        
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully!")
    else:
        print("❌ Scaler file not found")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    model_status = "✅ Loaded" if model is not None else "❌ Not Loaded"
    scaler_status = "✅ Loaded" if scaler is not None else "❌ Not Loaded"
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Machine Learning API</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }}
            h1 {{
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }}
            .subtitle {{
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }}
            .status-box {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            }}
            .status-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                margin: 5px 0;
                background: white;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }}
            .endpoint {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 15px 0;
                border-left: 4px solid #667eea;
            }}
            .endpoint h3 {{
                color: #667eea;
                margin-bottom: 10px;
            }}
            .method {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 0.9em;
                margin-right: 10px;
            }}
            .get {{
                background: #d4edda;
                color: #155724;
            }}
            .post {{
                background: #fff3cd;
                color: #856404;
            }}
            code {{
                background: #2d2d2d;
                color: #f8f8f2;
                padding: 15px;
                border-radius: 5px;
                display: block;
                margin: 10px 0;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
            }}
            .example {{
                background: #e7f3ff;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            }}
            .example h3 {{
                color: #0066cc;
                margin-bottom: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Machine Learning API</h1>
            <p class="subtitle">Your model is deployed and ready to use!</p>
            
            <div class="status-box">
                <h2>📊 System Status</h2>
                <div class="status-item">
                    <strong>Model:</strong>
                    <span>{model_status}</span>
                </div>
                <div class="status-item">
                    <strong>Scaler:</strong>
                    <span>{scaler_status}</span>
                </div>
            </div>
            
            <h2>🔗 Available Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /</h3>
                <p>API information and documentation (this page)</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /health</h3>
                <p>Check API health status</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /predict</h3>
                <p>Make predictions using the machine learning model</p>
            </div>
            
            <div class="example">
                <h3>💡 Example Usage</h3>
                <p><strong>Using cURL:</strong></p>
                <code>curl -X POST https://machine-learning-1-uo9w.onrender.com/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"features": [1.0, 2.0, 3.0, 4.0]}}'</code>
                
                <p style="margin-top: 20px;"><strong>Using Python:</strong></p>
                <code>import requests

response = requests.post(
    'https://machine-learning-1-uo9w.onrender.com/predict',
    json={{'features': [1.0, 2.0, 3.0, 4.0]}}
)
print(response.json())</code>
                
                <p style="margin-top: 20px;"><strong>Expected Response:</strong></p>
                <code>{{
  "prediction": 1,
  "probability": [0.23, 0.77]
}}</code>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/health')
def health():
    status = "healthy" if model is not None and scaler is not None else "unhealthy"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }), 200 if status == "healthy" else 503

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model or scaler not loaded'
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Please provide features array'}), 400
        
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
