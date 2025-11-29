from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize a Flask application instance
app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

# Load the scaler
scaler = joblib.load('standard_scaler.joblib')

print("Model and Scaler loaded successfully!")



@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request body
    data = request.get_json(force=True)

    # Convert the received data to a DataFrame
    try:
        # Ensure the order of columns matches the training data
        # Features are: 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        input_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_df = pd.DataFrame([data], columns=input_features)
    except Exception as e:
        return jsonify({"error": f"Invalid input data format: {e}"}), 400

    # Scale the input features
    scaled_input = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(scaled_input)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
