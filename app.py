import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model, label encoder, and scaler
model = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['N'], data['P'], data['K'], data['ph']]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prediction_decoded = label_encoder.inverse_transform(prediction)
    return jsonify({'prediction': prediction_decoded[0]})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        df['ph'] = df['ph'].astype(float)  # Ensure ph is treated as float
        features = df[['N', 'P', 'K', 'ph']]
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        predictions_decoded = label_encoder.inverse_transform(predictions)
        return jsonify({'predictions': predictions_decoded.tolist()})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
