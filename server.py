from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ Ensure model exists before loading
model_path = "/app/models/iris_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
else:
    print("🚨 Model file not found!")
    model = None

@app.route('/predict', methods=['POST'])  # ✅ Use POST instead of GET
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Missing 'input' key in JSON request"}), 400

        input_data = np.array(data["input"]).reshape(1, -1)
        prediction = model.predict(input_data)

        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return 'Welcome to the prediction API!'

# ✅ Fix for `GET /favicon.ico 404`
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return empty response to avoid 404 errors

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
