from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load saved ML model
model = joblib.load("rain_model.pkl")

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    roof_area = data['roof_area']
    rainfall = data['rainfall']
    runoff = data['runoff']

    prediction = model.predict([[roof_area, rainfall, runoff]])[0]
    return jsonify({"predicted_water_liters": prediction})

if __name__ == '__main__':
    app.run(debug=True)
