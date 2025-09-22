# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib

# # Load saved ML model
# model = joblib.load("rain_model.pkl")

# # Create Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS

# @app.route('/', methods=['POST'])
# def predict():
#     data = request.get_json()
#     roof_area = data['roof_area']
#     rainfall = data['rainfall']
#     runoff = data['runoff']

#     prediction = model.predict([[roof_area, rainfall, runoff]])[0]
#     return jsonify({"predicted_water_liters": prediction})

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

# Load ML model
model = joblib.load("rain_model.pkl")

app = Flask(__name__)
CORS(app)

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')  # serves index.html

# Predict API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    roof_area = float(data['roof_area'])
    rainfall = float(data['rainfall'])
    runoff = float(data['runoff'])

    prediction = model.predict([[roof_area, rainfall, runoff]])[0]
    return jsonify({"predicted_water_liters": float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
