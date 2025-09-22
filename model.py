import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Load dataset
try:
    df = pd.read_csv("rain.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("rain.csv", encoding="latin-1")

# Step 2: Clean numeric columns (remove commas, convert to float)
for col in ["Roof_Area", "Rainfall", "Runoff_Coeff", "Water_Collected"]:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

# Step 3: Features (X) and Target (y)
X = df[["Roof_Area", "Rainfall", "Runoff_Coeff"]]
y = df["Water_Collected"]

# Step 4: Train ML model (Linear Regression)
model = LinearRegression()
model.fit(X, y)

# Step 5: Test prediction
sample_input = [[120, 900, 0.85]]  # Example input
prediction = model.predict(sample_input)
print("Predicted Water Collected:", prediction[0])

# Step 6: Save trained model
joblib.dump(model, "rain_model.pkl")
print("Model saved as rain_model.pkl")
