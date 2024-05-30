# filename: make_prediction_with_scaler.py

import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load("housing_model_with_scaler.pkl")
scaler = joblib.load("scaler.pkl")

# Example new data point
# Replace with real feature values; the last 5 values represent one-hot encoded categories for ocean proximity
new_data = np.array([[-122.23, 37.88, 41.0, 880, 129.0, 322, 126, 8.3252, 0, 0, 0, 0, 1]])

# Scale the new data point
new_data_scaled = scaler.transform(new_data)

# Make prediction
predicted_value = model.predict(new_data_scaled)
print(f"Predicted Median House Value: {predicted_value[0]}")