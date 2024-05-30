# filename: make_prediction.py

import joblib
import numpy as np

# Load the saved model
model = joblib.load("housing_model.pkl")

# Example new data point (ensure the data is preprocessed similarly to the training data)
new_data = np.array([[-122.23, 37.88, 41.0, 880, 129.0, 322, 126, 8.3252, 0, 0, 0, 0, 1]])

# Make prediction
predicted_value = model.predict(new_data)
print(f"Predicted Median House Value: {predicted_value[0]}")