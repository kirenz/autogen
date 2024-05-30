# filename: housing_regression_with_scaler.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Download and Load the Data
url = 'https://raw.githubusercontent.com/kirenz/data-science-projects/master/end-to-end/datasets/housing/housing.csv'
housing_data = pd.read_csv(url)

# Step 2: Explore the Data
print(housing_data.info())
print(housing_data.describe())
print(housing_data.isnull().sum())

# Step 3: Preprocess the Data
# Handle missing values
housing_data = housing_data.dropna(subset=['total_bedrooms'])
median_value = housing_data['total_bedrooms'].median()
housing_data['total_bedrooms'] = housing_data['total_bedrooms'].fillna(median_value)

# Convert categorical variables to numerical ones using one-hot encoding
housing_data = pd.get_dummies(housing_data, columns=["ocean_proximity"])

# Separate features and target variable
X = housing_data.drop("median_house_value", axis=1)
y = housing_data["median_house_value"]

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale features and Train the Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the Model and Scaler
joblib.dump(model, "housing_model_with_scaler.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model training and evaluation complete. The model and scaler are saved.")