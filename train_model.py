import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import joblib

# Load data
df = pd.read_excel(r"C:\Users\sree4\OneDrive\Desktop\Project\vehicle_dataset_20000.xlsx")
print(df.head())

#  Convert selling_price to LAKHS (only if data is in rupees)
df["selling_price"] = df["selling_price"] * 1000

# Create age feature
df["age"] = 2025 - df["model_year"]

# Feature selection
numeric_cols = ["km_driven", "engine", "max_power", "torque_nm", "condition_score", "age"]
categorical_cols = ["brand", "car_name", "fuel", "transmission", "owner", "city", "seats", "seller_type"]

X = df[numeric_cols + categorical_cols]
y = df["selling_price"]

# Preprocessing
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Model
model = Pipeline([
    ("prep", preprocess),
    ("lr", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

#  Metrics now in LAKHS
print("MAE :", mean_absolute_error(y_test, y_predict))
print("R² Score:", r2_score(y_test, y_predict))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_predict)))

# Save model
model_filename = "train_model.pkl"
joblib.dump(model, model_filename)



