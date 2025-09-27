import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- 1. Configuration and Data Loading ---
DATA_FILE = 'ev_battery_charging_data.csv'
MODEL_PATH = 'rf_charging_model.pkl'
SCALER_PATH = 'scaler.pkl'

df = pd.read_csv(DATA_FILE)
# Clean column names
df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')

# --- 2. Data Preprocessing (Matching the Final Workflow) ---
target = 'Optimal_Charging_Duration_Class'
cols_to_drop_as_features = ['Charging_Mode', 'Efficiency', 'EV_Model']

# Drop the excluded columns
df_processed = df.drop(columns=cols_to_drop_as_features, errors='ignore')

# Categorical Encoding: Only 'Battery_Type' remains
categorical_cols_to_encode = ['Battery_Type']
df_encoded = pd.get_dummies(df_processed, columns=categorical_cols_to_encode, drop_first=True)

# Define Features (X) and Target (y)
features = df_encoded.drop(columns=[target]).columns.tolist()
X = df_encoded[features]
y = df_encoded[target]

# Split data to get the training set for fitting the scaler and model
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the numerical features (excluding the Battery_Type dummy column)
numerical_features = [col for col in X.columns if not col.startswith('Battery_Type_')]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
# Fit and transform the scaler on the training data
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])

# --- 3. Model Training ---
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
print("Training complete.")

# --- 4. Save Model Artifacts ---
joblib.dump(rf_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")