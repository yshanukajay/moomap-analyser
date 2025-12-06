from app.db import get_db
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------------------
# Fetch data from MongoDB
# ---------------------------
db = get_db()
collection = db['dummy_data_CSV_unlabeled']
docs = list(collection.find())
df = pd.DataFrame(docs)

# ---------------------------
# Preprocessing
# ---------------------------
df['battery_percent'] = df['battery'].apply(lambda x: x.get('percent', 0))
df['battery_voltage'] = df['battery'].apply(lambda x: x.get('voltage', 0))
df['ts_ms'] = df['battery'].apply(lambda x: x.get('ts_ms'))

# Sort by device and timestamp
df = df.sort_values(by=['device_id', 'ts_ms'])

# Target: battery percent after next timestamp
df['target_battery_percent'] = df.groupby('device_id')['battery_percent'].shift(-1)
df = df.dropna(subset=['target_battery_percent'])

# Feature: battery drop rate
df['battery_drop_per_s'] = df.groupby('device_id')['battery_percent'].diff() / df.groupby('device_id')['ts_ms'].diff() * 1000
df['battery_drop_per_s'].fillna(0, inplace=True)

# Replace inf/-inf with 0
df['battery_drop_per_s'].replace([np.inf, -np.inf], 0, inplace=True)

df['battery_drop_per_s'] = df['battery_drop_per_s'].fillna(0)

# Optional: drop rows with missing battery or voltage
df = df.dropna(subset=['battery_percent', 'battery_voltage'])


# ---------------------------
# Features and target
# ---------------------------
feature_cols = ['battery_percent', 'battery_voltage', 'battery_drop_per_s']
X = df[feature_cols]
y = df['target_battery_percent']


# Optional: scale features for stability
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/battery_status_scaler.joblib')

# ---------------------------
# Train model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/battery_predictor_model.joblib')
print("Battery predictor model trained and saved!")
