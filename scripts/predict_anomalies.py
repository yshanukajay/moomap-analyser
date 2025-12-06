from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
import os

from app.ai.preprocessing import docs_to_dataframe, add_derived_features, prepare_features

# ----------------------
# Configuration
# ----------------------
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://admin:A9fT3xPq@213.199.51.193:27017/moomap?authSource=admin')
DB_NAME = os.getenv('DB_NAME', 'moomap')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'dummy_data_CSV_unlabeled')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/isolation_forest_model.joblib')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.joblib')

# ----------------------
# Helpers
# ----------------------
def connect_db(uri=MONGO_URI):
    client = MongoClient(uri)
    return client

def fetch_collection_documents(client, db_name, collection_name, limit=None):
    db = client[db_name]
    coll = db[collection_name]
    cursor = coll.find().sort('_id', -1)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)

def docs_to_dataframe(docs):
    rows = []
    for d in docs:
        gps = d.get('gps', {})
        battery = d.get('battery', {})
        lat = gps.get('lat', np.nan)
        lon = gps.get('lon', np.nan)
        percent = battery.get('percent', np.nan)
        voltage = battery.get('voltage', np.nan)
        ts = battery.get('ts_ms', None)
        rows.append({
            'device_id': d.get('device_id'),
            'lat': float(lat) if lat is not None else np.nan,
            'lon': float(lon) if lon is not None else np.nan,
            'battery_percent': float(percent) if percent is not None else np.nan,
            'battery_voltage': float(voltage) if voltage is not None else np.nan,
            'ts_ms': int(ts) if ts is not None else None
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=['lat', 'lon', 'battery_percent', 'battery_voltage'])
    df = df.sort_values(['device_id', 'ts_ms']).reset_index(drop=True)
    return df

def add_derived_features(df):
    df['speed_m_s'] = np.nan
    df['battery_drop_per_s'] = 0.0
    for dev, group in df.groupby('device_id'):
        group = group.sort_values('ts_ms')
        speeds, drops = [], []
        prev = None
        for _, row in group.iterrows():
            if prev is None:
                speeds.append(0.0)
                drops.append(0.0)
            else:
                prev_coord = (prev['lat'], prev['lon'])
                cur_coord = (row['lat'], row['lon'])
                try:
                    dist_m = geodesic(prev_coord, cur_coord).meters
                except:
                    dist_m = 0.0
                dt = max(1, (row['ts_ms'] - prev['ts_ms']) / 1000.0)
                speed = dist_m / dt
                battery_drop = max(0.0, prev['battery_percent'] - row['battery_percent']) / dt
                speeds.append(speed)
                drops.append(battery_drop)
            prev = row
        df.loc[group.index, 'speed_m_s'] = speeds
        df.loc[group.index, 'battery_drop_per_s'] = drops
    df['speed_m_s'] = df['speed_m_s'].replace([np.inf, -np.inf], 0.0)
    df['battery_drop_per_s'] = df['battery_drop_per_s'].replace([np.inf, -np.inf], 0.0)
    return df

def prepare_features(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['lat', 'lon', 'battery_percent', 'battery_voltage', 'speed_m_s', 'battery_drop_per_s']
    X = df[feature_cols].values
    return X, feature_cols

# ----------------------
# Main Prediction Flow
# ----------------------
if __name__ == '__main__':
    client = connect_db()
    print("Connected to MongoDB")

    docs = fetch_collection_documents(client, DB_NAME, COLLECTION_NAME, limit=500)
    print(f"Fetched {len(docs)} documents from {COLLECTION_NAME}")

    df = docs_to_dataframe(docs)
    df = add_derived_features(df)
    X, feature_cols = prepare_features(df)

    # Load model and scaler
    model_data = joblib.load(MODEL_PATH)
    clf = model_data['model']
    scaler = joblib.load(SCALER_PATH)

    X_scaled = scaler.transform(X)
    preds = clf.predict(X_scaled)  # -1 anomaly, 1 normal

    df['prediction'] = np.where(preds == -1, 'anomalous', 'normal')
    df['anomaly_score'] = clf.decision_function(X_scaled)

    print(df[['device_id', 'battery_percent', 'battery_voltage', 'prediction', 'anomaly_score']].head(20))
