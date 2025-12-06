"""
Isolation Forest training script for IoT device data stored in MongoDB.

- Connects to MongoDB collection `dummy_data_CSV_unlabeled` (configurable)
- Extracts features: lat, lon, battery.percent, battery.voltage
- Creates derived features per device: speed (meters/sec) and battery drop rate
- Trains a pipeline: StandardScaler -> IsolationForest
- Saves trained model and scaler to disk (joblib)
- Optional: evaluates using a labeled collection `dummy_data_CSV_labeled` if present

Requirements:
pip install pymongo pandas numpy scikit-learn joblib geopy

Usage:
python isolation_forest_training.py

"""

from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
from geopy.distance import geodesic

from app.ai.preprocessing import docs_to_dataframe, add_derived_features, prepare_features

# ----------------------
# Configuration
# ----------------------
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://admin:A9fT3xPq@213.199.51.193:27017/moomap?authSource=admin')
DB_NAME = os.getenv('DB_NAME', 'moomap')
UNLABELED_COLLECTION = os.getenv('UNLABELED_COLLECTION', 'dummy_data_CSV_unlabeled')
LABELED_COLLECTION = os.getenv('LABELED_COLLECTION', 'dummy_data_CSV_labeled')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/isolation_forest_model.joblib')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.joblib')

# Isolation Forest parameters (tweakable)
IF_PARAMS = {
    'n_estimators': 200,
    'max_samples': 'auto',
    'contamination': 0.05,  # expected fraction of anomalies
    'random_state': 42,
    'n_jobs': -1
}

# ----------------------
# Helpers
# ----------------------

def connect_db(uri=MONGO_URI):
    client = MongoClient(uri)
    return client


def fetch_collection_documents(client, db_name, collection_name, limit=None):
    db = client[db_name]
    coll = db[collection_name]
    cursor = coll.find()
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)


def docs_to_dataframe(docs):
    # Normalize and extract features. Keep device_id and timestamp for derived features.
    rows = []
    for d in docs:
        try:
            dev = d.get('device_id')
            gps = d.get('gps', {})
            battery = d.get('battery', {})
            lat = gps.get('lat', np.nan)
            lon = gps.get('lon', np.nan)
            valid = gps.get('valid', True)
            percent = battery.get('percent', np.nan)
            voltage = battery.get('voltage', np.nan)
            ts = battery.get('ts_ms')
            # if timestamp is tiny (seconds), convert to ms conservatively
            if ts and ts < 1e12:
                ts = int(ts * 1000)

            rows.append({
                'device_id': dev,
                'lat': float(lat) if lat is not None else np.nan,
                'lon': float(lon) if lon is not None else np.nan,
                'valid': bool(valid),
                'battery_percent': float(percent) if percent is not None else np.nan,
                'battery_voltage': float(voltage) if voltage is not None else np.nan,
                'ts_ms': int(ts) if ts is not None else None
            })
        except Exception as e:
            # skip malformed doc
            continue

    df = pd.DataFrame(rows)
    # drop rows missing essential values
    df = df.dropna(subset=['lat', 'lon', 'battery_percent', 'battery_voltage'])
    # ensure ts_ms exists; if not, create incremental timestamps (fallback)
    if df['ts_ms'].isnull().any():
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        df['ts_ms'] = df['ts_ms'].fillna(method='ffill').fillna(now_ms)

    # sort by device and timestamp
    df = df.sort_values(['device_id', 'ts_ms']).reset_index(drop=True)
    return df


def add_derived_features(df):
    # Compute distance and speed between consecutive points per device, and battery drop rate
    df['speed_m_s'] = np.nan
    df['battery_drop_per_s'] = 0.0

    for dev, group in df.groupby('device_id'):
        group = group.sort_values('ts_ms')
        speeds = []
        drops = []
        prev = None
        for _, row in group.iterrows():
            if prev is None:
                speeds.append(0.0)
                drops.append(0.0)
            else:
                # compute geodesic distance (meters)
                prev_coord = (prev['lat'], prev['lon'])
                cur_coord = (row['lat'], row['lon'])
                try:
                    dist_m = geodesic(prev_coord, cur_coord).meters
                except Exception:
                    dist_m = 0.0
                dt = max(1, (row['ts_ms'] - prev['ts_ms']) / 1000.0)
                speed = dist_m / dt
                battery_drop = max(0.0, prev['battery_percent'] - row['battery_percent']) / dt
                speeds.append(speed)
                drops.append(battery_drop)
            prev = row

        df.loc[group.index, 'speed_m_s'] = speeds
        df.loc[group.index, 'battery_drop_per_s'] = drops

    # Replace any inf or extremely large values
    df['speed_m_s'] = df['speed_m_s'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df['battery_drop_per_s'] = df['battery_drop_per_s'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def prepare_features(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['lat', 'lon', 'battery_percent', 'battery_voltage', 'speed_m_s', 'battery_drop_per_s']
    X = df[feature_cols].values
    return X, feature_cols


# ----------------------
# Training
# ----------------------

def train_and_save_model(X, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(**IF_PARAMS)
    clf.fit(X_scaled)

    # Save both
    joblib.dump({'model': clf, 'feature_cols': None}, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    return clf, scaler


# ----------------------
# Evaluation (optional)
# ----------------------
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_on_labeled_collection(client, db_name, labeled_coll_name, clf, scaler, feature_cols):
    db = client[db_name]
    coll = db[labeled_coll_name]
    docs = list(coll.find())
    if len(docs) == 0:
        print('No labeled docs found; skipping evaluation')
        return

    df_lab = docs_to_dataframe(docs)
    df_lab = add_derived_features(df_lab)
    X_lab, _ = prepare_features(df_lab, feature_cols)
    X_lab_scaled = scaler.transform(X_lab)
    scores = clf.decision_function(X_lab_scaled)
    preds = clf.predict(X_lab_scaled)  # -1 anomaly, 1 normal
    preds_label = np.where(preds == -1, 'anomalous', 'normal')

    # Use doc.label if present
    true_labels = [d.get('label', 'normal') for d in docs]

    print('Classification Report (using labeled collection):')
    print(classification_report(true_labels, preds_label, labels=['normal', 'anomalous']))
    print('Confusion Matrix:')
    print(confusion_matrix(true_labels, preds_label, labels=['normal', 'anomalous']))


# ----------------------
# Main flow
# ----------------------
if __name__ == '__main__':
    client = connect_db()
    print('Connected to MongoDB')

    docs = fetch_collection_documents(client, DB_NAME, UNLABELED_COLLECTION)
    print(f'Fetched {len(docs)} documents from {UNLABELED_COLLECTION}')

    df = docs_to_dataframe(docs)
    df = add_derived_features(df)
    X, feature_cols = prepare_features(df)

    print('Training Isolation Forest...')
    clf, scaler = train_and_save_model(X, MODEL_PATH, SCALER_PATH)

    # Save feature columns alongside the model for future use
    joblib.dump({'model': clf, 'feature_cols': feature_cols}, MODEL_PATH)

    # Optional evaluation if labeled collection exists
    if LABELED_COLLECTION in client[DB_NAME].list_collection_names():
        print('Labeled collection found — running evaluation')
        evaluate_on_labeled_collection(client, DB_NAME, LABELED_COLLECTION, clf, scaler, feature_cols)
    else:
        print('No labeled collection found; skip evaluation')

    print('Done')
