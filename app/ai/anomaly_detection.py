import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from app.ai.preprocessing import prepare_features

MODEL_PATH = "app/ai/model_anomaly.pkl"

def train_anomaly_model(device_ids, limit=1000):
    """
    Train Anomaly Detection model using Isolation Forest.
    device_ids -> list of all device_ids
    """
    all_data = []

    for dev in device_ids:
        df = prepare_features(dev, limit)
        if df.empty:
            continue
        features = df[["speed_m_s", "distance_m", "battery_percent"]].values
        all_data.append(features)

    if not all_data:
        return {"success": False, "message": "No data to train model"}

    X = np.vstack(all_data)

    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(X)

    joblib.dump(model, MODEL_PATH)

    return {"success": True, "message": "Anomaly model trained", "samples": len(X)}

def detect_anomalies(device_id, limit=100):
    """
    Run anomaly detection on new data
    """
    from os.path import exists
    if not exists(MODEL_PATH):
        return {"success": False, "error": "Model not trained"}

    df = prepare_features(device_id, limit)
    if df.empty:
        return {"success": False, "error": "No data found"}

    model = joblib.load(MODEL_PATH)
    X = df[["speed_m_s", "distance_m", "battery_percent"]].values

    df["anomaly"] = model.predict(X)  # -1 = anomaly, 1 = normal

    anomalies = df[df["anomaly"] == -1]

    return {
        "success": True,
        "device_id": device_id,
        "total_points": len(df),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies.to_dict(orient="records")
    }
