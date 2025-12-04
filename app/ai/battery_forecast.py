import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from app.ai.preprocessing import prepare_features

BATTERY_MODEL_PATH = "app/ai/model_battery.pkl"

def train_battery_model(device_ids, limit=1000):
    X_all, y_all = [], []

    for dev in device_ids:
        df = prepare_features(dev, limit)
        if df.empty:
            continue

        df["t"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
        X_all.append(df["t"].values.reshape(-1, 1))
        y_all.append(df["battery_percent"].values)

    if not X_all:
        return {"success": False, "message": "No data"}

    X = np.vstack(X_all)
    y = np.hstack(y_all)

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, BATTERY_MODEL_PATH)

    return {"success": True, "message": "Battery prediction model trained"}

def predict_battery(device_id, hours=2, limit=200):
    from os.path import exists
    if not exists(BATTERY_MODEL_PATH):
        return {"success": False, "error": "Battery model not trained"}

    df = prepare_features(device_id, limit)
    if df.empty:
        return {"success": False, "error": "No data found"}

    df["t"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

    model = joblib.load(BATTERY_MODEL_PATH)

    future_seconds = hours * 3600
    future_t = df["t"].max() + future_seconds

    predicted_percent = model.predict([[future_t]])[0]

    return {
        "success": True,
        "device_id": device_id,
        "current_battery": df["battery_percent"].iloc[-1],
        "predicted_battery_after_hours": hours,
        "prediction": float(predicted_percent)
    }
