from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from app.db import get_db

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="MoOmap Battery + Anomaly API")

# ---------------------------
# Load models
# ---------------------------
# Battery model
battery_model = joblib.load('models/battery_predictor_model.joblib')
scaler = joblib.load('models/battery_status_scaler.joblib')

# Anomaly model (if already trained)
try:
    anomaly_model = joblib.load('models/isolation_forest_model.joblib')
    anomaly_scaler = joblib.load('models/scaler.joblib')
except:
    anomaly_model = None
    anomaly_scaler = None

# ---------------------------
# Pydantic request models
# ---------------------------
class DeviceBatteryData(BaseModel):
    device_id: str
    battery_percent: float
    battery_voltage: float
    battery_drop_per_s: float  # Can be calculated before sending
    hours_ahead: float = 1     # Default 1 hour prediction

class BulkBatteryData(BaseModel):
    devices: List[DeviceBatteryData]

class DeviceAnomalyData(BaseModel):
    device_id: str
    battery_percent: float
    battery_voltage: float
    battery_drop_per_s: float
    # Add other features required for anomaly prediction

class BulkAnomalyData(BaseModel):
    devices: List[DeviceAnomalyData]

# ---------------------------
# Battery status helper
# ---------------------------
def battery_status(percent):
    if percent >= 70:
        return "good"
    elif percent >= 40:
        return "intermediate"
    else:
        return "bad"

# ---------------------------
# Battery prediction endpoint (bulk)
# ---------------------------
@app.post("/predict_battery_bulk")
def predict_battery_bulk(data: BulkBatteryData):
    results = []
    for device in data.devices:
        # Prepare features for prediction
        df = pd.DataFrame([{
            'battery_percent': device.battery_percent,
            'battery_voltage': device.battery_voltage,
            'battery_drop_per_s': device.battery_drop_per_s
        }])
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Predict next battery percent
        predicted_percent_next = battery_model.predict(df_scaled)[0]
        
        # Predict battery after X hours
        seconds_ahead = device.hours_ahead * 3600
        predicted_future_percent = predicted_percent_next - device.battery_drop_per_s * seconds_ahead
        
        # Ensure battery percent is between 0 and 100
        predicted_future_percent = max(0, min(100, predicted_future_percent))
        
        # Determine battery status
        status = battery_status(predicted_future_percent)
        
        results.append({
            "device_id": device.device_id,
            "predicted_battery_percent_next": round(predicted_percent_next, 2),
            "predicted_battery_percent_future": round(predicted_future_percent, 2),
            "status": status
        })
    return {"results": results}

# ---------------------------
# Anomaly prediction endpoint (bulk)
# ---------------------------
@app.post("/predict_anomaly_bulk")
def predict_anomaly_bulk(data: BulkAnomalyData):
    if anomaly_model is None or anomaly_scaler is None:
        return {"error": "Anomaly model not trained yet."}

    results = []
    for device in data.devices:
        df = pd.DataFrame([{
            'battery_percent': device.battery_percent,
            'battery_voltage': device.battery_voltage,
            'battery_drop_per_s': device.battery_drop_per_s
            # Include any other features used in anomaly training
        }])
        df_scaled = anomaly_scaler.transform(df)
        pred = anomaly_model.predict(df_scaled)[0]
        label = "normal" if pred == 1 else "anomalous"
        
        results.append({
            "device_id": device.device_id,
            "anomaly_label": label
        })
    return {"results": results}

# ---------------------------
# Combined endpoint: battery + anomaly
# ---------------------------
@app.post("/predict_device_health_bulk")
def predict_device_health_bulk(battery_data: BulkBatteryData, anomaly_data: BulkAnomalyData):
    battery_results = predict_battery_bulk(battery_data)['results']
    anomaly_results = predict_anomaly_bulk(anomaly_data)['results']

    # Merge results by device_id
    merged = []
    for b in battery_results:
        a = next((item for item in anomaly_results if item["device_id"] == b["device_id"]), None)
        merged.append({
            **b,
            "anomaly_label": a["anomaly_label"] if a else "unknown"
        })
    return {"results": merged}

# ---------------------------
# Optional: simple health check
# ---------------------------
@app.get("/")
def root():
    return {"message": "MoOmap Battery + Anomaly API running ✅"}
