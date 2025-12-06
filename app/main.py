from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from pymongo import MongoClient
import threading
from typing import Dict
from app.cattle_osm.cattle_surroundings import get_surroundings, check_boundaries





# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="MoOmap Battery + Anomaly API")

# ---------------------------
# Load models
# ---------------------------
battery_model = joblib.load('models/battery_predictor_model.joblib')
battery_scaler = joblib.load('models/battery_status_scaler.joblib')

try:
    anomaly_model = joblib.load('models/isolation_forest_model.joblib')
    anomaly_scaler = joblib.load('models/scaler.joblib')
except Exception as e:
    anomaly_model = None
    anomaly_scaler = None
    print("Anomaly model not loaded:", e)


# ---------------------------
# Pydantic request models
# ---------------------------
class DeviceBatteryData(BaseModel):
    device_id: str
    battery_percent: float
    battery_voltage: float
    battery_drop_per_s: float
    hours_ahead: float = 1


class BulkBatteryData(BaseModel):
    devices: List[DeviceBatteryData]


class DeviceAnomalyData(BaseModel):
    device_id: str
    lat: float
    lon: float
    battery_percent: float
    battery_voltage: float
    speed_m_s: float
    battery_drop_per_s: float


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
        df = pd.DataFrame([{
            'battery_percent': device.battery_percent,
            'battery_voltage': device.battery_voltage,
            'battery_drop_per_s': device.battery_drop_per_s
        }])
        df_scaled = battery_scaler.transform(df)
        predicted_percent_next = battery_model.predict(df_scaled)[0]

        seconds_ahead = device.hours_ahead * 3600
        predicted_future_percent = predicted_percent_next - device.battery_drop_per_s * seconds_ahead
        predicted_future_percent = max(0, min(100, predicted_future_percent))
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
        return {"error": "Anomaly model not trained or missing."}

    results = []
    for device in data.devices:
        try:
            df = pd.DataFrame([{
                'lat': device.lat,
                'lon': device.lon,
                'battery_percent': device.battery_percent,
                'battery_voltage': device.battery_voltage,
                'speed_m_s': device.speed_m_s,
                'battery_drop_per_s': device.battery_drop_per_s
            }])
            df_scaled = anomaly_scaler.transform(df)
            pred = anomaly_model.predict(df_scaled)[0]
            label = "normal" if pred == 1 else "anomalous"

            results.append({
                "device_id": device.device_id,
                "anomaly_label": label
            })
        except Exception as e:
            results.append({
                "device_id": device.device_id,
                "anomaly_label": "error",
                "error_msg": str(e)
            })
    return {"results": results}


# ---------------------------
# Combined endpoint: battery + anomaly
# ---------------------------
@app.post("/predict_device_health_bulk")
def predict_device_health_bulk(battery_data: BulkBatteryData, anomaly_data: BulkAnomalyData):
    battery_results = predict_battery_bulk(battery_data)['results']
    anomaly_results = predict_anomaly_bulk(anomaly_data)['results']

    merged = []
    for b in battery_results:
        a = next((item for item in anomaly_results if item["device_id"] == b["device_id"]), None)
        merged.append({
            **b,
            "anomaly_label": a["anomaly_label"] if a else "unknown"
        })
    return {"results": merged}


# ---------------------------
# Health check endpoint
# ---------------------------
@app.get("/")
def root():
    return {"message": "MoOmap Battery + Anomaly API running ✅"}


# ============================================================
# REAL-TIME MONGODB WATCHER (Runs Automatically)
# ============================================================

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "ProjectDB"
COLLECTION_NAME = "Collar_data"

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

def process_new_record(doc):
    """Extract features and run both ML predictions."""
    try:
        device_id = doc.get("device_id")
        gps = doc.get("gps", {})
        battery = doc.get("battery", {})

        features_battery = {
            "device_id": device_id,
            "battery_percent": battery.get("percent", 0),
            "battery_voltage": battery.get("voltage", 0),
            "battery_drop_per_s": 0.0001
        }

        features_anomaly = {
            "device_id": device_id,
            "lat": gps.get("lat", 0),
            "lon": gps.get("lon", 0),
            "battery_percent": battery.get("percent", 0),
            "battery_voltage": battery.get("voltage", 0),
            "speed_m_s": doc.get("speed_m_s", 0),
            "battery_drop_per_s": 0.0001
        }

        battery_data = BulkBatteryData(devices=[DeviceBatteryData(**features_battery)])
        anomaly_data = BulkAnomalyData(devices=[DeviceAnomalyData(**features_anomaly)])

        result = predict_device_health_bulk(battery_data, anomaly_data)
        print("\n🔥 NEW DEVICE PREDICTION 🔥")
        print(result)

    except Exception as e:
        print("Error processing MongoDB record:", e)


def mongo_watch_thread():
    print("🔎 MongoDB watcher running...")
    try:
        with collection.watch([{"$match": {"operationType": "insert"}}]) as stream:
            for change in stream:
                new_doc = change["fullDocument"]
                process_new_record(new_doc)
    except Exception as e:
        print("MongoDB watcher error:", e)


def start_watcher():
    watcher = threading.Thread(target=mongo_watch_thread, daemon=True)
    watcher.start()


start_watcher()

# ---------------------------
# Cattle Surroundings + Boundary Alert Endpoint
# ---------------------------

class CattleLocation(BaseModel):
    device_id: str
    lat: float
    lon: float

class PolygonArea(BaseModel):
    name: str
    coordinates: List[List[float]]  # List of [lat, lon] points forming the polygon

class CattleAreaRequest(BaseModel):
    cattles: List[CattleLocation]
    polygons: List[PolygonArea]

@app.post("/cattle_surroundings")
def cattle_surroundings(data: CattleAreaRequest):
    results = []

    for cattle in data.cattles:
        # Identify nearby objects using OpenStreetMap
        surroundings = get_surroundings(cattle.lat, cattle.lon)

        # Check if cattle is outside any user-defined polygon
        boundary_alerts = []
        for poly in data.polygons:
            alert = check_boundaries(cattle.lat, cattle.lon, poly.coordinates)
            if alert:
                boundary_alerts.append(poly.name)

        results.append({
            "device_id": cattle.device_id,
            "lat": cattle.lat,
            "lon": cattle.lon,
            "surroundings": surroundings,
            "boundary_alerts": boundary_alerts  # empty list if inside all polygons
        })

    return {"results": results}
