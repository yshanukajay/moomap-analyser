from app.db import get_db
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime

def fetch_device_data(device_id, limit=1000):
    """
    Fetch latest N data points for a device from MongoDB
    """
    db = get_db()
    collection = db[f"dev_{device_id}"]
    data = list(collection.find().sort("_id", -1).limit(limit))
    if not data:
        return pd.DataFrame()  # return empty dataframe if no data

    # Normalize into DataFrame
    df = pd.DataFrame([
        {
            "timestamp": doc["_id"].generation_time,
            "lat": doc["gps"]["lat"],
            "lon": doc["gps"]["lon"],
            "battery_percent": doc["battery"]["percent"],
            "voltage": doc["battery"]["voltage"]
        }
        for doc in data
    ])

    df = df.sort_values("timestamp")  # sort chronologically
    df.reset_index(drop=True, inplace=True)
    return df


def compute_movement_features(df):
    """
    Compute distance, speed, and inactivity based on GPS coordinates
    """
    distances = []
    speeds = []

    for i in range(len(df)):
        if i == 0:
            distances.append(0)
            speeds.append(0)
        else:
            prev = (df.loc[i-1, "lat"], df.loc[i-1, "lon"])
            curr = (df.loc[i, "lat"], df.loc[i, "lon"])
            distance_m = geodesic(prev, curr).meters
            time_diff_s = (df.loc[i, "timestamp"] - df.loc[i-1, "timestamp"]).total_seconds()
            speed_m_s = distance_m / time_diff_s if time_diff_s > 0 else 0
            distances.append(distance_m)
            speeds.append(speed_m_s)

    df["distance_m"] = distances
    df["speed_m_s"] = speeds
    df["idle"] = df["speed_m_s"] < 0.1  # mark very slow movements as idle
    return df


def prepare_features(device_id, limit=1000):
    """
    Full pipeline: fetch, clean, and compute features
    """
    df = fetch_device_data(device_id, limit)
    if df.empty:
        return df
    df = compute_movement_features(df)
    return df
