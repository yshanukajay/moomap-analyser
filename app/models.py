# MongoDB models and helpers
from .db import get_db
from datetime import datetime
from bson import ObjectId

# ------------------------------------------------------
# Get latest cattle/device GPS status
# ------------------------------------------------------
def get_latest_cattle(device_id):
    db = get_db()
    collection_name = f"dev_{device_id}"
    collection = db[collection_name]

    doc = collection.find_one(sort=[("_id", -1)])
    if not doc:
        return None

    gps = doc.get("gps", {})
    battery = doc.get("battery", {})
    if gps.get("lon") is None or gps.get("lat") is None:
        return None

    return {
        "device_id": doc.get("device_id"),
        "location": {
            "type": "Point",
            "coordinates": [gps.get("lon"), gps.get("lat")]
        },
        "battery": {
            "percent": battery.get("percent"),
            "voltage": battery.get("voltage"),
        },
        "signal": doc.get("signal", {}),
        "updatedAt": doc.get("_id").generation_time
    }

# ------------------------------------------------------
# Get last N data points
# ------------------------------------------------------
def get_all_device_data(device_id, limit=50):
    db = get_db()
    collection = db[f"dev_{device_id}"]
    return list(collection.find().sort("_id", -1).limit(limit))

# ------------------------------------------------------
# Polygon save
# ------------------------------------------------------
def save_polygon(user_id, name, coordinates):
    db = get_db()
    doc = {
        "userId": user_id,
        "name": name,
        "polygon": {
            "type": "Polygon",
            "coordinates": coordinates
        },
        "createdAt": datetime.utcnow()
    }
    res = db.polygon_areas.insert_one(doc)
    doc["_id"] = res.inserted_id
    return doc

# ------------------------------------------------------
# Polygon retrieve
# ------------------------------------------------------
def get_polygon(polygon_id):
    db = get_db()
    return db.polygon_areas.find_one({"_id": ObjectId(polygon_id)})
