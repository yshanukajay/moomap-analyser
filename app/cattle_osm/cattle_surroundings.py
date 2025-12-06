# app/cattle_osm/cattle_surroundings.py
from app.cattle_osm.osm_fetcher import fetch_osm_objects
from app.cattle_osm.geo_utils import point_in_polygon
from pymongo import MongoClient
import os

# ----------------------
# MongoDB Configuration
# ----------------------
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://admin:A9fT3xPq@213.199.51.193:27017/moomap?authSource=admin')
DB_NAME = os.getenv('DB_NAME', 'moomap')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'dummy_data_CSV_unlabeled')

def connect_db():
    client = MongoClient(MONGO_URI)
    return client

def fetch_cattle_locations(client, limit=None):
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]
    cursor = coll.find({'gps.valid': True})
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)

def analyze_cattle_surroundings(cattle_docs, user_polygons, radius=100):
    """
    cattle_docs: list of cattle records from MongoDB
    user_polygons: dict with device_id as key, polygon list as value
    radius: search radius for OSM objects
    Returns: dict with cattle info, surroundings, and alerts
    """
    results = []

    for doc in cattle_docs:
        device_id = doc['device_id']
        lat = doc['gps']['lat']
        lon = doc['gps']['lon']

        # Fetch nearby objects
        nearby_objects = fetch_osm_objects(lat, lon, radius=radius)

        # Filter objects inside user polygon(s)
        polygons = user_polygons.get(device_id, [])
        objects_in_polygons = []
        alerts = []

        for obj in nearby_objects:
            for poly in polygons:
                if point_in_polygon(obj['lat'], obj['lon'], poly):
                    objects_in_polygons.append(obj)

        # Check if cattle is outside polygon(s)
        inside_any_polygon = any(point_in_polygon(lat, lon, poly) for poly in polygons)
        if not inside_any_polygon:
            alerts.append("Cattle outside user-defined boundary!")

        results.append({
            "device_id": device_id,
            "lat": lat,
            "lon": lon,
            "nearby_objects_in_polygon": objects_in_polygons,
            "alerts": alerts
        })

    return results
