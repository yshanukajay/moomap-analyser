import requests
import json

# ----------------------------
# Configuration
# ----------------------------
BASE_URL = "http://localhost:5000/api"  # replace with your PC IP if using phone
DEVICE_ID = "7454927D7850"  # replace with your cattle/device id

# Example polygon coordinates (replace with your area)
POLYGON_COORDINATES = [
    [[79.97, 7.31], [79.98, 7.31], [79.98, 7.32], [79.97, 7.31]]
]

USER_ID = "user123"
POLYGON_NAME = "Farm Area"

# ----------------------------
# Step 1: Save polygon
# ----------------------------
polygon_payload = {
    "userId": USER_ID,
    "name": POLYGON_NAME,
    "coordinates": POLYGON_COORDINATES
}

resp = requests.post(f"{BASE_URL}/polygon", json=polygon_payload)
if resp.status_code != 200:
    print("Failed to save polygon:", resp.status_code, resp.text)
    exit(1)

polygon_data = resp.json()
polygon_id = polygon_data["_id"]
print("Polygon saved:", polygon_data)

# ----------------------------
# Step 2: Identify objects
# ----------------------------
identify_payload = {
    "device_id": DEVICE_ID,
    "polygonId": polygon_id,
    "radiusMeters": 200
}


resp = requests.post(f"{BASE_URL}/polygon", json=polygon_payload)
print("Polygon save response:", resp.status_code, resp.text)

if resp.status_code != 200:
    print("Failed to save polygon")
    exit(1)

polygon_data = resp.json()

# Safely get _id
polygon_id = polygon_data.get("_id")
if not polygon_id:
    print("No _id returned. Cannot proceed.")
    exit(1)

print("Polygon saved:", polygon_data)
