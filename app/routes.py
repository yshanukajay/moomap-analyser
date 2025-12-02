# Flask API routes
from flask import Blueprint, request, jsonify
from .models import get_latest_cattle, save_polygon, get_polygon, get_all_device_data
from .services.identification import identify_objects

api_bp = Blueprint('api', __name__)

# ------------------------------------------------------
# Get latest device/cattle status
# ------------------------------------------------------
@api_bp.route('/device/<device_id>/latest', methods=['GET'])
def get_device_latest(device_id):
    data = get_latest_cattle(device_id)
    if not data:
        return jsonify({"success": False, "error": "Device not found"}), 404
    return jsonify({"success": True, "data": data})

# ------------------------------------------------------
# Get device history
# ------------------------------------------------------
@api_bp.route('/device/<device_id>/history', methods=['GET'])
def get_device_history(device_id):
    limit = int(request.args.get("limit", 100))
    data = get_all_device_data(device_id, limit=limit)
    for d in data:
        d["_id"] = str(d["_id"])
    return jsonify({"success": True, "data": data})

# ------------------------------------------------------
# Save Polygon
# ------------------------------------------------------
@api_bp.route('/polygon', methods=['POST'])
def post_polygon():
    data = request.get_json() or {}
    userId = data.get('userId')
    name = data.get('name')
    coordinates = data.get('coordinates')

    if not userId or not coordinates:
        return jsonify({"success": False, "error": "userId and coordinates required"}), 400

    poly = save_polygon(userId, name, coordinates)
    poly['_id'] = str(poly['_id'])
    return jsonify({"success": True, "data": poly})

# ------------------------------------------------------
# Identify objects around cattle/device
# ------------------------------------------------------
@api_bp.route('/identify', methods=['POST'])
def identify():
    data = request.get_json() or {}
    device_id = data.get('device_id')
    polygonId = data.get('polygonId')
    polygonCoordinates = data.get('polygonCoordinates')
    radiusMeters = data.get('radiusMeters', 200)

    if not device_id:
        return jsonify({"success": False, "error": "device_id required"}), 400

    device = get_latest_cattle(device_id)
    if not device:
        return jsonify({"success": False, "error": "Device not found"}), 404

    polygon = None
    if polygonId:
        polygon_doc = get_polygon(polygonId)
        if not polygon_doc:
            return jsonify({"success": False, "error": "Polygon not found"}), 404
        polygon = polygon_doc.get("polygon")
    elif polygonCoordinates:
        polygon = {"type": "Polygon", "coordinates": polygonCoordinates}

    result = identify_objects(device, polygon, radiusMeters)
    return jsonify({"success": True, "data": result})
