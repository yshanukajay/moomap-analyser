# Object identification and filtering logic
from shapely.geometry import Point, Polygon, shape, LineString
from geopy.distance import geodesic
from .osm_service import fetch_objects_in_bbox
import math
import logging

logging.basicConfig(level=logging.INFO)

# -----------------------------------------------
# Helper: bounding box from user polygon
# -----------------------------------------------
def _bbox_from_polygon(polygon):
    coords = polygon['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    south = min(lats)
    north = max(lats)
    west = min(lons)
    east = max(lons)
    return south, west, north, east


# -----------------------------------------------
# Helper: bounding box from point + radius (meters)
# -----------------------------------------------
def _bbox_from_point_radius(point_coords, radius_m):
    lon, lat = point_coords
    lat_delta = radius_m / 111000.0
    lon_delta = radius_m / (111000.0 * abs(math.cos(math.radians(lat)))) if lat != 0 else radius_m / 111000.0
    south = lat - lat_delta
    north = lat + lat_delta
    west = lon - lon_delta
    east = lon + lon_delta
    return south, west, north, east


# -----------------------------------------------
# Convert OSM element to GeoJSON-like feature
# -----------------------------------------------
def osm_element_to_feature(el):
    if el.get('type') == 'node':
        return {
            'id': f"node/{el.get('id')}",
            'geometry': {'type': 'Point', 'coordinates': [el.get('lon'), el.get('lat')]},
            'properties': el.get('tags', {})
        }
    if el.get('type') == 'way' and el.get('geometry'):
        coords = [[g.get('lon'), g.get('lat')] for g in el.get('geometry')]
        is_closed = len(coords) > 2 and coords[0] == coords[-1]
        return {
            'id': f"way/{el.get('id')}",
            'geometry': {
                'type': 'Polygon' if is_closed else 'LineString',
                'coordinates': coords if not is_closed else [coords]
            },
            'properties': el.get('tags', {})
        }
    return None


# -----------------------------------------------
# Main function: identify objects around cattle/device
# -----------------------------------------------
def identify_objects(device, polygon=None, radius_m=200):
    """
    device: {'device_id': ..., 'location': {'coordinates': [lon, lat]}}
    polygon: optional GeoJSON Polygon
    radius_m: search radius if no polygon is provided
    """
    device_coords = device['location']['coordinates']
    device_point = Point(device_coords[0], device_coords[1])

    # Determine bounding box
    if polygon:
        south, west, north, east = _bbox_from_polygon(polygon)
    else:
        south, west, north, east = _bbox_from_point_radius(device_coords, radius_m)

    # Fetch objects from OSM
    elements = fetch_objects_in_bbox(south, west, north, east)
    features = [osm_element_to_feature(el) for el in elements if osm_element_to_feature(el)]

    results = []
    for feat in features:
        geom = feat['geometry']
        inside_polygon = False
        distance_m = None

        try:
            if geom['type'] == 'Point':
                pt = Point(geom['coordinates'][0], geom['coordinates'][1])
                if polygon:
                    user_poly = Polygon(polygon['coordinates'][0])
                    inside_polygon = user_poly.contains(pt)
                distance_m = geodesic(
                    (device_coords[1], device_coords[0]),
                    (geom['coordinates'][1], geom['coordinates'][0])
                ).meters

            elif geom['type'] == 'Polygon':
                poly_feat = Polygon(geom['coordinates'][0])
                if polygon:
                    user_poly = Polygon(polygon['coordinates'][0])
                    inside_polygon = user_poly.intersects(poly_feat)
                centroid = poly_feat.centroid
                distance_m = geodesic(
                    (device_coords[1], device_coords[0]),
                    (centroid.y, centroid.x)
                ).meters

            elif geom['type'] == 'LineString':
                line = LineString(geom['coordinates'])
                if polygon:
                    user_poly = Polygon(polygon['coordinates'][0])
                    inside_polygon = user_poly.intersects(line)
                closest_point = line.interpolate(line.project(device_point))
                distance_m = geodesic(
                    (device_coords[1], device_coords[0]),
                    (closest_point.y, closest_point.x)
                ).meters
        except Exception as e:
            logging.warning(f"Error processing feature {feat['id']}: {e}")
            distance_m = None

        results.append({
            'id': feat['id'],
            'geometry': feat['geometry'],
            'properties': feat['properties'],
            'insidePolygon': inside_polygon,
            'distanceMeters': distance_m
        })

    # Filter nearby dangerous objects (<20m)
    nearby_dangerous = [
        {
            'id': r['id'],
            'type': r['properties'].get('building') and 'building' or
                    (r['properties'].get('highway') and 'road' or 'object'),
            'distanceMeters': r['distanceMeters']
        }
        for r in results
        if r['distanceMeters'] is not None and r['distanceMeters'] <= 20
    ]

    logging.info(f"Total objects fetched: {len(results)}, nearby dangerous: {len(nearby_dangerous)}")

    return {
        'cattle': {
            'device_id': device.get('device_id'),
            'location': device['location']
        },
        'objects': results,
        'summary': {
            'nearbyDangerous': nearby_dangerous
        }
    }
