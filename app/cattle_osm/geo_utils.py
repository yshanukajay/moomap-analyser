# app/cattle_osm/geo_utils.py
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic

def is_point_in_polygon(lat, lon, polygon_coords):
    """
    Check if a point (lat, lon) is inside a polygon.
    polygon_coords: list of (lat, lon) tuples defining the polygon
    """
    point = Point(lon, lat)  # shapely uses (x, y) => (lon, lat)
    polygon = Polygon([(lng, lt) for lt, lng in polygon_coords])
    return polygon.contains(point)

def distance_between_points(coord1, coord2):
    """
    Returns distance in meters between two points (lat, lon)
    """
    return geodesic(coord1, coord2).meters

def filter_objects_within_radius(objects, center_coord, radius_m):
    """
    Filter a list of objects based on distance from a center point
    objects: list of dicts with 'lat' and 'lon' keys
    center_coord: (lat, lon)
    radius_m: radius in meters
    """
    nearby = []
    for obj in objects:
        obj_coord = (obj['lat'], obj['lon'])
        dist = distance_between_points(center_coord, obj_coord)
        if dist <= radius_m:
            nearby.append(obj)
    return nearby
