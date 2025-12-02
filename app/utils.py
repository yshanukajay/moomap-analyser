# Utility helpers (notifications)
from shapely.geometry import Point, Polygon

def is_inside_polygon(lat, lon, polygon_points):
    """
    Check if a given latitude/longitude is inside a polygon.
    
    Parameters:
    - lat (float): latitude of the point
    - lon (float): longitude of the point
    - polygon_points (list): list of [lon, lat] coordinates defining the polygon
    
    Returns:
    - bool: True if point is inside polygon, False otherwise
    """
    poly = Polygon(polygon_points)
    point = Point(lon, lat)
    return poly.contains(point)


def normalize_coordinates(coords):
    """
    Ensure coordinates are in [lon, lat] format for Shapely/GeoJSON.
    
    Parameters:
    - coords: list of coordinates (may be [lat, lon] or [lon, lat])
    
    Returns:
    - list of [lon, lat] coordinates
    """
    normalized = []
    for c in coords:
        if c[0] < -180 or c[0] > 180:  # probably [lat, lon]
            normalized.append([c[1], c[0]])
        else:
            normalized.append([c[0], c[1]])
    return normalized
