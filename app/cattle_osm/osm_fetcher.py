# app/cattle_osm/osm_fetcher.py
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def fetch_osm_objects(lat, lon, radius=100):
    """
    Fetch nearby OSM objects around a point.
    lat, lon: center point
    radius: search radius in meters
    Returns a list of dicts: [{'type': ..., 'lat': ..., 'lon': ..., 'tags': {...}}, ...]
    """
    # Overpass QL query
    query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon})["amenity"];
      node(around:{radius},{lat},{lon})["natural"];
      node(around:{radius},{lat},{lon})["building"];
    );
    out center;
    """
    try:
        response = requests.post(OVERPASS_URL, data={'data': query}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("OSM fetch error:", e)
        return []

    results = []
    for element in data.get('elements', []):
        results.append({
            "type": element.get('type'),
            "lat": element.get('lat'),
            "lon": element.get('lon'),
            "tags": element.get('tags', {})
        })
    return results
