# OSM Overpass API service
import requests
from flask import current_app
import logging

logging.basicConfig(level=logging.INFO)


def build_overpass_query_for_bbox(south, west, north, east):
    """
    Build Overpass QL query for a bounding box.
    Fetch natural features, waterways, buildings, roads, railway.
    """
    q = f"""
[out:json][timeout:25];
(
    node["natural"~"water|tree"]({south},{west},{north},{east});
    way["natural"~"water|wood|scrub"]({south},{west},{north},{east});
    relation["natural"~"water|wood|scrub"]({south},{west},{north},{east});
    node["waterway"]({south},{west},{north},{east});
    way["waterway"]({south},{west},{north},{east});
    node["building"]({south},{west},{north},{east});
    way["building"]({south},{west},{north},{east});
    node["highway"]({south},{west},{north},{east});
    way["highway"]({south},{west},{north},{east});
    node["railway"]({south},{west},{north},{east});
    way["railway"]({south},{west},{north},{east});
);
out body geom;
"""
    return q


def fetch_objects_in_bbox(south, west, north, east):
    """
    Fetch OSM elements in bounding box from Overpass API.
    Returns list of elements.
    """
    url = current_app.config.get('OVERPASS_URL')
    if not url:
        logging.error("OVERPASS_URL not set in Flask config")
        return []

    q = build_overpass_query_for_bbox(south, west, north, east)
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'CattleTrackingApp/1.0 (example@example.com)'
    }

    try:
        res = requests.post(url, data={'data': q}, headers=headers, timeout=30)
        res.raise_for_status()
        data = res.json()
        elements = data.get('elements', [])
        logging.info(f"Fetched {len(elements)} elements from OSM")
        return elements

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching OSM data: {e}")
        return []
