# Configuration loader
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB connection string
    MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/ProjectDB')

    # Overpass API endpoint
    OVERPASS_URL = os.environ.get('OVERPASS_URL', 'https://overpass-api.de/api/interpreter')

    # Flask server settings
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
