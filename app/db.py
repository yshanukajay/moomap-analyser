# MongoDB connection setup
from flask import current_app
from pymongo import MongoClient

def init_db(app):
    """
    Initialize MongoDB connection and attach client & database to Flask app.
    """
    try:
        client = MongoClient(app.config['MONGODB_URI'])
        
        # Use the database from URI, or default to 'moomap'
        db = client.get_default_database()
        if db is None:
            # If URI does not specify DB, default to 'moomap'
            db = client['moomap']

        app.mongo_client = client
        app.db = db

        print(f"[INFO] Connected to MongoDB database: {db.name}")

    except Exception as e:
        print("[ERROR] Failed to connect to MongoDB:", e)
        raise e

def get_db():
    """
    Get the MongoDB database instance from the current Flask app context.
    """
    return current_app.db
