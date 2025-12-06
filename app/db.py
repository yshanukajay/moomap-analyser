# app/db.py
from pymongo import MongoClient
import os

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://admin:A9fT3xPq@213.199.51.193:27017/moomap?authSource=admin')
DB_NAME = os.getenv('DB_NAME', 'moomap')

def get_db():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db
