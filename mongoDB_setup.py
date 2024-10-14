from pymongo import MongoClient
import gridfs
import logging
import log_config


# MongoDB Setup
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["pdf_db"]
    metadata_collection = db["pdf_data"]
    logging.info("Connected to MongoDB successfully.")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
