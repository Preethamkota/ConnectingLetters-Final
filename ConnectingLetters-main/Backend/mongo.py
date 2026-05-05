from pymongo import MongoClient
from datetime import datetime

import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["game"]
collection = db["reports"]


def save_result(data):
    try:
        data["timestamp"] = datetime.utcnow()
        result = collection.insert_one(data)
        return str(result.inserted_id)
    except Exception as e:
        print("mongo insert error",e)
        return None