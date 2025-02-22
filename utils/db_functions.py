import os
import urllib.parse
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_mongo_client():
    # Get the MongoDB credentials and details from the environment
    username = os.getenv("MONGO_USERNAME")
    password = os.getenv("MONGO_PASSWORD")
    cluster = os.getenv("MONGO_CLUSTER")
    # database = os.getenv("MONGO_DB")

    # URL-encode the username and password to handle special characters
    username_encoded = urllib.parse.quote_plus(username)
    password_encoded = urllib.parse.quote_plus(password)

    # # Build the connection string
    # mongo_uri = (
    #     f"mongodb+srv://{username_encoded}:{password_encoded}@{cluster}/"
    #     f"{database}?retryWrites=true&w=majority"
    # )

    mongo_uri = f"mongodb+srv://{username_encoded}:{password_encoded}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"

    # Create and return a MongoClient
    client = MongoClient(mongo_uri)
    return client

def get_db():
    client = get_mongo_client()
    # Use the database specified in the .env file
    db_name = os.getenv("MONGO_DB")
    return client[db_name]

def insert_roadmap(entry):
    db = get_db()
    collection = db["roadmaps"]
    return collection.insert_one(entry)

def get_all_roadmaps():
    db = get_db()
    collection = db["roadmaps"]
    # Retrieve all documents, sorted by timestamp (latest first)
    return list(collection.find().sort("timestamp", -1))
