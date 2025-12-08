"""
Debug script to check all collections and find where users are stored
"""
import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

def debug_collections():
    """Check all collections and find users"""
    
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://admin:meow@192.168.0.106:27017/")
    mongodb_database = os.getenv("MONGODB_DATABASE", "victor_rag")
    
    client = MongoClient(mongodb_uri)
    db = client[mongodb_database]
    
    print(f"🔍 Checking database: {mongodb_database}")
    print(f"🔗 URI: {mongodb_uri}")
    
    # List all collections
    collections = db.list_collection_names()
    print(f"\\n📚 Collections found: {collections}")
    
    # Check each collection for user-like documents
    for collection_name in collections:
        collection = db[collection_name]
        count = collection.count_documents({})
        print(f"\\n📋 Collection: {collection_name} ({count} documents)")
        
        if count > 0:
            # Sample some documents
            samples = list(collection.find({}).limit(3))
            for i, doc in enumerate(samples, 1):
                print(f"   Sample {i}: {list(doc.keys())[:10]}...")
                if 'email' in doc:
                    print(f"      → Email found: {doc.get('email')}")
                if 'name' in doc:
                    print(f"      → Name found: {doc.get('name')}")
                if '_id' in doc:
                    print(f"      → ID: {doc['_id']}")
    
    client.close()

if __name__ == "__main__":
    debug_collections()