import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
mongodb_database = os.getenv("MONGODB_DATABASE")

print(f"Testing MongoDB Connection...")
print(f"URI: {mongodb_uri}")
print(f"Database: {mongodb_database}")

try:
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=3000)
    client.admin.command('ping')
    print("✅ MongoDB Connected Successfully!")
    
    db = client[mongodb_database]
    print(f"\n📚 Collections in {mongodb_database}:")
    for col in db.list_collection_names():
        count = db[col].count_documents({})
        print(f"   - {col}: {count} documents")
    
    # Check conversations
    if "conversations" in db.list_collection_names():
        conv_count = db["conversations"].count_documents({})
        print(f"\n✅ Conversations: {conv_count} total")
    
    client.close()
    
except Exception as e:
    print(f"❌ Connection Failed: {e}")