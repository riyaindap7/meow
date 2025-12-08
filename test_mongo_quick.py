"""Quick test to check if MongoDB is working"""
from pymongo import MongoClient
import sys

try:
    print("🔄 Connecting to MongoDB at localhost:27017...")
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    
    # Test connection
    client.admin.command('ping')
    print("✅ Connected to MongoDB successfully!")
    
    # Check victor_rag database
    db = client["victor_rag"]
    print(f"\n📊 Database: victor_rag")
    
    # List collections
    collections = db.list_collection_names()
    print(f"📋 Available collections ({len(collections)}):")
    for coll_name in collections:
        count = db[coll_name].count_documents({})
        print(f"   ✅ {coll_name}: {count} documents")
    
    # Check conversations collection specifically
    if "conversations" in collections:
        print(f"\n🔍 Conversations Collection Details:")
        conversations = db["conversations"]
        
        # Get total count
        total = conversations.count_documents({})
        print(f"   Total conversations: {total}")
        
        # Get sample conversation
        sample = conversations.find_one({}, {"_id": 0, "conversation_id": 1, "user_id": 1, "title": 1, "created_at": 1, "updated_at": 1, "messages": 1})
        if sample:
            print(f"\n   Sample conversation:")
            print(f"      ID: {sample.get('conversation_id', 'N/A')}")
            print(f"      User: {sample.get('user_id', 'N/A')}")
            print(f"      Title: {sample.get('title', 'N/A')}")
            print(f"      Created: {sample.get('created_at', 'N/A')} (type: {type(sample.get('created_at')).__name__})")
            print(f"      Updated: {sample.get('updated_at', 'N/A')} (type: {type(sample.get('updated_at')).__name__})")
            print(f"      Messages: {len(sample.get('messages', []))}")
    
    print("\n" + "="*80)
    print("✅ MONGODB IS WORKING CORRECTLY!")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print(f"❌ MONGODB CONNECTION FAILED!")
    sys.exit(1)
