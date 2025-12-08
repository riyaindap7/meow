#!/usr/bin/env python3
"""
Simple MongoDB connection test - no imports needed
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent / "backend" / ".env"
load_dotenv(env_file)

print("=" * 60)
print("🔍 VICTOR RAG MongoDB Connection Test")
print("=" * 60)

# Check environment variables
mongodb_uri = os.getenv("MONGODB_URI", "NOT SET")
mongodb_db = os.getenv("MONGODB_DATABASE", "NOT SET")

print("\n📋 Environment Variables:")
print(f"  MONGODB_URI: {mongodb_uri}")
print(f"  MONGODB_DATABASE: {mongodb_db}")

if mongodb_uri == "NOT SET" or mongodb_db == "NOT SET":
    print("\n❌ Missing environment variables!")
    sys.exit(1)

# Try direct pymongo import
print("\n🔄 Testing imports...")
try:
    import pymongo
    print(f"✅ PyMongo version: {pymongo.__version__}")
except ImportError as e:
    print(f"❌ PyMongo import failed: {e}")
    sys.exit(1)

try:
    from pymongo import MongoClient
    print("✅ MongoClient imported")
except ImportError as e:
    print(f"❌ MongoClient import failed: {e}")
    sys.exit(1)

try:
    from bson import ObjectId
    print("✅ ObjectId imported")
except ImportError as e:
    print(f"❌ ObjectId import failed: {e}")
    sys.exit(1)

# Try MongoDB connection
print("\n🔄 Connecting to MongoDB...")
try:
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
    # Verify connection
    client.admin.command('ping')
    print(f"✅ Connected to MongoDB!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    sys.exit(1)

# Check database
print(f"\n🔄 Checking database '{mongodb_db}'...")
try:
    db = client[mongodb_db]
    print(f"✅ Database selected: {mongodb_db}")
except Exception as e:
    print(f"❌ Database selection failed: {e}")
    sys.exit(1)

# Check/create conversations collection
print("\n🔄 Checking conversations collection...")
try:
    collections = db.list_collection_names()
    print(f"✅ Collections in database: {collections}")
    
    if "conversations" in collections:
        print("✅ 'conversations' collection EXISTS")
        doc_count = db["conversations"].count_documents({})
        print(f"   Documents: {doc_count}")
    else:
        print("❌ 'conversations' collection NOT found")
        print("   Creating collection...")
        
        # Create a test document
        test_doc = {
            "conversation_id": "test-001",
            "user_id": "test_user",
            "title": "Test Conversation",
            "created_at": "2025-12-05T10:00:00Z",
            "messages": []
        }
        
        result = db["conversations"].insert_one(test_doc)
        print(f"✅ Test document inserted: {result.inserted_id}")
        
        # Verify collection was created
        collections = db.list_collection_names()
        if "conversations" in collections:
            print("✅ 'conversations' collection NOW EXISTS!")
        else:
            print("❌ Collection still not found after insert")
            
except Exception as e:
    print(f"❌ Collection check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create indexes
print("\n🔄 Creating indexes...")
try:
    db["conversations"].create_index("conversation_id", unique=True)
    db["conversations"].create_index("user_id")
    db["conversations"].create_index([("created_at", -1)])
    print("✅ Indexes created successfully")
    
    # List indexes
    indexes = db["conversations"].list_indexes()
    print("   Indexes:")
    for idx in indexes:
        print(f"     - {idx['name']}: {idx['key']}")
        
except Exception as e:
    print(f"❌ Index creation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! MongoDB is ready.")
print("=" * 60)
print("\n📊 Summary:")
print(f"  ✅ MongoDB Connected")
print(f"  ✅ Database: {mongodb_db}")
print(f"  ✅ Collection: conversations")
print(f"  ✅ Indexes: Created")
print("\n✨ You can now use the conversation service!")
