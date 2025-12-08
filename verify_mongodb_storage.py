#!/usr/bin/env python3
"""Verify what's being stored in MongoDB"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
import json

# Load environment variables
env_file = Path(__file__).parent / "backend" / ".env"
load_dotenv(env_file)

print("=" * 70)
print("🔍 MongoDB Storage Verification")
print("=" * 70)

mongodb_uri = os.getenv("MONGODB_URI")
mongodb_db = os.getenv("MONGODB_DATABASE")

print(f"\n✅ MongoDB URI: {mongodb_uri}")
print(f"✅ Database: {mongodb_db}")

try:
    # Connect to MongoDB
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
    db = client[mongodb_db]
    
    print(f"\n🔄 Connecting to MongoDB...")
    client.admin.command('ping')
    print(f"✅ Connected to MongoDB")
    
    # Check if conversations collection exists
    collections = db.list_collection_names()
    print(f"\n📚 Collections in database:")
    for col in collections:
        count = db[col].count_documents({})
        print(f"   - {col}: {count} documents")
    
    if "conversations" not in collections:
        print(f"\n❌ 'conversations' collection not found!")
        sys.exit(1)
    
    print(f"\n" + "=" * 70)
    print(f"💾 Conversation Records")
    print(f"=" * 70)
    
    # Get all conversations
    conversations = list(db["conversations"].find({}, {"_id": 0}))
    
    if not conversations:
        print(f"\n⚠️  No conversations found in database")
    else:
        print(f"\n✅ Found {len(conversations)} conversation(s)")
        
        for i, conv in enumerate(conversations, 1):
            print(f"\n📋 Conversation {i}:")
            print(f"   ID: {conv.get('conversation_id')}")
            print(f"   Title: {conv.get('title')}")
            print(f"   Created: {conv.get('created_at')}")
            print(f"   User ID: {conv.get('user_id')}")
            print(f"   Archived: {conv.get('archived', False)}")
            
            messages = conv.get('messages', [])
            print(f"\n   📝 Messages: {len(messages)} total")
            
            for j, msg in enumerate(messages, 1):
                print(f"\n      Message {j}:")
                print(f"        Role: {msg.get('role')}")
                print(f"        Content: {msg.get('content')[:80]}...")
                print(f"        Created: {msg.get('created_at')}")
                
                sources = msg.get('sources', [])
                if sources:
                    print(f"        Sources: {len(sources)} documents")
                    for src in sources[:2]:  # Show first 2
                        print(f"          - {src.get('source')} (page {src.get('page')})")
            
            # Show settings and metadata
            settings = conv.get('settings', {})
            metadata = conv.get('metadata', {})
            
            if settings:
                print(f"\n   ⚙️  Settings:")
                print(f"      Temperature: {settings.get('temperature')}")
                print(f"      Top K: {settings.get('top_k')}")
            
            if metadata:
                print(f"\n   📋 Metadata:")
                print(f"      {json.dumps(metadata, indent=6)}")
    
    print(f"\n" + "=" * 70)
    print(f"✨ MongoDB Storage Verification Complete!")
    print(f"=" * 70)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
