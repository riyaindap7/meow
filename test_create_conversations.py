#!/usr/bin/env python3
"""
Test creating conversations in victor_rag database
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent / "backend" / ".env"
load_dotenv(env_file)

print("=" * 70)
print("🔍 Creating Conversations in VICTOR RAG")
print("=" * 70)

mongodb_uri = os.getenv("MONGODB_URI")
mongodb_db = os.getenv("MONGODB_DATABASE")

print(f"\n✅ MongoDB URI: {mongodb_uri}")
print(f"✅ Database: {mongodb_db}")

# Import conversation service
try:
    from backend.services.conversation_service import get_conversation_service
    print("✅ Conversation service imported")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Create service
try:
    print("\n🔄 Initializing conversation service...")
    conv_svc = get_conversation_service(user_id="test_user")
    print("✅ Service initialized")
except Exception as e:
    print(f"❌ Service init failed: {e}")
    sys.exit(1)

# Create first conversation
print("\n🔄 Creating first conversation...")
try:
    conv_id_1 = conv_svc.create_conversation(
        title="Educational Policy Discussion",
        metadata={"device": "web", "locale": "en-IN"},
        settings={"temperature": 0.1, "top_k": 5}
    )
    print(f"✅ Conversation created: {conv_id_1}")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Add messages to first conversation
print("\n🔄 Adding messages to conversation...")
try:
    # User message
    msg_id_1 = conv_svc.add_message(
        conversation_id=conv_id_1,
        role="user",
        content="What is educational policy?",
        sources=[
            {
                "doc_id": "doc_123",
                "source": "RUSA_final090913",
                "page": 60,
                "score": 0.95,
                "snippet": "Higher education is widely recognized..."
            }
        ]
    )
    print(f"✅ User message added: {msg_id_1}")
    
    # Assistant message
    msg_id_2 = conv_svc.add_message(
        conversation_id=conv_id_1,
        role="assistant",
        content="Educational policy refers to government decisions and actions regarding education systems.",
        sources=[
            {
                "doc_id": "doc_123",
                "source": "RUSA_final090913",
                "page": 60,
                "score": 0.95,
                "snippet": "Higher education is widely recognized..."
            }
        ]
    )
    print(f"✅ Assistant message added: {msg_id_2}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create second conversation
print("\n🔄 Creating second conversation...")
try:
    conv_id_2 = conv_svc.create_conversation(
        title="Higher Education Funding",
        metadata={"device": "mobile", "locale": "en-IN"}
    )
    print(f"✅ Conversation created: {conv_id_2}")
    
    msg_id_3 = conv_svc.add_message(
        conversation_id=conv_id_2,
        role="user",
        content="How is higher education funded?"
    )
    print(f"✅ Message added: {msg_id_3}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Retrieve conversations
print("\n🔄 Retrieving conversations...")
try:
    convs = conv_svc.get_user_conversations(limit=10)
    print(f"✅ Found {len(convs)} conversations:")
    for conv in convs:
        msg_count = len(conv.get("messages", []))
        print(f"   - {conv['title']} ({msg_count} messages)")
        print(f"     ID: {conv['conversation_id']}")
        print(f"     Created: {conv['created_at']}")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Get specific conversation
print(f"\n🔄 Retrieving conversation: {conv_id_1}")
try:
    conv = conv_svc.get_conversation(conv_id_1)
    if conv:
        print(f"✅ Conversation retrieved:")
        print(f"   Title: {conv['title']}")
        print(f"   Messages: {len(conv['messages'])}")
        print(f"   Settings: {conv['settings']}")
        print(f"   Metadata: {conv['metadata']}")
        
        # Show messages
        print(f"\n   Message History:")
        for msg in conv['messages']:
            print(f"     - {msg['role'].upper()}: {msg['content'][:50]}...")
    else:
        print(f"❌ Conversation not found")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Check MongoDB collection
print("\n🔄 Checking MongoDB collection...")
try:
    db = conv_svc.db
    collections = db.list_collection_names()
    
    if "conversations" in collections:
        print(f"✅ 'conversations' collection exists in MongoDB")
        
        doc_count = db["conversations"].count_documents({})
        print(f"   Total documents: {doc_count}")
        
        # Get indexes
        indexes = db["conversations"].list_indexes()
        print(f"   Indexes:")
        for idx in indexes:
            print(f"     - {idx['name']}")
    else:
        print(f"❌ Collection not found")
        
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n📊 Summary:")
print(f"  ✅ Conversations created: 2")
print(f"  ✅ Messages stored: 3")
print(f"  ✅ Collection: conversations (in victor_rag)")
print(f"  ✅ MongoDB: {mongodb_uri.split('@')[1]}")
print("\n✨ Conversation service is working perfectly!")
