#!/usr/bin/env python3
"""
Test script to verify conversations collection is created in MongoDB
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path.parent))

# Load environment variables
env_file = Path(__file__).parent / "backend" / ".env"
load_dotenv(env_file)

print("=" * 60)
print("🔍 VICTOR RAG Conversations Collection Test")
print("=" * 60)

# Check environment variables
print("\n📋 Environment Variables:")
print(f"  MONGODB_URI: {os.getenv('MONGODB_URI', 'NOT SET')}")
print(f"  MONGODB_DATABASE: {os.getenv('MONGODB_DATABASE', 'NOT SET')}")

# Import and test conversation service
try:
    from backend.services.conversation_service import get_conversation_service
    print("\n✅ Conversation service imported successfully")
except Exception as e:
    print(f"\n❌ Failed to import conversation service: {e}")
    sys.exit(1)

# Initialize conversation service
try:
    print("\n🔄 Initializing conversation service...")
    conv_svc = get_conversation_service(user_id="test_user")
    print("✅ Conversation service initialized")
except Exception as e:
    print(f"❌ Failed to initialize conversation service: {e}")
    sys.exit(1)

# Check if collection exists
try:
    print("\n🔄 Checking MongoDB collections...")
    db = conv_svc.db
    collections = db.list_collection_names()
    print(f"✅ Collections in '{db.name}' database:")
    for col in collections:
        print(f"   - {col}")
    
    if "conversations" in collections:
        print("✅ 'conversations' collection exists!")
        
        # Check document count
        doc_count = db["conversations"].count_documents({})
        print(f"   Documents: {doc_count}")
        
        # Get indexes
        indexes = db["conversations"].list_indexes()
        print(f"   Indexes:")
        for idx in indexes:
            print(f"     - {idx['key']}")
    else:
        print("❌ 'conversations' collection NOT found")
        print("   Creating collection now...")
        
        # Try creating a conversation to initialize collection
        try:
            conv_id = conv_svc.create_conversation(
                title="Test Conversation",
                metadata={"device": "test", "locale": "en-IN"}
            )
            print(f"✅ Collection created with first conversation: {conv_id}")
            
            # Verify collection was created
            collections = db.list_collection_names()
            if "conversations" in collections:
                print("✅ 'conversations' collection now exists!")
            else:
                print("❌ Collection still not found")
        except Exception as e:
            print(f"❌ Failed to create conversation: {e}")
            sys.exit(1)

except Exception as e:
    print(f"❌ Error checking collections: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ Test completed successfully!")
print("=" * 60)
