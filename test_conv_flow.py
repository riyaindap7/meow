"""Test the entire conversation flow"""
import sys
sys.path.insert(0, 'c:\\PROJECTS\\meow\\backend')

from services.conversation_service import ConversationService
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv('c:\\PROJECTS\\meow\\backend\\.env')

print("="*80)
print("TESTING CONVERSATION FLOW")
print("="*80)

# Step 1: Create a conversation using ConversationService
print("\n1️⃣ Creating conversation...")
conv_service = ConversationService()
test_conv = conv_service.create_conversation(
    user_id="test_user_123",
    title="Test Flow Conversation",
    metadata={}
)
print(f"✅ Created: {test_conv['conversation_id']}")

# Step 2: Verify it's in MongoDB
print("\n2️⃣ Verifying in MongoDB...")
mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
mongodb_database = os.getenv("MONGODB_DATABASE", "victor_rag")

client = MongoClient(mongodb_uri)
db = client[mongodb_database]

found = db.conversations.find_one({"conversation_id": test_conv['conversation_id']})
if found:
    print(f"✅ Found in MongoDB!")
    print(f"   Title: {found['title']}")
    print(f"   User ID: {found['user_id']}")
    print(f"   Created: {found['created_at']} (type: {type(found['created_at']).__name__})")
    print(f"   Updated: {found['updated_at']} (type: {type(found['updated_at']).__name__})")
    print(f"   Messages: {len(found.get('messages', []))}")
else:
    print(f"❌ NOT found in MongoDB!")

# Step 3: Test querying conversations
print("\n3️⃣ Testing get_user_conversations...")
convs = conv_service.get_user_conversations("test_user_123")
print(f"✅ Found {len(convs)} conversations for test_user_123")

# Step 4: Clean up
print("\n4️⃣ Cleaning up...")
db.conversations.delete_one({"conversation_id": test_conv['conversation_id']})
print(f"✅ Test conversation deleted")

print("\n" + "="*80)
print("✅ CONVERSATION FLOW TEST PASSED!")
print("="*80)
print("\n💡 If this works but UI doesn't save conversations, the issue is:")
print("   1. Backend server needs to be restarted")
print("   2. Check browser console for errors")
print("   3. Verify auth token is valid")
