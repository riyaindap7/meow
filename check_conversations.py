"""Check conversations in MongoDB"""
from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb://192.168.0.106:27017/")
db = client["victor_rag"]
conversations = db["conversations"]

print("=" * 80)
print("CHECKING CONVERSATIONS IN MONGODB")
print("=" * 80)

# Get all conversations
all_convs = list(conversations.find({}).sort("updated_at", -1).limit(10))

print(f"\n📊 Total conversations in DB: {conversations.count_documents({})}")
print(f"📋 Showing last 10 conversations:\n")

for i, conv in enumerate(all_convs, 1):
    print(f"\n{i}. Conversation ID: {conv.get('conversation_id', 'N/A')}")
    print(f"   User ID: {conv.get('user_id', 'N/A')}")
    print(f"   Title: {conv.get('title', 'N/A')}")
    print(f"   Created At: {conv.get('created_at', 'N/A')} (type: {type(conv.get('created_at')).__name__})")
    print(f"   Updated At: {conv.get('updated_at', 'N/A')} (type: {type(conv.get('updated_at')).__name__})")
    print(f"   Messages: {len(conv.get('messages', []))}")
    print(f"   Archived: {conv.get('archived', False)}")

print("\n" + "=" * 80)
