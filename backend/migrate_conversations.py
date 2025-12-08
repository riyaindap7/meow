"""
Migration script to link existing conversations to actual user ObjectIds
"""
import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

def migrate_conversations_to_users():
    """Link existing conversations with 'default' user_id to actual users"""
    
    # Get MongoDB connection
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://admin:meow@192.168.0.106:27017/")
    mongodb_database = os.getenv("MONGODB_DATABASE", "victor_rag")
    
    client = MongoClient(mongodb_uri)
    db = client[mongodb_database]
    
    users_collection = db["user"]  # Fixed: use 'user' not 'users'
    conversations_collection = db["conversations"]
    
    print("🔄 Starting conversation migration...")
    
    # Get all users
    users = list(users_collection.find({}))
    print(f"👥 Found {len(users)} users:")
    for user in users:
        print(f"   - {user['_id']}: {user['name']} ({user['email']})")
    
    # Get conversations with 'default' user_id
    default_conversations = list(conversations_collection.find({"user_id": "default"}))
    print(f"💬 Found {len(default_conversations)} conversations with 'default' user_id")
    
    if not users:
        print("❌ No users found! Make sure users have signed up first.")
        return
    
    if not default_conversations:
        print("✅ No conversations with 'default' user_id found.")
        return
    
    # Strategy: Assign conversations to users based on creation time or email
    # For demo purposes, let's assign to the first user or ask user
    
    print("\\n📋 Migration options:")
    print("1. Assign all conversations to first user (Durva)")
    print("2. Assign all conversations to second user (Shruti)")
    print("3. Split conversations between users")
    print("4. Skip migration (manual assignment)")
    
    choice = input("\\nChoose option (1-4): ").strip()
    
    if choice == "1":
        # Assign all to first user
        target_user = users[0]
        target_user_id = str(target_user["_id"])
        
        result = conversations_collection.update_many(
            {"user_id": "default"},
            {"$set": {"user_id": target_user_id}}
        )
        
        print(f"✅ Assigned {result.modified_count} conversations to {target_user['name']} ({target_user_id})")
    
    elif choice == "2" and len(users) > 1:
        # Assign all to second user
        target_user = users[1]
        target_user_id = str(target_user["_id"])
        
        result = conversations_collection.update_many(
            {"user_id": "default"},
            {"$set": {"user_id": target_user_id}}
        )
        
        print(f"✅ Assigned {result.modified_count} conversations to {target_user['name']} ({target_user_id})")
    
    elif choice == "3" and len(users) > 1:
        # Split conversations between users
        half = len(default_conversations) // 2
        
        # First half to first user
        first_user = users[0]
        first_user_id = str(first_user["_id"])
        first_conv_ids = [conv["conversation_id"] for conv in default_conversations[:half]]
        
        result1 = conversations_collection.update_many(
            {"conversation_id": {"$in": first_conv_ids}},
            {"$set": {"user_id": first_user_id}}
        )
        
        # Second half to second user
        second_user = users[1]
        second_user_id = str(second_user["_id"])
        second_conv_ids = [conv["conversation_id"] for conv in default_conversations[half:]]
        
        result2 = conversations_collection.update_many(
            {"conversation_id": {"$in": second_conv_ids}},
            {"$set": {"user_id": second_user_id}}
        )
        
        print(f"✅ Assigned {result1.modified_count} conversations to {first_user['name']}")
        print(f"✅ Assigned {result2.modified_count} conversations to {second_user['name']}")
    
    else:
        print("⏭️ Skipping migration. You can manually assign conversations later.")
        return
    
    # Verify migration
    print("\\n🔍 Verifying migration...")
    remaining_default = conversations_collection.count_documents({"user_id": "default"})
    print(f"Remaining conversations with 'default' user_id: {remaining_default}")
    
    for user in users:
        user_id = str(user["_id"])
        user_conversations = conversations_collection.count_documents({"user_id": user_id})
        print(f"{user['name']}: {user_conversations} conversations")
    
    client.close()
    print("\\n✅ Migration completed!")

if __name__ == "__main__":
    migrate_conversations_to_users()