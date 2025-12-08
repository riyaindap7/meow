"""
Conversation Service for VICTOR RAG
Manages conversations with enhanced schema including messages, context, and metadata
MongoDB collection: victor_rag.conversations
"""

import os
from typing import List, Dict, Optional
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()

class ConversationService:
    def __init__(self):
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        mongodb_database = os.getenv("MONGODB_DATABASE", "victor_rag")
        
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[mongodb_database]
        self.conversations = self.db["conversations"]
        
        print(f"✅ ConversationService initialized with database: {mongodb_database}")
    
    def create_conversation(self, user_id: str, title: str, metadata: Dict = None) -> Dict:
        """Create a new conversation for a specific user"""
        conversation_id = str(uuid.uuid4())
        
        conversation = {
            "conversation_id": conversation_id,
            "user_id": user_id,  # This should be the MongoDB ObjectId as string
            "title": title,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "archived": False,
            "metadata": metadata or {}
        }
        
        result = self.conversations.insert_one(conversation)
        conversation["_id"] = result.inserted_id
        
        print(f"📝 Created conversation {conversation_id} for user {user_id}")
        return conversation
    
    def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all conversations for a specific user"""
        print(f"🔍 Looking for conversations with user_id: {user_id}")
        
        conversations = list(
            self.conversations.find(
                {"user_id": user_id, "archived": False}  # Filter by user ObjectId
            ).sort("updated_at", -1).limit(limit)
        )
        
        print(f"📚 Found {len(conversations)} conversations for user {user_id}")
        if len(conversations) == 0:
            # Debug: Check all conversations to see what user_ids exist
            all_convs = list(self.conversations.find({}, {"user_id": 1, "title": 1}).limit(5))
            print(f"🔍 Debug - Sample conversations in DB: {all_convs}")
        
        return conversations
    
    def get_conversation(self, conversation_id: str, user_id: str = None) -> Optional[Dict]:
        """Get a specific conversation, optionally filtered by user"""
        query = {"conversation_id": conversation_id}
        
        # If user_id provided, ensure user owns the conversation
        if user_id:
            query["user_id"] = user_id
        
        conversation = self.conversations.find_one(query)
        
        if conversation:
            print(f"📖 Retrieved conversation {conversation_id}")
        else:
            print(f"❌ Conversation {conversation_id} not found for user {user_id}")
        
        return conversation
    
    def add_message(self, conversation_id: str, user_id: str, role: str, content: str, sources: List = None, metadata: Dict = None):
        """Add a message to a conversation (with user ownership check)"""
        
        # Verify user owns the conversation
        conversation = self.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found for user {user_id}")
        
        message = {
            "message_id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "sources": sources or [],
            "metadata": metadata or {}
        }
        
        result = self.conversations.update_one(
            {"conversation_id": conversation_id, "user_id": user_id},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        if result.modified_count > 0:
            print(f"✅ Added {role} message to conversation {conversation_id}")
        else:
            print(f"❌ Failed to add message to conversation {conversation_id}")
        
        return message
    
    def update_conversation_title(self, conversation_id: str, user_id: str, title: str) -> bool:
        """Update conversation title (with user ownership check)"""
        result = self.conversations.update_one(
            {"conversation_id": conversation_id, "user_id": user_id},
            {
                "$set": {
                    "title": title,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0
    
    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation (with user ownership check)"""
        result = self.conversations.update_one(
            {"conversation_id": conversation_id, "user_id": user_id},
            {
                "$set": {
                    "archived": True,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0

# Global instance
_conversation_service = None

def get_conversation_service() -> ConversationService:
    """Get or create ConversationService singleton"""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service
