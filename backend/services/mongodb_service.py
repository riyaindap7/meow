# backend/services/mongodb_service.py

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from bson import ObjectId
from typing import Optional, Dict, List, Any
import os
import copy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
backend_dir = Path(__file__).parent.parent
env_file = backend_dir / ".env"
load_dotenv(env_file)

# Lazy initialization
_mongo_client: Optional[MongoClient] = None
_mongo_db: Optional[Database] = None

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))


def get_ist_now() -> datetime:
    """Get current time in Indian Standard Time (timezone-naive for MongoDB compatibility)"""
    # Get UTC time and convert to IST, then remove timezone info for MongoDB
    utc_now = datetime.now(timezone.utc)
    ist_time = utc_now.astimezone(IST)
    # Return timezone-naive datetime in IST
    return ist_time.replace(tzinfo=None)


def serialize_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert MongoDB document to JSON-serializable format"""
    if doc is None:
        return None
    
    # Work on a deep copy to avoid mutating the original
    doc = copy.deepcopy(doc)
    
    # Convert ObjectId to string
    if "_id" in doc and isinstance(doc["_id"], ObjectId):
        doc["_id"] = str(doc["_id"])
    
    # Convert any nested ObjectIds
    for key, value in list(doc.items()):
        if isinstance(value, ObjectId):
            doc[key] = str(value)
        elif isinstance(value, datetime):
            doc[key] = value.isoformat()
        elif isinstance(value, list):
            doc[key] = [
                serialize_doc(item) if isinstance(item, dict) 
                else str(item) if isinstance(item, ObjectId)
                else item.isoformat() if isinstance(item, datetime)
                else item
                for item in value
            ]
        elif isinstance(value, dict):
            doc[key] = serialize_doc(value)
    
    return doc


def serialize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert list of MongoDB documents to JSON-serializable format"""
    return [serialize_doc(doc) for doc in docs]


def prepare_query(query: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string IDs to ObjectIds in query"""
    # Create a copy to avoid mutating the original
    query_copy = query.copy()
    if "_id" in query_copy and isinstance(query_copy["_id"], str):
        try:
            query_copy["_id"] = ObjectId(query_copy["_id"])
        except Exception:
            pass  # Invalid ObjectId string, leave as is
    return query_copy


def get_mongo_client() -> MongoClient:
    """Get MongoDB client singleton"""
    global _mongo_client
    if _mongo_client is None:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        _mongo_client = MongoClient(mongo_uri)
        print(f"MongoDB client initialized: {mongo_uri}")
    return _mongo_client


def get_mongo_db() -> Database:
    """Get MongoDB database singleton"""
    global _mongo_db
    if _mongo_db is None:
        client = get_mongo_client()
        db_name = os.getenv("MONGODB_DATABASE", "victor_rag")
        _mongo_db = client[db_name]
        print(f"MongoDB database initialized: {db_name}")
        # Create indexes on first connection
        _create_indexes()
    return _mongo_db


def _create_indexes():
    """Create necessary indexes for collections"""
    db = _mongo_db
    
    # Documents collection indexes
    documents = db.documents
    documents.create_index([("file_hash", ASCENDING)], unique=True, sparse=True)
    documents.create_index([("filename", ASCENDING)])
    documents.create_index([("category", ASCENDING)])
    documents.create_index([("status", ASCENDING)])
    documents.create_index([("created_at", DESCENDING)])
    documents.create_index([("google_drive_id", ASCENDING)], unique=True, sparse=True)
    documents.create_index([("local_path", ASCENDING)])
    
    # Collections collection indexes
    collections = db.collections
    collections.create_index([("name", ASCENDING)])
    collections.create_index([("created_at", DESCENDING)])
    
    # Sync logs collection indexes
    sync_logs = db.sync_logs
    sync_logs.create_index([("timestamp", DESCENDING)])
    sync_logs.create_index([("sync_type", ASCENDING)])
    
    # Users collection indexes
    users = db.users
    users.create_index([("email", ASCENDING)], unique=True)
    users.create_index([("created_at", DESCENDING)])
    
    # Sessions collection indexes
    sessions = db.sessions
    sessions.create_index([("token", ASCENDING)], unique=True)
    sessions.create_index([("expires_at", ASCENDING)])
    sessions.create_index([("user_id", ASCENDING)])
    
    print("MongoDB indexes created successfully")


# ==================== Document Operations ====================

def insert_document(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert a new document record into MongoDB
    
    Schema:
    {
        "filename": str,
        "file_hash": str,  # SHA256 hash of file content
        "file_size": int,
        "category": str,  # "policies", "circulars", "budgets", "notifications", etc.
        "google_drive_id": str,
        "google_drive_path": str,
        "google_drive_modified_time": datetime,
        "local_path": str,
        "mime_type": str,
        "status": str,  # "synced", "processing", "parsed", "embedded", "error"
        "org_id": str,
        "uploader_id": str,
        "parsed_json_local_path": str,
        "images_local_path": str,
        "milvus_collection": str,  # Milvus collection name for embeddings
        "milvus_ids": List[int],  # IDs of vectors in Milvus
        "chunk_count": int,
        "error_message": str,
        "metadata": Dict,  # Additional flexible metadata
        "created_at": datetime,
        "updated_at": datetime
    }
    """
    db = get_mongo_db()
    documents = db.documents
    
    # Add timestamps
    now = get_ist_now()
    data["created_at"] = data.get("created_at", now)
    data["updated_at"] = now
    
    try:
        result = documents.insert_one(data)
        data["_id"] = result.inserted_id
        print(f"Document inserted: {data.get('filename')} (ID: {result.inserted_id})")
        return serialize_doc(data)
    except Exception as e:
        print(f"Error inserting document: {e}")
        raise


def update_document(filter_query: Dict[str, Any], update_data: Dict[str, Any]) -> bool:
    """Update document record"""
    db = get_mongo_db()
    documents = db.documents
    
    # Convert string IDs to ObjectIds
    filter_query = prepare_query(filter_query)
    
    # Add updated timestamp
    update_data["updated_at"] = get_ist_now()
    
    try:
        result = documents.update_one(filter_query, {"$set": update_data})
        return result.modified_count > 0
    except Exception as e:
        print(f"Error updating document: {e}")
        raise


def find_document(query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a single document"""
    db = get_mongo_db()
    documents = db.documents
    query = prepare_query(query)
    return serialize_doc(documents.find_one(query))


def find_documents(query: Dict[str, Any], limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """Find multiple documents"""
    db = get_mongo_db()
    documents = db.documents
    query = prepare_query(query)
    return serialize_docs(list(documents.find(query).skip(skip).limit(limit).sort("created_at", DESCENDING)))


def delete_document(query: Dict[str, Any]) -> bool:
    """Delete a document"""
    db = get_mongo_db()
    documents = db.documents
    query = prepare_query(query)
    result = documents.delete_one(query)
    return result.deleted_count > 0


def get_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Get document by file hash"""
    return find_document({"file_hash": file_hash})


def get_document_by_drive_id(drive_id: str) -> Optional[Dict[str, Any]]:
    """Get document by Google Drive ID"""
    return find_document({"google_drive_id": drive_id})


def get_documents_by_status(status: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get documents by status"""
    return find_documents({"status": status}, limit=limit)


def get_documents_by_category(category: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get documents by category"""
    return find_documents({"category": category}, limit=limit)


# ==================== Collection Operations ====================

def insert_collection(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert a new collection record
    
    Schema:
    {
        "name": str,
        "description": str,
        "google_drive_folder_id": str,
        "local_folder_path": str,
        "document_count": int,
        "org_id": str,
        "created_at": datetime,
        "updated_at": datetime
    }
    """
    db = get_mongo_db()
    collections = db.collections
    
    now = get_ist_now()
    data["created_at"] = data.get("created_at", now)
    data["updated_at"] = now
    
    try:
        result = collections.insert_one(data)
        data["_id"] = result.inserted_id
        print(f"Collection inserted: {data.get('name')} (ID: {result.inserted_id})")
        return serialize_doc(data)
    except Exception as e:
        print(f"Error inserting collection: {e}")
        raise


def find_collection(query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a single collection"""
    db = get_mongo_db()
    collections = db.collections
    query = prepare_query(query)
    return serialize_doc(collections.find_one(query))


def find_collections(query: Dict[str, Any] = {}, limit: int = 100) -> List[Dict[str, Any]]:
    """Find multiple collections"""
    db = get_mongo_db()
    collections = db.collections
    query = prepare_query(query)
    return serialize_docs(list(collections.find(query).limit(limit).sort("created_at", DESCENDING)))


# ==================== Sync Log Operations ====================

def log_sync_event(sync_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log a sync event
    
    Schema:
    {
        "sync_type": str,  # "full_sync", "incremental_sync", "manual_sync"
        "timestamp": datetime,
        "files_checked": int,
        "files_downloaded": int,
        "files_updated": int,
        "files_deleted": int,
        "bytes_downloaded": int,
        "duration_seconds": float,
        "errors": List[str],
        "details": Dict
    }
    """
    db = get_mongo_db()
    sync_logs = db.sync_logs
    
    log_entry = {
        "sync_type": sync_type,
        "timestamp": get_ist_now(),
        **details
    }
    
    try:
        result = sync_logs.insert_one(log_entry)
        log_entry["_id"] = result.inserted_id
        return serialize_doc(log_entry)
    except Exception as e:
        print(f"Error logging sync event: {e}")
        raise


def get_recent_sync_logs(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent sync logs"""
    db = get_mongo_db()
    sync_logs = db.sync_logs
    return serialize_docs(list(sync_logs.find().limit(limit).sort("timestamp", DESCENDING)))


def get_last_sync_time() -> Optional[datetime]:
    """Get timestamp of last successful sync"""
    logs = get_recent_sync_logs(limit=1)
    if logs:
        return logs[0].get("timestamp")
    return None


# ==================== Statistics ====================

def get_stats() -> Dict[str, Any]:
    """Get database statistics"""
    db = get_mongo_db()
    
    documents = db.documents
    collections = db.collections
    sync_logs = db.sync_logs
    
    return {
        "total_documents": documents.count_documents({}),
        "documents_by_status": {
            "synced": documents.count_documents({"status": "synced"}),
            "processing": documents.count_documents({"status": "processing"}),
            "parsed": documents.count_documents({"status": "parsed"}),
            "embedded": documents.count_documents({"status": "embedded"}),
            "error": documents.count_documents({"status": "error"})
        },
        "total_collections": collections.count_documents({}),
        "total_sync_logs": sync_logs.count_documents({}),
        "last_sync_time": get_last_sync_time()
    }


# Conversation and Message Management Functions
def get_conversations_collection():
    """Get the conversations collection"""
    db = get_mongo_db()
    return db.conversations

def get_messages_collection():
    """Get the messages collection"""
    db = get_mongo_db()
    return db.messages

def create_conversation(conversation_id: str, user_id: str, title: str = None) -> Dict[str, Any]:
    """Create a new conversation with smart title generation"""
    try:
        conversations = get_conversations_collection()
        
        # Generate smart title if none provided
        if not title:
            title = "New Conversation"
        else:
            # Clean and truncate title from first query
            title = title.strip()
            if len(title) > 50:
                title = title[:47] + "..."
        
        conversation_doc = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": title,
            "messages": [],  # Keep for backward compatibility but will be empty
            "created_at": get_ist_now(),
            "updated_at": get_ist_now(),
            "archived": False,
            "metadata": {},
            "context": {}  # For LLM-extracted context
        }
        
        result = conversations.insert_one(conversation_doc)
        conversation_doc["_id"] = str(result.inserted_id)
        
        return serialize_doc(conversation_doc)
        
    except Exception as e:
        print(f"Error creating conversation: {e}")
        return None

def get_conversation(conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a conversation by ID and user"""
    try:
        conversations = get_conversations_collection()
        conversation = conversations.find_one({
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        return serialize_doc(conversation)
    except Exception as e:
        print(f"Error getting conversation: {e}")
        return None

def add_message(conversation_id: str, user_id: str, role: str, content: str, metadata: Dict = None) -> bool:
    """Add a message to the separate messages collection"""
    try:
        messages = get_messages_collection()
        conversations = get_conversations_collection()
        
        # Create message document
        import uuid
        message_doc = {
            "message_id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": get_ist_now(),
            "metadata": metadata or {}
        }
        
        # Insert message
        messages.insert_one(message_doc)
        
        # Update conversation timestamp
        conversations.update_one(
            {"conversation_id": conversation_id, "user_id": user_id},
            {"$set": {"updated_at": get_ist_now()}}
        )
        
        return True
        
    except Exception as e:
        print(f"Error adding message: {e}")
        return False

def get_last_messages(conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get the last N messages from a conversation"""
    try:
        messages = get_messages_collection()
        
        # Get messages sorted by timestamp (oldest first for proper order)
        message_docs = list(messages.find({
            "conversation_id": conversation_id
        }).sort("timestamp", ASCENDING).limit(limit * 2))  # Get more to be safe
        
        # Take the last 'limit' messages
        if len(message_docs) > limit:
            message_docs = message_docs[-limit:]
        
        return serialize_docs(message_docs)
        
    except Exception as e:
        print(f"Error getting last messages: {e}")
        return []

def get_user_conversations(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get all conversations for a user"""
    try:
        conversations = get_conversations_collection()
        
        conversation_docs = list(conversations.find({
            "user_id": user_id,
            "archived": {"$ne": True}
        }).sort("updated_at", DESCENDING).limit(limit))
        
        return serialize_docs(conversation_docs)
        
    except Exception as e:
        print(f"Error getting user conversations: {e}")
        return []

def update_conversation_title(conversation_id: str, user_id: str, title: str) -> bool:
    """Update conversation title based on first user query"""
    try:
        conversations = get_conversations_collection()
        
        # Clean and format title
        clean_title = title.strip()
        if len(clean_title) > 50:
            clean_title = clean_title[:47] + "..."
        
        result = conversations.update_one(
            {
                "conversation_id": conversation_id,
                "user_id": user_id
            },
            {
                "$set": {
                    "title": clean_title,
                    "updated_at": get_ist_now()
                }
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error updating conversation title: {e}")
        return False

def update_conversation_context(conversation_id: str, user_id: str, context_data: Dict[str, Any]) -> bool:
    """Update conversation with LLM-extracted context"""
    try:
        conversations = get_conversations_collection()
        
        result = conversations.update_one(
            {
                "conversation_id": conversation_id,
                "user_id": user_id
            },
            {
                "$set": {
                    "context": context_data,
                    "updated_at": get_ist_now()
                }
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error updating conversation context: {e}")
        return False

def update_conversation_title(conversation_id: str, user_id: str, title: str) -> bool:
    """Update conversation title based on first user message"""
    try:
        conversations = get_conversations_collection()
        
        # Clean and truncate title
        title = title.strip()
        if len(title) > 50:
            title = title[:47] + "..."
        
        # Only update if current title is "New Conversation"
        result = conversations.update_one(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "title": "New Conversation"  # Only update if still default
            },
            {
                "$set": {
                    "title": title,
                    "updated_at": get_ist_now()
                }
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error updating conversation title: {e}")
        return False

def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Delete a conversation and all its messages"""
    try:
        conversations = get_conversations_collection()
        messages = get_messages_collection()
        
        # Delete all messages for this conversation
        messages.delete_many({"conversation_id": conversation_id})
        
        # Delete the conversation
        result = conversations.delete_one({
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        
        return result.deleted_count > 0
        
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return False


# Create a service instance for easy importing
class MongoDBService:
    """Service class for conversation and message management"""
    
    def create_conversation(self, conversation_id: str, user_id: str, title: str = "New Conversation"):
        return create_conversation(conversation_id, user_id, title)
    
    def get_conversation(self, conversation_id: str, user_id: str):
        return get_conversation(conversation_id, user_id)
    
    def add_message(self, conversation_id: str, user_id: str, role: str, content: str, metadata: Dict = None):
        return add_message(conversation_id, user_id, role, content, metadata)
    
    def get_last_messages(self, conversation_id: str, limit: int = 10):
        return get_last_messages(conversation_id, limit)
    
    def get_user_conversations(self, user_id: str, limit: int = 50):
        return get_user_conversations(user_id, limit)
    
    def update_conversation_context(self, conversation_id: str, user_id: str, context_data: Dict[str, Any]):
        return update_conversation_context(conversation_id, user_id, context_data)
    
    def update_conversation_title(self, conversation_id: str, user_id: str, title: str):
        return update_conversation_title(conversation_id, user_id, title)
    
    def delete_conversation(self, conversation_id: str, user_id: str):
        return delete_conversation(conversation_id, user_id)

# Create service instance
mongodb_service = MongoDBService()

