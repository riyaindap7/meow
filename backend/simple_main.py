import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import pymongo
from pymongo import MongoClient
from bson import ObjectId
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="VICTOR RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:meow@192.168.0.106:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "victor_rag")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5
    temperature: float = 0.1

class CreateConversationRequest(BaseModel):
    title: str
    metadata: Dict = {}

class ConversationMetadata(BaseModel):
    conversation_id: str
    user_id: str
    title: Optional[str]
    created_at: str
    message_count: int

class ListConversationsResponse(BaseModel):
    conversations: List[ConversationMetadata]
    count: int

# Simple auth function
def get_user_from_token(request: Request) -> str:
    """Extract user ID from Authorization header"""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    try:
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
        
        # For development, assume the token IS the user ObjectId
        # In production, you'd decode JWT and extract user_id
        return token
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

@app.get("/health")
async def health_check():
    """Health check - No auth required"""
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/conversations")
async def create_conversation(request: CreateConversationRequest, req: Request):
    """Create new conversation for authenticated user"""
    try:
        user_id = get_user_from_token(req)
        print(f"📝 Creating conversation for user: {user_id}")
        
        conversation_id = str(uuid.uuid4())
        
        conversation = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": request.title,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "archived": False,
            "metadata": request.metadata
        }
        
        result = db.conversations.insert_one(conversation)
        print(f"✅ Created conversation: {conversation_id}")
        
        return ConversationMetadata(
            conversation_id=conversation_id,
            user_id=user_id,
            title=request.title,
            created_at=conversation["created_at"].isoformat(),
            message_count=0
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@app.get("/conversations")
async def list_conversations(req: Request):
    """List conversations for authenticated user only"""
    try:
        user_id = get_user_from_token(req)
        print(f"📚 Fetching conversations for user: {user_id}")
        
        conversations = list(
            db.conversations.find(
                {"user_id": user_id, "archived": False}
            ).sort("updated_at", -1).limit(50)
        )
        
        print(f"✅ Found {len(conversations)} conversations for user {user_id}")
        
        return ListConversationsResponse(
            conversations=[
                ConversationMetadata(
                    conversation_id=conv["conversation_id"],
                    user_id=conv.get("user_id"),
                    title=conv.get("title"),
                    created_at=conv.get("created_at").isoformat(),
                    message_count=len(conv.get("messages", []))
                )
                for conv in conversations
            ],
            count=len(conversations)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, req: Request):
    """Get conversation messages for authenticated user only"""
    try:
        user_id = get_user_from_token(req)
        print(f"📖 Fetching messages for conversation: {conversation_id}, user: {user_id}")
        
        conversation = db.conversations.find_one({
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = conversation.get("messages", [])
        print(f"✅ Found {len(messages)} messages")
        
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "messages": messages,
            "message_count": len(messages)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.post("/ask")
async def ask(request: QueryRequest, req: Request):
    """Ask a question with RAG - Simple version for testing"""
    try:
        user_id = get_user_from_token(req)
        print(f"🔵 RAG Request from user: {user_id}")
        print(f"   Query: {request.query}")
        
        # For now, return a simple response
        answer = f"This is a test response to your question: '{request.query}'. The backend is working and you are authenticated as user {user_id}."
        
        # Store messages in conversation if conversation_id provided
        if request.conversation_id:
            try:
                # Add user message
                user_message = {
                    "message_id": str(uuid.uuid4()),
                    "role": "user",
                    "content": request.query,
                    "created_at": datetime.utcnow(),
                    "sources": []
                }
                
                # Add assistant message
                assistant_message = {
                    "message_id": str(uuid.uuid4()),
                    "role": "assistant", 
                    "content": answer,
                    "created_at": datetime.utcnow(),
                    "sources": []
                }
                
                # Update conversation
                result = db.conversations.update_one(
                    {"conversation_id": request.conversation_id, "user_id": user_id},
                    {
                        "$push": {"messages": {"$each": [user_message, assistant_message]}},
                        "$set": {"updated_at": datetime.utcnow()}
                    }
                )
                
                if result.modified_count > 0:
                    print(f"✅ Messages saved to conversation {request.conversation_id}")
                else:
                    print(f"❌ Failed to save messages")
                    
            except Exception as e:
                print(f"⚠️ Could not save to conversation: {str(e)}")
        
        return {
            "query": request.query,
            "answer": answer,
            "sources": [],
            "conversation_id": request.conversation_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting VICTOR RAG API...")
    print(f"📊 MongoDB URI: {MONGODB_URI}")
    print(f"🗄️  Database: {MONGODB_DATABASE}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)