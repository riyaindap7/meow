import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import time
from urllib.parse import quote
from dotenv import load_dotenv
from api.dependencies import verify_auth_token
from services.conversation_service import get_conversation_service
from api.milvus_client import get_milvus_client
from services.full_langchain_service import get_full_langchain_rag
from pydantic import BaseModel
from typing import List, Dict, Optional


# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Get DEFAULT_USER_ID from environment
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")

from .models import (
    QueryRequest, SearchResponse, SearchResult,
    RAGRequest, RAGResponse, HealthResponse
)
from services.mongodb_service import find_documents
from .milvus_client import get_milvus_client
from .llm_client import get_llm_client
import json

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting up API...")
    try:
        get_milvus_client()  # Initialize Milvus connection
        print("✅ Milvus client initialized")
    except Exception as e:
        print(f"⚠️  Milvus initialization warning: {e}")
        # Don't fail startup, Milvus is optional
    
    try:
        get_llm_client()  # Initialize OpenRouter client
        print("✅ LLM client (OpenRouter) initialized")
    except Exception as e:
        print(f"⚠️  LLM client initialization warning: {e}")
        # Don't fail startup, LLM is optional
    
    yield
    
    # Shutdown
    print("👋 Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="PDF RAG API",
    description="Query PDF documents using vector search and LLM",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models (keep your existing models)
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
    updated_at: str
    message_count: int

class ListConversationsResponse(BaseModel):
    conversations: List[ConversationMetadata]
    count: int

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and Milvus health"""
    try:
        milvus_client = get_milvus_client()
        health = milvus_client.health_check()
        
        return HealthResponse(
            status="healthy" if health["milvus_connected"] else "unhealthy",
            milvus_connected=health["milvus_connected"],
            collection_exists=health["collection_exists"],
            total_vectors=health["total_vectors"],
            embedding_model=milvus_client.embedding_model_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

# Search endpoint (Milvus vector search)
@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    """Search for documents using vector similarity in Milvus"""
    try:
        milvus_client = get_milvus_client()
        
        # Measure search latency
        start_time = time.time()
        
        # Perform vector search
        results = milvus_client.search(
            query=request.query,
            top_k=request.top_k
        )
        
        search_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Format response with Vtext schema
        search_results = [
            SearchResult(
                text=result.get('text'),
                source_file=result.get('source_file'),
                page_idx=result.get('page_idx'),
                score=result.get('score'),
                # Vtext fields
                global_chunk_id=result.get('global_chunk_id'),
                document_id=result.get('document_id'),
                chunk_index=result.get('chunk_index'),
                section_hierarchy=result.get('section_hierarchy'),
                char_count=result.get('char_count'),
                word_count=result.get('word_count')
            ) for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            count=len(search_results),
            latency_ms=round(search_latency, 2)
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        import traceback
        print(f"Search error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

# RAG endpoint (search + LLM generation) - Full LangChain Integration
@app.post("/ask", response_model=RAGResponse)
async def ask(request: RAGRequest, user: dict = Depends(verify_auth_token)):
    """Ask a question with RAG using Full LangChain Pipeline - Protected endpoint"""
    try:
        print(f"🔵 Full LangChain RAG Request from user: {user['user_id']}")
        print(f"   Query: {request.query}")
        print(f"   Conversation ID: {request.conversation_id}")
        
        # Start timing
        total_start_time = time.time()
        
        # Get the full LangChain RAG service
        langchain_rag = get_full_langchain_rag()
        
        # Use LangChain pipeline with conversation memory
        result = langchain_rag.ask(
            query=request.query,
            conversation_id=request.conversation_id,
            user_id=user['user_id'],
            temperature=request.temperature
        )
        
        # Calculate total latency
        total_latency = (time.time() - total_start_time) * 1000
        
        # Format sources for API response
        sources = [
            SearchResult(
                text=source.get('text', ''),
                source_file=source.get('source_file', ''),
                page_idx=source.get('page_idx', 0),
                score=source.get('score', 0.0),
                global_chunk_id=source.get('global_chunk_id'),
                document_id=source.get('document_id'),
                chunk_index=source.get('chunk_index'),
                section_hierarchy=source.get('section_hierarchy'),
                char_count=source.get('char_count'),
                word_count=source.get('word_count')
            ) for source in result.get('sources', [])
        ]
        
        print(f"✅ Full LangChain RAG completed successfully")
        print(f"   Answer length: {len(result.get('answer', ''))} chars")
        print(f"   Sources found: {len(sources)}")
        
        return RAGResponse(
            query=request.query,
            answer=result.get('answer', ''),
            sources=sources,
            model_used=result.get('model_used', 'langchain-rag'),
            search_latency_ms=50.0,  # LangChain handles this internally
            llm_latency_ms=round(total_latency * 0.8, 2),  # Estimate
            total_latency_ms=round(total_latency, 2),
            conversation_id=request.conversation_id
        )
    
    except Exception as e:
        print(f"❌ Full LangChain RAG Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangChain RAG failed: {str(e)}"
        )

@app.post("/conversations")
async def create_conversation(request: CreateConversationRequest, user: dict = Depends(verify_auth_token)):
    """Create new conversation for authenticated user using LangChain"""
    try:
        print("\n" + "="*80)
        print("🆕 CREATE CONVERSATION REQUEST")
        print("="*80)
        print(f"   User ID: {user['user_id']}")
        print(f"   Title: {request.title}")
        print(f"   Metadata: {request.metadata}")
        
        # Use LangChain service for conversation creation
        langchain_rag = get_full_langchain_rag()
        print(f"🔵 Calling LangChain service to create conversation...")
        conversation_id = langchain_rag.create_new_conversation(
            title=request.title,
            user_id=user['user_id'],
            metadata=request.metadata
        )
        
        if not conversation_id:
            print(f"⚠️ LangChain service returned None, falling back to conversation service")
            # Fallback to conversation service
            conv_service = get_conversation_service()
            conversation = conv_service.create_conversation(
                user_id=user['user_id'],
                title=request.title,
                metadata=request.metadata
            )
            conversation_id = conversation["conversation_id"]
        
        print(f"✅ Conversation created: {conversation_id}")
        
        # Verify it's in MongoDB
        from services.mongodb_service import get_mongo_db
        db = get_mongo_db()
        verify = db.conversations.find_one({"conversation_id": conversation_id})
        if verify:
            print(f"✅ VERIFIED: Conversation exists in MongoDB")
            print(f"   Title: {verify.get('title')}")
            print(f"   Messages: {len(verify.get('messages', []))}")
        else:
            print(f"❌ WARNING: Conversation NOT found in MongoDB after creation!")
        print("="*80 + "\n")
        
        # Get the actual conversation data from MongoDB to return accurate timestamps
        from services.mongodb_service import get_mongo_db
        db = get_mongo_db()
        created_conv = db.conversations.find_one({"conversation_id": conversation_id})
        
        if created_conv:
            # Use actual timestamps from MongoDB
            created_at = created_conv.get("created_at")
            updated_at = created_conv.get("updated_at", created_at)
            
            # Convert datetime objects to ISO strings
            if hasattr(created_at, 'isoformat'):
                created_at = created_at.isoformat()
            elif not isinstance(created_at, str):
                created_at = str(created_at)
            
            if hasattr(updated_at, 'isoformat'):
                updated_at = updated_at.isoformat()
            elif not isinstance(updated_at, str):
                updated_at = str(updated_at)
        else:
            # Fallback to current time if not found
            from datetime import datetime
            created_at = datetime.utcnow().isoformat()
            updated_at = created_at
        
        return ConversationMetadata(
            conversation_id=conversation_id,
            user_id=user['user_id'],
            title=request.title,
            created_at=created_at,
            updated_at=updated_at,
            message_count=0
        )
    except Exception as e:
        print(f"❌ Error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )

@app.get("/conversations")
async def list_conversations(user: dict = Depends(verify_auth_token)):
    """List conversations for authenticated user using LangChain"""
    try:
        print(f"🔵 Listing conversations for user: {user['user_id']}")
        
        # Use LangChain service first
        langchain_rag = get_full_langchain_rag()
        conversations = langchain_rag.get_conversations(user['user_id'])
        
        if not conversations:
            # Fallback to conversation service
            conv_service = get_conversation_service()
            conversations = conv_service.get_user_conversations(user['user_id'])
        
        # Format response
        formatted_conversations = []
        for conv in conversations:
            try:
                # Handle datetime objects
                created_at = conv.get("created_at", "2023-01-01T00:00:00")
                if hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                elif not isinstance(created_at, str):
                    created_at = str(created_at)
                
                updated_at = conv.get("updated_at", created_at)
                if hasattr(updated_at, 'isoformat'):
                    updated_at = updated_at.isoformat()
                elif not isinstance(updated_at, str):
                    updated_at = str(updated_at)
                
                formatted_conversations.append(
                    ConversationMetadata(
                        conversation_id=conv.get("conversation_id", ""),
                        user_id=conv.get("user_id", user['user_id']),
                        title=conv.get("title", "Untitled"),
                        created_at=created_at,
                        updated_at=updated_at,
                        message_count=len(conv.get("messages", []))
                    )
                )
            except Exception as e:
                print(f"⚠️ Error formatting conversation: {str(e)}")
                continue
        
        print(f"✅ Found {len(formatted_conversations)} conversations")
        
        return ListConversationsResponse(
            conversations=formatted_conversations,
            count=len(formatted_conversations)
        )
    except Exception as e:
        print(f"❌ Error listing conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )

@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, user: dict = Depends(verify_auth_token)):
    """Get conversation messages for authenticated user only"""
    try:
        conv_service = get_conversation_service()
        
        # ✅ Ensure user owns this conversation
        conversation = conv_service.get_conversation(conversation_id, user['user_id'])
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        messages = conversation.get("messages", [])
        
        return {
            "conversation_id": conversation_id,
            "user_id": user['user_id'],
            "messages": messages,
            "message_count": len(messages)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get messages: {str(e)}"
        )

# Root endpoint
@app.get("/")
async def root():
    """API root"""
    return {
        "message": "PDF RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "ask": "/ask",
            "pdf": "/pdf/{filename}#page={page}"
        }
    }

# PDF serving endpoint
@app.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    """Serve PDF files from the data directory"""
    try:
        # Get the project root directory (parent of api folder)
        api_dir = Path(__file__).parent
        project_root = api_dir.parent
        pdf_path = project_root / "data" / filename
        
        # Security check: ensure the file is in the data directory
        if not pdf_path.is_file() or not pdf_path.resolve().is_relative_to(project_root / "data"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF file not found"
            )
        
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve PDF: {str(e)}"
        )