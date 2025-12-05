from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import time
from pathlib import Path
from urllib.parse import quote
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from .models import (
    QueryRequest, SearchResponse, SearchResult,
    RAGRequest, RAGResponse, HealthResponse
)
from backend.services.mongodb_service import find_documents
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
        
        # Format response
        search_results = [
            SearchResult(
                **result,
                pdf_url=f"/pdf/{quote(result['source'])}"
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

# RAG endpoint (search + LLM generation)
@app.post("/ask", response_model=RAGResponse)
async def ask(request: RAGRequest):
    """Ask a question with RAG (Retrieval-Augmented Generation)"""
    try:
        print(f"\n🔵 RAG Request received")
        print(f"   Query: {request.query}")
        print(f"   Top K: {request.top_k}")
        print(f"   Temperature: {request.temperature}")
        
        milvus_client = get_milvus_client()
        llm_client = get_llm_client()
        
        # Measure total latency
        total_start = time.time()
        
        # Step 1: Retrieve relevant documents
        search_start = time.time()
        search_results = milvus_client.search(
            query=request.query,
            top_k=request.top_k
        )
        search_latency = (time.time() - search_start) * 1000
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for your query"
            )
        
        print(f"Found {len(search_results)} relevant documents")
        
        # Step 2: Generate answer using LLM
        print(f"Generating answer using model: {llm_client.model}")
        llm_start = time.time()
        try:
            answer = await llm_client.generate_answer(
                query=request.query,
                contexts=search_results,
                temperature=request.temperature
            )
        except Exception as llm_err:
            print(f"LLM Error: {str(llm_err)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM generation failed: {str(llm_err)}"
            )
        
        llm_latency = (time.time() - llm_start) * 1000
        
        total_latency = (time.time() - total_start) * 1000
        
        # Format response
        sources = [
            SearchResult(
                **result,
                pdf_url=f"/pdf/{quote(result['source'])}"
            ) for result in search_results
        ]
        
        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            model_used=llm_client.model,
            search_latency_ms=round(search_latency, 2),
            llm_latency_ms=round(llm_latency, 2),
            total_latency_ms=round(total_latency, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ RAG Error: {str(e)}")
        print(error_trace)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG failed: {str(e)}"
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