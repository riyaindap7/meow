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
from .milvus_client import get_milvus_client
from .llm_client import get_llm_client

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting up API...")
    try:
        get_milvus_client()
        print("✅ Milvus client initialized")
    except Exception as e:
        print(f"⚠️  Milvus initialization warning: {e}")
    
    try:
        get_llm_client()
        print("✅ LLM client (OpenRouter) initialized")
    except Exception as e:
        print(f"⚠️  LLM client initialization warning: {e}")
    
    yield
    print("👋 Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="PDF RAG API",
    description="Query PDF documents using vector search, BM25, or hybrid search",
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
            embedding_model=milvus_client.embedding_model_name,
            hybrid_enabled=health.get("hybrid_enabled", False)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    """Search using vector, BM25, or hybrid method"""
    try:
        milvus_client = get_milvus_client()
        start_time = time.time()
        
        # Use the method from request
        results = milvus_client.search(
            query=request.query,
            top_k=request.top_k,
            method=request.method
        )
        
        search_latency = (time.time() - start_time) * 1000
        
        search_results = [
            SearchResult(
                text=result.get('text'),
                source=result.get('document_name'),
                page=result.get('page_idx'),
                score=result.get('score'),
                document_id=result.get('document_id'),
                chunk_id=result.get('chunk_id'),
                global_chunk_id=result.get('global_chunk_id'),
                chunk_index=result.get('chunk_index'),
                section_hierarchy=result.get('section_hierarchy'),
                heading_context=result.get('heading_context'),
                char_count=result.get('char_count'),
                word_count=result.get('word_count')
            ) for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            count=len(search_results),
            latency_ms=round(search_latency, 2),
            method=request.method
        )
    
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
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
    """RAG with vector, BM25, or hybrid retrieval"""
    try:
        print(f"\n🔵 RAG Request: {request.query} | Method: {request.method}")
        
        milvus_client = get_milvus_client()
        llm_client = get_llm_client()
        
        total_start = time.time()
        
        # Retrieve with chosen method
        search_start = time.time()
        search_results = milvus_client.search(
            query=request.query,
            top_k=request.top_k,
            method=request.method
        )
        search_latency = (time.time() - search_start) * 1000
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found"
            )
        
        print(f"📚 Found {len(search_results)} results using {request.method} search")
        
        # Step 2: Generate answer using LLM
        print(f"🤖 Generating answer using model: {llm_client.model}")
        llm_start = time.time()
        try:
            answer = await llm_client.generate_answer(
                query=request.query,
                contexts=search_results,
                temperature=request.temperature
            )
        except Exception as llm_err:
            print(f"❌ LLM Error: {str(llm_err)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM generation failed: {str(llm_err)}"
            )
        
        llm_latency = (time.time() - llm_start) * 1000
        total_latency = (time.time() - total_start) * 1000
        
        sources = [
            SearchResult(
                text=result.get('text'),
                source=result.get('document_name'),
                page=result.get('page_idx'),
                score=result.get('score'),
                document_id=result.get('document_id'),
                chunk_id=result.get('chunk_id'),
                global_chunk_id=result.get('global_chunk_id'),
                chunk_index=result.get('chunk_index'),
                section_hierarchy=result.get('section_hierarchy'),
                heading_context=result.get('heading_context'),
                char_count=result.get('char_count'),
                word_count=result.get('word_count')
            ) for result in search_results
        ]
        
        print(f"✅ RAG complete | Search: {search_latency:.0f}ms | LLM: {llm_latency:.0f}ms")
        
        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            model_used=llm_client.model,
            search_latency_ms=round(search_latency, 2),
            llm_latency_ms=round(llm_latency, 2),
            total_latency_ms=round(total_latency, 2),
            method=request.method
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
        "version": "2.0.0",
        "search_methods": ["vector", "sparse", "hybrid"],
        "description": {
            "vector": "Dense semantic search (HNSW)",
            "sparse": "Lexical search (BGE-M3 sparse weights)",
            "hybrid": "RRF fusion of dense + sparse (recommended)"
        },
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "ask": "/ask",
            "pdf": "/pdf/{filename}"
        }
    }

# PDF serving endpoint
@app.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    """Serve PDF files"""
    try:
        pdf_path = Path(__file__).parent.parent / "data" / filename
        
        if not pdf_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found")
        
        return FileResponse(path=str(pdf_path), media_type="application/pdf", filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve PDF: {str(e)}"
        )