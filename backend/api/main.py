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
    RAGRequest, RAGResponse, HealthResponse,
    RerankRequest, ComparisonResponse, ComparisonResult, ComparisonMetrics
)
from backend.services.mongodb_service import find_documents
from .milvus_client import get_milvus_client
from .llm_client import get_llm_client
from backend.services.self_query_retriever import create_self_query_retriever
import json

# Global self-query retriever instance
_self_query_retriever = None

def get_self_query_retriever():
    """Get or create self-query retriever singleton"""
    global _self_query_retriever
    if _self_query_retriever is None:
        _self_query_retriever = create_self_query_retriever(
            collection_name="VictorText",
            top_k=5,
            rerank=True,
            enable_llm_decomposition=True  # ✅ Enable LLM-based query decomposition
        )
        print("✅ Self-Query Retriever initialized (LLM decomposition enabled)")
    return _self_query_retriever

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
    
    try:
        get_self_query_retriever()
        print("✅ Self-Query Retriever initialized")
    except Exception as e:
        print(f"⚠️  Self-Query Retriever initialization warning: {e}")
    
    yield
    print("👋 Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="PDF RAG API with Hybrid Search",
    description="Query PDF documents using vector, sparse, or hybrid search with self-query capabilities",
    version="2.0.0",
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

# Import and include routers
from backend.api.routers import collections, documents, search, chat, upload, sync, scraper
from backend.api.routers.parse_marker import router as parse_marker_router
from backend.api.routers import self_query_search

# Include all routers
app.include_router(collections.router, prefix="/api/collections", tags=["Collections"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(sync.router, prefix="/api/sync", tags=["Sync"])
app.include_router(scraper.router, prefix="/api/scraper", tags=["Scraper"])
app.include_router(parse_marker_router, prefix="/api/parse", tags=["Parse"])
app.include_router(self_query_search.router, prefix="/api/self-query", tags=["Self-Query"])

# ============================================================================
# HEALTH CHECK
# ============================================================================

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
            hybrid_enabled=health.get("hybrid_enabled", False),
            has_dense_field=health.get("has_dense_field"),
            has_sparse_field=health.get("has_sparse_field"),
            reranker_enabled=health.get("reranker_enabled"),
            reranker_model=health.get("reranker_model")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    """Search using vector, sparse, or hybrid method"""
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

@app.post("/search/self-query", response_model=SearchResponse)
async def search_self_query(request: QueryRequest):
    """
    🆕 Self-Query Search - Automatically extracts metadata filters from queries
    Supports vector, sparse, or hybrid search methods
    """
    try:
        print(f"\n{'🔍'*40}")
        print(f"🔍 SELF-QUERY SEARCH REQUEST")
        print(f"{'🔍'*40}")
        print(f"📊 Search Method: {request.method}")
        
        retriever = get_self_query_retriever()
        llm_client = get_llm_client()  # ✅ Get LLM client for decomposition
        
        start_time = time.time()
        
        # Perform self-query retrieval with specified method
        results, decomposition = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            llm_client=llm_client,  # ✅ Pass LLM client for intelligent decomposition
            verbose=True,
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
        
        print(f"✅ Self-Query Search Complete: {len(search_results)} results in {search_latency:.2f}ms\n")
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            count=len(search_results),
            latency_ms=round(search_latency, 2),
            method=f"self-query-{request.method}"
        )
    
    except Exception as e:
        import traceback
        print(f"❌ Self-query search error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Self-query search failed: {str(e)}"
        )

@app.post("/search/rerank", response_model=SearchResponse)
async def search_with_rerank(request: RerankRequest):
    """Search with optional re-ranking"""
    try:
        milvus_client = get_milvus_client()
        start_time = time.time()
        
        if request.use_reranker:
            results = milvus_client.search_with_rerank(
                query=request.query,
                top_k=request.top_k,
                method=request.method
            )
        else:
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
    
    except Exception as e:
        import traceback
        print(f"Search error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/search/compare", response_model=ComparisonResponse)
async def compare_search_methods(request: RerankRequest):
    """Compare search vs re-ranked search side-by-side"""
    try:
        milvus_client = get_milvus_client()
        comparison = milvus_client.get_comparison_data(
            query=request.query,
            top_k=request.top_k,
            method=request.method
        )
        
        before_results = [
            ComparisonResult(
                rank=i+1,
                text=result.get('text'),
                source=result.get('document_name'),
                page=result.get('page_idx'),
                vector_score=result.get('score'),
                rerank_score=None,
                score_delta=None,
                document_id=result.get('document_id'),
                chunk_id=result.get('chunk_id'),
                global_chunk_id=result.get('global_chunk_id'),
                chunk_index=result.get('chunk_index'),
                section_hierarchy=result.get('section_hierarchy'),
                heading_context=result.get('heading_context'),
                char_count=result.get('char_count'),
                word_count=result.get('word_count')
            ) for i, result in enumerate(comparison['before_reranking'])
        ]
        
        after_results = [
            ComparisonResult(
                rank=i+1,
                text=result.get('text'),
                source=result.get('document_name'),
                page=result.get('page_idx'),
                vector_score=result.get('vector_score', result.get('score')),
                rerank_score=result.get('rerank_score'),
                score_delta=result.get('rerank_score', 0) - result.get('vector_score', result.get('score', 0)),
                document_id=result.get('document_id'),
                chunk_id=result.get('chunk_id'),
                global_chunk_id=result.get('global_chunk_id'),
                chunk_index=result.get('chunk_index'),
                section_hierarchy=result.get('section_hierarchy'),
                heading_context=result.get('heading_context'),
                char_count=result.get('char_count'),
                word_count=result.get('word_count')
            ) for i, result in enumerate(comparison['after_reranking'])
        ]
        
        metrics = ComparisonMetrics(**comparison['metrics'])
        
        return ComparisonResponse(
            query=comparison['query'],
            top_k=comparison['top_k'],
            before_reranking=before_results,
            after_reranking=after_results,
            metrics=metrics,
            latency_ms=comparison['latency_ms'],
            method=request.method
        )
    
    except Exception as e:
        import traceback
        print(f"Comparison error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )

# ============================================================================
# RAG ENDPOINTS
# ============================================================================

@app.post("/ask", response_model=RAGResponse)
async def ask(request: RAGRequest):
    """RAG with vector, sparse, or hybrid retrieval"""
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
        
        # Generate answer using LLM
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

@app.post("/ask/self-query", response_model=RAGResponse)
async def ask_self_query(request: RAGRequest):
    """
    🆕 RAG Q&A with Self-Query - Automatically extracts metadata filters
    """
    try:
        print(f"\n{'🔵'*40}")
        print(f"🔵 SELF-QUERY RAG REQUEST")
        print(f"{'🔵'*40}")
        print(f"📥 Input Query: '{request.query}'")
        print(f"📊 Top K: {request.top_k}")
        print(f"🌡️  Temperature: {request.temperature}")
        print(f"🔬 Search Method: {request.method}")
        
        # ⏱️ Start total timing
        total_start = time.time()
        
        # Step 1: Initialize clients
        init_start = time.time()
        retriever = get_self_query_retriever()
        llm_client = get_llm_client()
        init_time = (time.time() - init_start) * 1000
        print(f"⏱️  Client initialization: {init_time:.2f}ms")
        
        # Step 2: Self-query retrieval (includes decomposition + search)
        search_start = time.time()
        search_results, decomposition = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            llm_client=llm_client,  # ✅ Pass LLM client for intelligent decomposition
            verbose=True,
            method=request.method
        )
        search_latency = (time.time() - search_start) * 1000
        print(f"⏱️  Total retrieval (decomposition + search): {search_latency:.2f}ms")
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for your query"
            )
        
        print(f"\n✅ Found {len(search_results)} relevant documents (with self-query filters)")
        
        # Step 3: Format sources
        format_start = time.time()
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
        format_time = (time.time() - format_start) * 1000
        print(f"⏱️  Source formatting: {format_time:.2f}ms")
        
        # Step 4: LLM answer generation
        llm_start = time.time()
        answer = await llm_client.generate_answer(
            query=request.query,
            contexts=search_results,
            temperature=request.temperature
        )
        llm_latency = (time.time() - llm_start) * 1000
        print(f"⏱️  LLM answer generation: {llm_latency:.2f}ms")
        
        total_latency = (time.time() - total_start) * 1000
        print(f"⏱️  LATENCY BREAKDOWN")
        print(f"{'='*80}")
        print(f"  Init:        {init_time:>8.2f}ms")
        print(f"  Retrieval:   {search_latency:>8.2f}ms  (decomposition + search + rerank)")
        print(f"  Format:      {format_time:>8.2f}ms")
        print(f"  LLM:         {llm_latency:>8.2f}ms")
        print(f"  {'─'*76}")
        print(f"  TOTAL:       {total_latency:>8.2f}ms")
        print(f"{'='*80}\n")
        
        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            model_used=llm_client.model,
            search_latency_ms=round(search_latency, 2),
            llm_latency_ms=round(llm_latency, 2),
            total_latency_ms=round(total_latency, 2),
            method=f"self-query-{request.method}"
        )
    
    except Exception as e:
        import traceback
        print(f"\n{'❌'*40}")
        print(f"❌ Self-Query RAG Error: {str(e)}")
        print(f"{'❌'*40}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Self-Query RAG failed: {str(e)}"
        )

# ============================================================================
# ROOT AND UTILITY ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root"""
    return {
        "message": "PDF RAG API with Hybrid Search & Self-Query",
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
            "search_self_query": "/search/self-query",
            "search_rerank": "/search/rerank",
            "search_compare": "/search/compare",
            "ask": "/ask",
            "ask_self_query": "/ask/self-query",
            "pdf": "/pdf/{filename}",
            "api_self_query": "/api/self-query/*"
        },
        "features": [
            "Hybrid Search (Dense + Sparse + RRF)",
            "Self-Query Retrieval (Auto metadata extraction)",
            "Cross-Encoder Reranking",
            "RAG with LLM Generation"
        ]
    }

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