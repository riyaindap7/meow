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
from backend.services.self_query_retriever import create_self_query_retriever  # ✅ NEW IMPORT
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
            enable_llm_decomposition=True  # Set to True if you want LLM-based decomposition
        )
        print("✅ Self-Query Retriever initialized")
    return _self_query_retriever

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
    
    try:
        get_llm_client()  # Initialize OpenRouter client
        print("✅ LLM client (OpenRouter) initialized")
    except Exception as e:
        print(f"⚠️  LLM client initialization warning: {e}")
    
    try:
        get_self_query_retriever()  # ✅ Initialize self-query retriever
        print("✅ Self-Query Retriever initialized")
    except Exception as e:
        print(f"⚠️  Self-Query Retriever initialization warning: {e}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="PDF RAG API with Self-Query",
    description="Query PDF documents using self-query retrieval, vector search and LLM",
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
from backend.api.routers import self_query_search  # ✅ NEW IMPORT

# Include all routers
app.include_router(collections.router, prefix="/api/collections", tags=["Collections"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(sync.router, prefix="/api/sync", tags=["Sync"])
app.include_router(scraper.router, prefix="/api/scraper", tags=["Scraper"])
app.include_router(parse_marker_router, prefix="/api/parse", tags=["Parse"])
app.include_router(self_query_search.router, prefix="/api/self-query", tags=["Self-Query"])  # ✅ NEW ROUTER

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

# ============================================================================
# STANDARD SEARCH ENDPOINTS (Existing)
# ============================================================================

@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    """
    Standard search using vector similarity + re-ranking
    Does NOT use self-query (no automatic metadata extraction)
    """
    try:
        milvus_client = get_milvus_client()
        
        start_time = time.time()
        results = milvus_client.search_with_rerank(
            query=request.query,
            top_k=request.top_k
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
            latency_ms=round(search_latency, 2)
        )
    
    except Exception as e:
        import traceback
        print(f"Search error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

# ============================================================================
# SELF-QUERY SEARCH ENDPOINT (NEW - Uses automatic metadata extraction)
# ============================================================================

@app.post("/search/self-query", response_model=SearchResponse)
async def search_self_query(request: QueryRequest):
    """
    🆕 Self-Query Search - Automatically extracts metadata filters from queries
    
    Examples:
    - "What are RUSA guidelines on page 5?"
    - "Tell me about organic food in document Food_Safety_Organic"
    - "Explain certification in section 3.1"
    
    The system automatically extracts filters like document_name, page_idx, etc.
    and applies them to improve search accuracy.
    """
    try:
        print(f"\n{'🔍'*40}")
        print(f"🔍 SELF-QUERY SEARCH REQUEST")
        print(f"{'🔍'*40}")
        
        retriever = get_self_query_retriever()
        llm_client = get_llm_client()  # ✅ ADD THIS LINE

        start_time = time.time()

        # Perform self-query retrieval (prints original vs enhanced query automatically)
        results, decomposition = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            llm_client=llm_client,
            verbose=True
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
            latency_ms=round(search_latency, 2)
        )
    
    except Exception as e:
        import traceback
        print(f"❌ Self-query search error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Self-query search failed: {str(e)}"
        )

# ============================================================================
# RAG ENDPOINTS
# ============================================================================

@app.post("/ask", response_model=RAGResponse)
async def ask(request: RAGRequest):
    """
    Standard RAG Q&A (uses re-ranking but NOT self-query)
    """
    try:
        print(f"\n🔵 RAG Request received")
        print(f"   Query: {request.query}")
        print(f"   Top K: {request.top_k}")
        
        milvus_client = get_milvus_client()
        llm_client = get_llm_client()
        
        total_start = time.time()
        search_start = time.time()
        
        search_results = milvus_client.search_with_rerank(
            query=request.query,
            top_k=request.top_k
        )
        
        search_latency = (time.time() - search_start) * 1000
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for your query"
            )
        
        print(f"📚 Found {len(search_results)} relevant documents")
        
        llm_start = time.time()
        answer = await llm_client.generate_answer(
            query=request.query,
            contexts=search_results,
            temperature=request.temperature
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
        
        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            model_used=llm_client.model,
            search_latency_ms=round(search_latency, 2),
            llm_latency_ms=round(llm_latency, 2),
            total_latency_ms=round(total_latency, 2)
        )
    
    except Exception as e:
        import traceback
        print(f"❌ RAG Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG failed: {str(e)}"
        )

@app.post("/ask/self-query", response_model=RAGResponse)
async def ask_self_query(request: RAGRequest):
    """
    🆕 RAG Q&A with Self-Query - Automatically extracts metadata filters
    
    This endpoint uses self-query retrieval to automatically detect and apply
    metadata filters from natural language queries before generating answers.
    
    Examples:
    - "What are the RUSA guidelines on page 5?"
    - "Tell me about organic certification in Food Safety document"
    - "Explain the rules in section 3.1"
    """
    try:
        print(f"\n{'🔵'*40}")
        print(f"🔵 SELF-QUERY RAG REQUEST")
        print(f"{'🔵'*40}")
        print(f"📥 Input Query: '{request.query}'")
        print(f"📊 Top K: {request.top_k}")
        print(f"🌡️  Temperature: {request.temperature}")
        
        retriever = get_self_query_retriever()
        llm_client = get_llm_client()
        
        total_start = time.time()
        search_start = time.time()
        
        # Use self-query retrieval (this will print original vs enhanced query)
        search_results, decomposition = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            llm_client=llm_client,
            verbose=True
        )
        
        search_latency = (time.time() - search_start) * 1000
        
        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for your query"
            )
        
        # ✅ ENHANCED LOGGING - Show what query is being used for LLM
        print(f"\n{'='*80}")
        print(f"📚 RETRIEVAL RESULTS")
        print(f"{'='*80}")
        print(f"✅ Found {len(search_results)} relevant documents")
        print(f"⏱️  Search latency: {search_latency:.2f}ms")
        
        # Show top 3 results
        for i, result in enumerate(search_results[:3], 1):
            print(f"\n  [{i}] {result.get('document_name')} (Page {result.get('page_idx')})")
            print(f"      Score: {result.get('score', 0):.4f}")
            if 'rerank_score' in result:
                print(f"      Vector: {result.get('vector_score', 0):.4f} | Rerank: {result.get('rerank_score', 0):.4f}")
            print(f"      Preview: {result.get('text', '')[:120]}...")
        
        print(f"\n{'='*80}")
        print(f"🤖 LLM GENERATION")
        print(f"{'='*80}")
        print(f"📝 Query for LLM: '{request.query}'")  # LLM gets original query
        print(f"🎯 Retrieved using: '{decomposition.semantic_query}'")  # But retrieval used enhanced
        print(f"🔧 Applied filters: {len(decomposition.metadata_filters)}")
        
        llm_start = time.time()
        answer = await llm_client.generate_answer(
            query=request.query,  # Pass original query to LLM for context
            contexts=search_results,
            temperature=request.temperature
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
        
        print(f"\n{'='*80}")
        print(f"✅ SELF-QUERY RAG COMPLETE")
        print(f"{'='*80}")
        print(f"⏱️  Search: {search_latency:.2f}ms | LLM: {llm_latency:.2f}ms | Total: {total_latency:.2f}ms")
        print(f"{'='*80}\n")
        
        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            model_used=llm_client.model,
            search_latency_ms=round(search_latency, 2),
            llm_latency_ms=round(llm_latency, 2),
            total_latency_ms=round(total_latency, 2)
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
# COMPARISON ENDPOINTS (Existing)
# ============================================================================

@app.post("/search/rerank", response_model=SearchResponse)
async def search_with_rerank(request: RerankRequest):
    """Search with optional re-ranking"""
    try:
        milvus_client = get_milvus_client()
        start_time = time.time()
        
        if request.use_reranker:
            results = milvus_client.search_with_rerank(
                query=request.query,
                top_k=request.top_k
            )
        else:
            results = milvus_client.search(
                query=request.query,
                top_k=request.top_k
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
            latency_ms=round(search_latency, 2)
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
    """Compare vector search vs re-ranked search side-by-side"""
    try:
        milvus_client = get_milvus_client()
        comparison = milvus_client.get_comparison_data(
            query=request.query,
            top_k=request.top_k
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
            latency_ms=comparison['latency_ms']
        )
    
    except Exception as e:
        import traceback
        print(f"Comparison error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )

# Root endpoint
@app.get("/")
async def root():
    """API root"""
    return {
        "message": "PDF RAG API with Self-Query",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "self_query_search": "/search/self-query",  # ✅ NEW
            "ask": "/ask",
            "ask_self_query": "/ask/self-query",  # ✅ NEW
            "compare": "/search/compare",
            "self_query_api": "/api/self-query/*"  # ✅ NEW dedicated API
        },
        "new_features": [
            "Self-Query Retrieval: Automatic metadata filter extraction",
            "Endpoints: /search/self-query and /ask/self-query",
            "Full API: /api/self-query/self-query, /api/self-query/custom-filter-search"
        ]
    }

# PDF serving endpoint
@app.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    """Serve PDF files from the data directory"""
    try:
        api_dir = Path(__file__).parent
        project_root = api_dir.parent
        pdf_path = project_root / "data" / filename
        
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