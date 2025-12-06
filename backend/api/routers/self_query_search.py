# backend/api/routers/self_query_search.py
"""
Self-Query Search Router

This router provides dedicated endpoints for self-query based retrieval.
Automatically extracts metadata filters from natural language queries.

INTEGRATION STATUS:
✅ Integrated in main.py at /api/self-query/*
✅ Also available as /search/self-query and /ask/self-query in main endpoints

ENDPOINTS:
1. POST /api/self-query/self-query - Full self-query search with decomposition details
2. POST /api/self-query/custom-filter-search - Manual filter specification
3. GET  /api/self-query/metadata-fields - Get available metadata fields
4. POST /api/self-query/test-decomposition - Test query decomposition
5. GET  /api/self-query/health - Health check

MAIN.PY INTEGRATION:
- /search/self-query - Simple self-query search (returns SearchResponse)
- /ask/self-query - Self-query RAG Q&A (returns RAGResponse with LLM answer)

USAGE EXAMPLES:

# Simple Search with Self-Query
curl -X POST http://localhost:8000/search/self-query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are RUSA guidelines on page 5?", "top_k": 5}'

# RAG Q&A with Self-Query
curl -X POST http://localhost:8000/ask/self-query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain organic food regulations in section 3.1", "top_k": 3}'

# Full Self-Query API with Decomposition Details
curl -X POST http://localhost:8000/api/self-query/self-query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find certification process in Food Safety document", "top_k": 5}'

# Custom Filters
curl -X POST http://localhost:8000/api/self-query/custom-filter-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "certification requirements",
    "filters": [
      {"field": "document_name", "operator": "==", "value": "Food_Safety_Organic"},
      {"field": "page_idx", "operator": ">=", "value": 5}
    ],
    "top_k": 5
  }'
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from backend.services.self_query_retriever import (
    SelfQueryRetriever,
    SelfQueryConfig,
    MetadataFilter,
    QueryDecomposition,
    create_self_query_retriever
)

router = APIRouter()

# Global retriever instance (initialized on first request)
_retriever: Optional[SelfQueryRetriever] = None


def get_retriever() -> SelfQueryRetriever:
    """Get or create retriever singleton"""
    global _retriever
    if _retriever is None:
        _retriever = create_self_query_retriever(
            collection_name="VictorText",
            top_k=5,
            rerank=True,
            enable_llm_decomposition=False
        )
    return _retriever


# ========================== REQUEST/RESPONSE MODELS ==========================

class SelfQueryRequest(BaseModel):
    """Request model for self-query search"""
    query: str = Field(..., description="Natural language query")
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=50)
    enable_llm_decomposition: Optional[bool] = Field(False, description="Use LLM for query decomposition")
    verbose: Optional[bool] = Field(True, description="Include decomposition details in response")


class CustomFilterRequest(BaseModel):
    """Request model for search with custom filters"""
    query: str = Field(..., description="Semantic query")
    filters: List[MetadataFilter] = Field(..., description="Metadata filters to apply")
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=50)


class SearchResult(BaseModel):
    """Single search result"""
    score: float
    document_name: str
    document_id: str
    chunk_id: str
    page_idx: int
    chunk_index: int
    section_hierarchy: Optional[str]
    heading_context: Optional[str]
    text: str
    char_count: int
    word_count: int
    rerank_score: Optional[float] = None
    vector_score: Optional[float] = None


class SelfQueryResponse(BaseModel):
    """Response model for self-query search"""
    query: str
    semantic_query: str
    metadata_filters: List[MetadataFilter]
    filter_expression: Optional[str]
    results: List[SearchResult]
    total_results: int
    decomposition_method: str  # "rule_based" or "llm_based"


# ========================== API ENDPOINTS ==========================

@router.post("/self-query", response_model=SelfQueryResponse)
async def self_query_search(request: SelfQueryRequest):
    """
    Self-querying search endpoint.
    Automatically extracts metadata filters from natural language queries.
    
    Example queries:
    - "What are organic food regulations on page 5?"
    - "Tell me about RUSA guidelines in document RUSA_final090913"
    - "Explain certification process in section 3.1"
    """
    try:
        retriever = get_retriever()
        
        # Perform self-query retrieval
        results, decomposition = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            llm_client=None,  # You can integrate your LLM client here
            verbose=request.verbose
        )
        
        # Build filter expression for response
        filter_expr = retriever.build_milvus_filter_expression(decomposition.metadata_filters)
        
        return SelfQueryResponse(
            query=decomposition.original_query,
            semantic_query=decomposition.semantic_query,
            metadata_filters=decomposition.metadata_filters,
            filter_expression=filter_expr,
            results=[SearchResult(**r) for r in results],
            total_results=len(results),
            decomposition_method="llm_based" if request.enable_llm_decomposition else "rule_based"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/custom-filter-search", response_model=List[SearchResult])
async def custom_filter_search(request: CustomFilterRequest):
    """
    Search with manually specified metadata filters.
    Bypass automatic query decomposition.
    
    Example:
    ```json
    {
        "query": "What are the regulations?",
        "filters": [
            {"field": "document_name", "operator": "==", "value": "Food Safety Regulations"},
            {"field": "page_idx", "operator": ">", "value": 10}
        ],
        "top_k": 5
    }
    ```
    """
    try:
        retriever = get_retriever()
        
        results = retriever.retrieve_with_custom_filters(
            query=request.query,
            custom_filters=request.filters,
            top_k=request.top_k,
            verbose=True
        )
        
        return [SearchResult(**r) for r in results]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/metadata-fields")
async def get_metadata_fields():
    """
    Get available metadata fields for filtering
    """
    retriever = get_retriever()
    return {
        "metadata_fields": retriever.config.metadata_fields,
        "collection_name": retriever.config.collection_name
    }


@router.post("/test-decomposition")
async def test_query_decomposition(query: str = Query(..., description="Query to decompose")):
    """
    Test query decomposition without performing search.
    Useful for debugging and understanding how queries are parsed.
    """
    try:
        retriever = get_retriever()
        decomposition = retriever.decomposer.decompose(query)
        filter_expr = retriever.build_milvus_filter_expression(decomposition.metadata_filters)
        
        return {
            "original_query": decomposition.original_query,
            "semantic_query": decomposition.semantic_query,
            "metadata_filters": [f.dict() for f in decomposition.metadata_filters],
            "milvus_filter_expression": filter_expr,
            "decomposition_method": "rule_based"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decomposition failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check for self-query retriever"""
    try:
        retriever = get_retriever()
        collection = retriever.get_collection()
        
        return {
            "status": "healthy",
            "milvus_connected": True,
            "collection": retriever.config.collection_name,
            "total_entities": collection.num_entities,
            "embedding_model": retriever.embedding_model_name,
            "reranking_enabled": retriever.config.rerank
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")