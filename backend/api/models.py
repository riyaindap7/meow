from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(3, description="Number of results to retrieve", ge=1, le=10)

class SearchResult(BaseModel):
    """Individual search result with full VictorText schema"""
    text: str
    source: str
    page: int
    score: float
    # VictorText schema fields
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    global_chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    section_hierarchy: Optional[str] = None
    heading_context: Optional[str] = None
    char_count: Optional[int] = None
    word_count: Optional[int] = None

class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: List[SearchResult]
    count: int
    latency_ms: float

class RAGRequest(BaseModel):
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(3, description="Number of context chunks", ge=1, le=10)
    temperature: Optional[float] = Field(0.0, description="LLM temperature", ge=0.0, le=1.0)

class RAGResponse(BaseModel):
    """RAG response with answer and sources"""
    query: str
    answer: str
    sources: List[SearchResult]
    model_used: str
    search_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool
    collection_exists: bool
    total_vectors: int
    embedding_model: str

# ============================================================================
# 1. UPDATE: backend/api/models.py - Add these new models
# ============================================================================

# Add these new models to your existing models.py file:

class ComparisonResult(BaseModel):
    """Single result in comparison (shows both vector and rerank scores)"""
    rank: int
    text: str
    source: str
    page: int
    vector_score: float
    rerank_score: Optional[float] = None
    score_delta: Optional[float] = None
    # VictorText schema fields
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    global_chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    section_hierarchy: Optional[str] = None
    heading_context: Optional[str] = None
    char_count: Optional[int] = None
    word_count: Optional[int] = None

class ComparisonMetrics(BaseModel):
    """Metrics showing improvement from re-ranking"""
    kept_count: int
    promoted_count: int
    demoted_count: int
    avg_score_before: float
    avg_score_after: float
    improvement_percent: float

class ComparisonResponse(BaseModel):
    """Side-by-side comparison of vector search vs re-ranked results"""
    query: str
    top_k: int
    before_reranking: List[ComparisonResult]
    after_reranking: List[ComparisonResult]
    metrics: ComparisonMetrics
    latency_ms: float

class RerankRequest(BaseModel):
    """Request for search with optional re-ranking"""
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of results to retrieve", ge=1, le=20)
    use_reranker: Optional[bool] = Field(True, description="Enable re-ranking")
    show_comparison: Optional[bool] = Field(False, description="Return before/after comparison")