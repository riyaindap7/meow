from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(3, description="Number of results to retrieve", ge=1, le=10)
    method: Optional[Literal["vector", "sparse", "hybrid"]] = Field(
        "hybrid", 
        description="Search method: vector (dense), sparse, or hybrid (RRF fusion)"
    )

class SearchResult(BaseModel):
    """Individual search result with full VictorText schema"""
    text: str
    source: str
    page: int
    score: float
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
    method: str = "hybrid"

class RAGRequest(BaseModel):
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(3, description="Number of context chunks", ge=1, le=10)
    temperature: Optional[float] = Field(0.0, description="LLM temperature", ge=0.0, le=1.0)
    method: Optional[Literal["vector", "sparse", "hybrid"]] = Field(
        "hybrid", 
        description="Search method: vector (dense), sparse, or hybrid (RRF fusion)"
    )

class RAGResponse(BaseModel):
    """RAG response with answer and sources"""
    query: str
    answer: str
    sources: List[SearchResult]
    model_used: str
    search_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    method: str = "hybrid"

class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool
    collection_exists: bool
    total_vectors: int
    embedding_model: str
    hybrid_enabled: bool = False