from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    method: Optional[Literal["vector", "sparse", "hybrid"]] = "hybrid"
    dense_weight: Optional[float] = 0.7
    sparse_weight: Optional[float] = 0.3
    filter_expr: Optional[str] = None
    
class SearchResult(BaseModel):
    text: str
    source: str = ""
    page: int = 0
    score: float = 0.0
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    global_chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    section_hierarchy: Optional[str] = None
    char_count: Optional[int] = None
    word_count: Optional[int] = None
    # New VictorText2 fields
    category: Optional[str] = None
    document_type: Optional[str] = None
    ministry: Optional[str] = None
    published_date: Optional[str] = None
    language: Optional[str] = None
    source_reference: Optional[str] = None
    # Additional fields for compatibility
    source_file: Optional[str] = None
    page_idx: Optional[int] = None
    document_name: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int
    latency_ms: float
    method: Optional[str] = "hybrid"

class RAGRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    method: Optional[Literal["vector", "sparse", "hybrid"]] = "hybrid"
    dense_weight: Optional[float] = 0.7
    sparse_weight: Optional[float] = 0.3
    filter_expr: Optional[str] = None

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]
    conversation_id: Optional[str] = None
    model_used: str = ""
    search_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    method: str = "hybrid"

class HybridSearchRequest(BaseModel):
    """Advanced hybrid search with filtering options"""
    query: str
    top_k: Optional[int] = 10
    dense_weight: Optional[float] = 0.7
    sparse_weight: Optional[float] = 0.3
    # Filter options
    category: Optional[str] = None
    ministry: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    # Additional filters
    source_reference: Optional[str] = None
    semantic_labels: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool = False
    collection_exists: bool = False
    total_vectors: int = 0
    embedding_model: str = ""
    hybrid_enabled: bool = False
    has_dense_field: Optional[bool] = None
    has_sparse_field: Optional[bool] = None

# Conversation models
class Message(BaseModel):
    message_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = {}

class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = []
    metadata: Optional[Dict[str, Any]] = {}

class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Conversation"
    metadata: Optional[Dict[str, Any]] = {}

class TranscriptResponse(BaseModel):
    transcript: str

# Filter models for advanced search
class DocumentFilter(BaseModel):
    category: Optional[str] = None
    ministry: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    published_date_from: Optional[str] = None
    published_date_to: Optional[str] = None

class FilteredSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    method: Optional[Literal["vector", "sparse", "hybrid"]] = "hybrid"
    filters: Optional[DocumentFilter] = None
    dense_weight: Optional[float] = 0.7
    sparse_weight: Optional[float] = 0.3

# Metadata models for discovery endpoints
class AvailableFilters(BaseModel):
    categories: List[str] = []
    ministries: List[str] = []
    document_types: List[str] = []
    languages: List[str] = []
    date_range: Optional[Dict[str, str]] = None

# ============================================================================
# COMPARISON AND RERANKING MODELS (Existing - maintained for compatibility)
# ============================================================================

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
    method: str = "vector"  # Which search method was used for comparison

class RerankRequest(BaseModel):
    """Request for search with optional re-ranking"""
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of results to retrieve", ge=1, le=20)
    use_reranker: Optional[bool] = Field(True, description="Enable re-ranking")
    show_comparison: Optional[bool] = Field(False, description="Return before/after comparison")
    method: Optional[Literal["vector", "sparse", "hybrid"]] = Field(
        "vector",
        description="Search method: vector (dense), sparse, or hybrid (RRF fusion)"
    )

class SearchStats(BaseModel):
    total_documents: int
    total_chunks: int
    available_filters: AvailableFilters
    collection_name: str
    last_updated: Optional[str] = None

class ConversationMetadata(BaseModel):
    """Metadata for conversation list view (without full message history)"""
    conversation_id: str
    user_id: str
    title: str
    created_at: str  # ISO format string for JSON serialization
    updated_at: str  # ISO format string for JSON serialization
    message_count: int = 0
    metadata: Optional[Dict[str, Any]] = {}

class ListConversationsResponse(BaseModel):
    """Response for listing user's conversations"""
    conversations: List[ConversationMetadata]
    count: int

class ConversationResponse(BaseModel):
    """Full conversation with messages"""
    conversation_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Message] = []
    metadata: Optional[Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    query: str
    conversation_id: Optional[str] = None
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    conversation_id: str
    message_id: str
    answer: str
    sources: List[SearchResult] = []
    model_used: str = ""
