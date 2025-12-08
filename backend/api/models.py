from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Literal
from datetime import datetime

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(3, description="Number of results to retrieve", ge=1, le=10)
    method: Optional[Literal["vector", "sparse", "hybrid"]] = Field(
        "vector",  # Default to vector for backward compatibility
        description="Search method: vector (dense), sparse, or hybrid (RRF fusion)"
    )

class SearchResult(BaseModel):
    """Individual search result with Vtext schema"""
    text: str
    source_file: str  # Changed from 'source'
    page_idx: int  # Changed from 'page'
    score: float
    # Vtext schema fields
    global_chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    section_hierarchy: Optional[str] = None
    char_count: Optional[int] = None
    word_count: Optional[int] = None

class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: List[SearchResult]
    count: int
    latency_ms: float
    method: str = "vector"  # Which search method was used

class RAGRequest(BaseModel):
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(3, description="Number of context chunks", ge=1, le=10)
    temperature: Optional[float] = Field(0.0, description="LLM temperature", ge=0.0, le=1.0)
    method: Optional[Literal["vector", "sparse", "hybrid"]] = Field(
        "vector",  # Default to vector for backward compatibility
        description="Search method: vector (dense), sparse, or hybrid (RRF fusion)"
    )
    conversation_id: Optional[str] = Field(None, description="Conversation ID for history tracking")  # Add this line

class RAGResponse(BaseModel):
    """RAG response with answer and sources"""
    query: str
    answer: str
    sources: List[SearchResult]
    model_used: str
    search_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    method: str = "vector"  # Which search method was used

class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool
    collection_exists: bool
    total_vectors: int
    embedding_model: str
    hybrid_enabled: bool = False
    has_dense_field: Optional[bool] = None
    has_sparse_field: Optional[bool] = None
    reranker_enabled: Optional[bool] = None
    reranker_model: Optional[str] = None

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

# ==================== CONVERSATION & CHAT MODELS ====================

class RetrievedDoc(BaseModel):
    """Retrieved document reference in context"""
    doc_id: str
    source: str
    page: int
    score: float
    snippet: Optional[str] = None
    chunk_id: Optional[str] = None

class ConversationContext(BaseModel):
    """Last retrieval snapshot used for generation"""
    last_search_at: Optional[datetime] = None
    query: Optional[str] = None
    retrieved_docs: List[RetrievedDoc] = []

class ConversationSummary(BaseModel):
    """Condensed summary of older turns"""
    summary_text: str
    updated_at: Optional[datetime] = None
    tool_version: str = "v1.0"

class ChatMessage(BaseModel):
    """Individual chat message in conversation"""
    message_id: Optional[str] = None
    role: str = Field(..., description="user, assistant, or system")
    content: str
    content_type: Optional[str] = "text"
    attachments: Optional[List[Dict]] = []
    created_at: Optional[datetime] = None
    tokens_estimate: Optional[int] = None
    meta: Optional[Dict] = {}

class ConversationMetadata(BaseModel):
    """Conversation metadata response"""
    conversation_id: str
    user_id: Optional[str] = None
    title: str
    created_at: datetime
    updated_at: datetime
    archived: bool = False
    message_count: Optional[int] = 0

class CreateConversationRequest(BaseModel):
    """Request to create new conversation"""
    title: Optional[str] = "New Conversation"
    metadata: Optional[Dict] = None
    settings: Optional[Dict] = None

class ConversationResponse(BaseModel):
    """Complete conversation response"""
    conversation_id: str
    user_id: Optional[str] = None
    title: str
    created_at: datetime
    updated_at: datetime
    archived: bool = False
    metadata: Optional[Dict] = None
    messages: List[ChatMessage] = []
    context: Optional[ConversationContext] = None
    summary: Optional[ConversationSummary] = None
    settings: Optional[Dict] = None

class ChatRequest(BaseModel):
    """Chat request with conversation context"""
    conversation_id: str
    query: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.1
    include_history: Optional[bool] = True

class ChatResponse(BaseModel):
    """Chat response with message and context"""
    conversation_id: str
    message_id: str
    role: str = "assistant"
    content: str
    sources: Optional[List[SearchResult]] = []
    model_used: str
    search_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float

class ListConversationsResponse(BaseModel):
    """List of user's conversations"""
    conversations: List[ConversationMetadata]
    count: int
    total: Optional[int] = None
