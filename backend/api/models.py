from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    method: Optional[Literal["vector", "sparse", "hybrid"]] = "hybrid"
    filter_expr: Optional[str] = None
    
    # NEW: Individual filter fields
    category: Optional[str] = None
    language: Optional[str] = None
    document_type: Optional[str] = None
    document_id: Optional[str] = None  # Filter by document ID (not document_name)
    date_from: Optional[str] = None
    date_to: Optional[str] = None

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
    heading_context: Optional[str] = None
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
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.1
    dense_weight: Optional[float] = 0.6
    sparse_weight: Optional[float] = 0.4
    method: Optional[Literal["vector", "sparse", "hybrid"]] = "hybrid"
    
    # ✅ ALL Filter parameters must be here
    category: Optional[str] = None
    language: Optional[str] = None
    document_type: Optional[str] = None
    document_id: Optional[str] = None  # Filter by document ID (not document_name)
    ministry: Optional[str] = None  # ✅ VERIFY THIS IS PRESENT
    date_from: Optional[str] = None
    date_to: Optional[str] = None

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
    query: str
    top_k: Optional[int] = 5
    category: Optional[str] = None
    ministry: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    document_id: Optional[str] = None  # Filter by document ID

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

class SearchStats(BaseModel):
    total_documents: int
    total_chunks: int
    available_filters: AvailableFilters
    collection_name: str
    last_updated: Optional[str] = None