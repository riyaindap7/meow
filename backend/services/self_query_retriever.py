# backend/services/self_query_retriever.py

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()


# ========================== DATA MODELS ==========================

class MetadataFilter(BaseModel):
    """Represents a metadata filter condition"""
    field: str
    operator: str  # "==", "!=", ">", "<", ">=", "<=", "in", "not in", "like"
    value: Any


class QueryDecomposition(BaseModel):
    """Decomposed query with semantic part and metadata filters"""
    semantic_query: str = Field(..., description="The semantic/content-based part of the query")
    metadata_filters: List[MetadataFilter] = Field(default_factory=list, description="Extracted metadata filters")
    original_query: str = Field(..., description="Original user query")


class SelfQueryConfig(BaseModel):
    """Configuration for self-query retriever"""
    collection_name: str = "VictorText"
    top_k: int = 5
    rerank: bool = True
    rerank_top_n: Optional[int] = None
    enable_llm_decomposition: bool = False  # Set True if you want to use LLM for query decomposition
    
    # Metadata field definitions for your schema
    metadata_fields: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "document_name": {
            "type": "string",
            "description": "Name of the source document/file"
        },
        "page_idx": {
            "type": "integer",
            "description": "Page number in the document (0-indexed)"
        },
        "section_hierarchy": {
            "type": "string",
            "description": "Hierarchical section path (e.g., 'Chapter 1 > Section 1.1')"
        },
        "heading_context": {
            "type": "string",
            "description": "Nearby heading or section title"
        },
        "char_count": {
            "type": "integer",
            "description": "Number of characters in the chunk"
        },
        "word_count": {
            "type": "integer",
            "description": "Number of words in the chunk"
        }
    })


# ========================== QUERY DECOMPOSER ==========================

class QueryDecomposer:
    """
    Decomposes user queries into semantic query + metadata filters
    Can use either rule-based or LLM-based decomposition
    """
    
    def __init__(self, config: SelfQueryConfig):
        self.config = config
        self.metadata_fields = config.metadata_fields
        
        # Patterns for rule-based extraction
        self.patterns = {
            "document_name": [
                r"(?:in|from|within)\s+(?:document|file|pdf)\s+['\"]?([^'\"]+)['\"]?",
                r"(?:document|file|pdf)\s+(?:named|called)\s+['\"]?([^'\"]+)['\"]?",
            ],
            "page_idx": [
                r"(?:on|in|from)\s+page\s+(\d+)",
                r"page\s+(?:number\s+)?(\d+)",
            ],
            "section_hierarchy": [
                r"(?:in|from|within)\s+(?:section|chapter)\s+['\"]?([^'\"]+)['\"]?",
            ],
            "heading_context": [
                r"(?:under|in)\s+(?:heading|title)\s+['\"]?([^'\"]+)['\"]?",
            ]
        }
    
    def decompose_query_rule_based(self, query: str) -> QueryDecomposition:
        """
        Rule-based query decomposition using regex patterns
        """
        metadata_filters = []
        cleaned_query = query
        
        # Extract metadata filters using patterns
        for field, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    value = match.group(1).strip()
                    
                    # Convert value to appropriate type
                    if self.metadata_fields.get(field, {}).get("type") == "integer":
                        try:
                            value = int(value)
                        except ValueError:
                            continue
                    
                    # Create filter
                    metadata_filters.append(MetadataFilter(
                        field=field,
                        operator="==" if field in ["document_name", "section_hierarchy", "heading_context"] else "==",
                        value=value
                    ))
                    
                    # Remove matched pattern from query
                    cleaned_query = cleaned_query.replace(match.group(0), "").strip()
        
        # Clean up the semantic query
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        
        return QueryDecomposition(
            semantic_query=cleaned_query if cleaned_query else query,
            metadata_filters=metadata_filters,
            original_query=query
        )
    
    def decompose_query_llm_based(self, query: str, llm_client=None) -> QueryDecomposition:
        """
        LLM-based query decomposition (more accurate but requires LLM)
        Uses your existing LLM client to decompose the query
        """
        if not llm_client:
            # Fallback to rule-based if no LLM client provided
            return self.decompose_query_rule_based(query)
        
        # Create prompt for LLM
        metadata_desc = "\n".join([
            f"- {field}: {info['description']} (type: {info['type']})"
            for field, info in self.metadata_fields.items()
        ])
        
        prompt = f"""You are a query decomposition assistant for a document retrieval system.
        
Available metadata fields:
{metadata_desc}

User Query: "{query}"

Decompose this query into:
1. A semantic query (the actual content/meaning to search for)
2. Metadata filters (constraints on document properties)

Return a JSON object with this structure:
{{
    "semantic_query": "the semantic part of the query",
    "metadata_filters": [
        {{
            "field": "field_name",
            "operator": "==",  // or >, <, >=, <=, in, like
            "value": "value"
        }}
    ]
}}

If no metadata filters are needed, return an empty array for metadata_filters.
Only extract filters for fields that are explicitly mentioned or clearly implied in the query.
"""
        
        try:
            # Call LLM to decompose query
            print(f"🔵 Calling LLM for query decomposition...")
            response = llm_client.generate(prompt, temperature=0.0, max_tokens=500)
            
            # ✅ CLEAN MARKDOWN CODE BLOCKS
            # Remove ```json and ``` wrappers if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]  # Remove ```
            
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            cleaned_response = cleaned_response.strip()
            
            print(f"🟢 Cleaned response: {cleaned_response[:200]}...")
            
            # Parse LLM response
            result = json.loads(cleaned_response)
            
            filters = [
                MetadataFilter(**f) for f in result.get("metadata_filters", [])
            ]
            
            print(f"✅ Query decomposition successful!")
            print(f"   Original: '{query}'")
            print(f"   Enhanced: '{result.get('semantic_query', query)}'")
            print(f"   Filters: {len(filters)}")
            
            return QueryDecomposition(
                semantic_query=result.get("semantic_query", query),
                metadata_filters=filters,
                original_query=query
            )
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print(f"   Raw response was: '{response}'")
            return self.decompose_query_rule_based(query)
        except Exception as e:
            print(f"⚠️ LLM decomposition failed: {e}. Falling back to rule-based.")
            import traceback
            traceback.print_exc()
            return self.decompose_query_rule_based(query)
    
    def decompose(self, query: str, llm_client=None) -> QueryDecomposition:
        """Main decomposition method"""
        if self.config.enable_llm_decomposition and llm_client:
            return self.decompose_query_llm_based(query, llm_client)
        else:
            return self.decompose_query_rule_based(query)


# ========================== SELF-QUERY RETRIEVER ==========================

class SelfQueryRetriever:
    """
    Self-querying retriever that automatically extracts metadata filters
    from natural language queries and applies them to vector search
    """
    
    def __init__(self, config: Optional[SelfQueryConfig] = None):
        self.config = config or SelfQueryConfig()
        
        # Initialize components
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize query decomposer
        self.decomposer = QueryDecomposer(self.config)
        
        # Connect to Milvus
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.connect()
    
    def connect(self):
        """Connect to Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def get_collection(self) -> Collection:
        """Get collection instance"""
        from pymilvus import utility
        if not utility.has_collection(self.config.collection_name):
            raise ValueError(f"Collection '{self.config.collection_name}' does not exist")
        
        collection = Collection(self.config.collection_name)
        collection.load()
        return collection
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=False
        )
        return embedding[0].tolist()
    
    def build_milvus_filter_expression(self, filters: List[MetadataFilter]) -> Optional[str]:
        """
        Build Milvus filter expression from metadata filters
        Milvus filter syntax: https://milvus.io/docs/boolean.md
        """
        if not filters:
            return None
        
        expressions = []
        
        for f in filters:
            field = f.field
            op = f.operator
            value = f.value
            
            # Format value based on type
            if isinstance(value, str):
                value_str = f'"{value}"'
            elif isinstance(value, (int, float)):
                value_str = str(value)
            elif isinstance(value, list):
                # For 'in' operator
                value_str = "[" + ", ".join([f'"{v}"' if isinstance(v, str) else str(v) for v in value]) + "]"
            else:
                value_str = str(value)
            
            # Build expression based on operator
            if op == "==":
                expr = f'{field} == {value_str}'
            elif op == "!=":
                expr = f'{field} != {value_str}'
            elif op == ">":
                expr = f'{field} > {value_str}'
            elif op == "<":
                expr = f'{field} < {value_str}'
            elif op == ">=":
                expr = f'{field} >= {value_str}'
            elif op == "<=":
                expr = f'{field} <= {value_str}'
            elif op == "in":
                expr = f'{field} in {value_str}'
            elif op == "not in":
                expr = f'{field} not in {value_str}'
            elif op == "like":
                # Milvus uses 'like' for pattern matching
                expr = f'{field} like "{value}"'
            else:
                print(f"⚠️ Unsupported operator: {op}")
                continue
            
            expressions.append(expr)
        
        # Combine with AND
        if expressions:
            return " && ".join(expressions)
        return None
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        llm_client=None,
        verbose: bool = True
    ) -> Tuple[List[Dict], QueryDecomposition]:
        """
        Main retrieval method with self-querying
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            llm_client: Optional LLM client for query decomposition
            verbose: Print decomposition details
        
        Returns:
            Tuple of (results, query_decomposition)
        """
        top_k = top_k or self.config.top_k
        
        # Step 1: Decompose query
        decomposition = self.decomposer.decompose(query, llm_client)
        
        # ✅ ENHANCED LOGGING - Always print original and enhanced query
        print(f"\n{'='*80}")
        print(f"🔍 QUERY DECOMPOSITION")
        print(f"{'='*80}")
        print(f"📝 Original Query:  '{decomposition.original_query}'")
        print(f"🎯 Enhanced Query:  '{decomposition.semantic_query}'")
        print(f"{'='*80}")
        
        if verbose:
            print(f"\n📋 Extracted Metadata Filters: {len(decomposition.metadata_filters)}")
            if decomposition.metadata_filters:
                for i, f in enumerate(decomposition.metadata_filters, 1):
                    print(f"  {i}. {f.field} {f.operator} {f.value}")
            else:
                print(f"  No metadata filters extracted")
            print(f"{'='*80}\n")
        
        # Step 2: Build Milvus filter expression
        filter_expr = self.build_milvus_filter_expression(decomposition.metadata_filters)
        
        if verbose and filter_expr:
            print(f"🔧 Milvus Filter Expression: {filter_expr}\n")
        
        # Step 3: Perform vector search with filters
        results = self._search_milvus(
            semantic_query=decomposition.semantic_query,
            filter_expr=filter_expr,
            top_k=top_k
        )
        
        if verbose:
            print(f"✅ Retrieved {len(results)} results\n")
        
        return results, decomposition
    
    def _search_milvus(
        self,
        semantic_query: str,
        filter_expr: Optional[str],
        top_k: int
    ) -> List[Dict]:
        """
        Perform vector search in Milvus with optional filters
        """
        collection = self.get_collection()
        
        # Generate query embedding
        query_embedding = self.embed_query(semantic_query)
        
        # Search parameters
        search_params = {
            "metric_type": "IP",
            "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
        }
        
        # Output fields
        output_fields = [
            "document_name",
            "document_id",
            "chunk_id",
            "global_chunk_id",
            "page_idx",
            "chunk_index",
            "section_hierarchy",
            "heading_context",
            "text",
            "char_count",
            "word_count"
        ]
        
        # Execute search
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k * 2 if self.config.rerank else top_k,  # Get more for reranking
            expr=filter_expr,
            output_fields=output_fields
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "score": float(hit.score),
                    "document_name": hit.entity.get("document_name"),
                    "document_id": hit.entity.get("document_id"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "global_chunk_id": hit.entity.get("global_chunk_id"),
                    "page_idx": hit.entity.get("page_idx"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "section_hierarchy": hit.entity.get("section_hierarchy"),
                    "heading_context": hit.entity.get("heading_context"),
                    "text": hit.entity.get("text"),
                    "char_count": hit.entity.get("char_count"),
                    "word_count": hit.entity.get("word_count")
                })
        
        # Optional: Apply reranking
        if self.config.rerank and formatted_results:
            formatted_results = self._rerank_results(semantic_query, formatted_results, top_k)
        
        return formatted_results
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """
        Rerank results using cross-encoder (if available)
        """
        try:
            from sentence_transformers import CrossEncoder
            
            reranker_model_name = os.getenv("RERANKER_MODEL")
            if not reranker_model_name:
                return results[:top_k]
            
            reranker = CrossEncoder(reranker_model_name)
            
            # Prepare query-document pairs
            pairs = [[query, doc['text']] for doc in results]
            
            # Get reranking scores
            rerank_scores = reranker.predict(pairs)
            
            # Add rerank scores to results
            for doc, score in zip(results, rerank_scores):
                doc['rerank_score'] = float(score)
                doc['vector_score'] = doc['score']
                doc['score'] = float(score)  # Use rerank score as primary
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            return reranked[:top_k]
        
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}. Using vector scores.")
            return results[:top_k]
    
    def retrieve_with_custom_filters(
        self,
        query: str,
        custom_filters: List[MetadataFilter],
        top_k: Optional[int] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Retrieve with manually specified metadata filters (bypass query decomposition)
        """
        top_k = top_k or self.config.top_k
        
        # Build filter expression
        filter_expr = self.build_milvus_filter_expression(custom_filters)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🔍 CUSTOM FILTER RETRIEVAL")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Filters: {filter_expr}")
            print(f"{'='*80}\n")
        
        # Perform search
        results = self._search_milvus(
            semantic_query=query,
            filter_expr=filter_expr,
            top_k=top_k
        )
        
        return results


# ========================== CONVENIENCE FUNCTIONS ==========================

def create_self_query_retriever(
    collection_name: str = "VictorText",
    top_k: int = 5,
    rerank: bool = True,
    enable_llm_decomposition: bool = False
) -> SelfQueryRetriever:
    """
    Factory function to create a self-query retriever
    """
    config = SelfQueryConfig(
        collection_name=collection_name,
        top_k=top_k,
        rerank=rerank,
        enable_llm_decomposition=enable_llm_decomposition
    )
    return SelfQueryRetriever(config)


# ========================== EXAMPLE USAGE ==========================

if __name__ == "__main__":
    # Create retriever
    retriever = create_self_query_retriever(
        collection_name="VictorText",
        top_k=5,
        rerank=True,
        enable_llm_decomposition=False  # Set True if you have LLM client
    )
    
    # Example queries with implicit metadata filters
    test_queries = [
        "What are the organic food regulations?",
        "Tell me about RUSA guidelines on page 5",
        "Find information about alcoholic beverages in document Food Safety and Standards",
        "What does section 3.1 say about labelling requirements?",
        "Explain the certification process in the organic food document",
    ]
    
    print("\n" + "="*100)
    print("🚀 SELF-QUERY RETRIEVER DEMO")
    print("="*100)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*100}")
        print(f"QUERY {i}/{len(test_queries)}")
        print(f"{'#'*100}\n")
        
        # Retrieve with self-querying
        results, decomposition = retriever.retrieve(query, verbose=True)
        
        # Display results
        print(f"📄 TOP RESULTS:\n")
        for rank, result in enumerate(results[:3], 1):
            print(f"[{rank}] Score: {result['score']:.4f}")
            print(f"    Document: {result['document_name']}")
            print(f"    Page: {result['page_idx']}")
            print(f"    Section: {result.get('section_hierarchy', 'N/A')}")
            print(f"    Text: {result['text'][:200]}...")
            print()
    
    print("\n" + "="*100)
    print("✅ DEMO COMPLETE")
    print("="*100)