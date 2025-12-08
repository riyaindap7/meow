from pymilvus import connections, Collection, utility, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer, CrossEncoder
from FlagEmbedding import BGEM3FlagModel
import os
from typing import List, Dict, Literal, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class MilvusClient:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("COLLECTION_NAME", "VictorText2")  # ✅ Changed
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        
        # Dense embeddings - SentenceTransformer (for backward compatibility)
        print(f"Loading dense embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.dense_model = self.embedding_model  # Alias
        
        # Sparse embeddings - FlagEmbedding (only loaded if hybrid search is used)
        self._sparse_model = None
        
        # ✅ Pre-load sparse model if hybrid is enabled
        if os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true":
            print(f"🚀 Pre-loading sparse model for hybrid search...")
            try:
                _ = self.sparse_model  # Trigger lazy load
            except Exception as e:
                print(f"⚠️  Could not pre-load sparse model: {e}")
    
        # Initialize re-ranker (lazy loading)
        self.reranker_model_name = os.getenv("RERANKER_MODEL")
        self._reranker = None
        self.enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        
        self.connect()
    
    @property
    def sparse_model(self):
        """Lazy load sparse model only when needed"""
        if self._sparse_model is None:
            try:
                print(f"⏳ Loading sparse embedding model: {self.embedding_model_name}")
                print(f"   This may take 30-60 seconds on first run (downloading model)...")
                print(f"   Subsequent queries will be instant (model cached in memory)")
                
                self._sparse_model = BGEM3FlagModel(
                    self.embedding_model_name,
                    use_fp16=False,
                    device='cpu'  # Force CPU to avoid GPU issues
                )
                
                print(f"✅ Sparse model loaded successfully")
                
            except Exception as e:
                print(f"❌ Error loading sparse model: {e}")
                print(f"⚠️  Falling back to dense-only search")
                import traceback
                traceback.print_exc()
                return None  # Return None instead of storing None
    
        return self._sparse_model
    
    @property
    def reranker(self):
        """Lazy load re-ranker only when needed"""
        if self._reranker is None and self.enable_reranking:
            print(f"Loading re-ranker model: {self.reranker_model_name}")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker
        
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
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist")
        
        collection = Collection(self.collection_name)
        collection.load()
        return collection
    
    def embed_query(self, query: str) -> List[float]:
        """Generate dense embedding (backward compatible)"""
        return self.embed_query_dense(query)
    
    def embed_query_dense(self, query: str) -> List[float]:
        """Generate dense embedding using SentenceTransformer"""
        embedding = self.dense_model.encode(
            [query],
            normalize_embeddings=False  # Keep as False for VictorText compatibility
        )
        return embedding[0].tolist()
    
    def embed_query_sparse(self, query: str) -> Dict[int, float]:
        """Generate sparse embedding using FlagEmbedding"""
        output = self.sparse_model.encode(
            [query],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        # ✅ Extract and return sparse weights
        sparse_weights = output['lexical_weights'][0]
        return {int(k): float(v) for k, v in sparse_weights.items()}
   
    def search(self, query: str, top_k: int = 5, filter_expr: str = None) -> List[Dict]:
        """
        Search for similar vectors in VictorText2 collection
        """
        collection = self.get_collection()
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search parameters
        search_params = {
            "metric_type": "IP",
            "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
        }
        
        # Use consistent output fields
        output_fields = self._get_output_fields()  # ✅ Use the method instead of inline list
        
        # Execute search
        results = collection.search(
            data=[query_embedding],
            anns_field="dense_embedding",  # ✅ VictorText2 uses dense_embedding
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        
        return self._format_results(results)
    
    def _get_output_fields(self) -> List[str]:
        """Common output fields for VictorText2 searches"""
        return [
            "document_name", "document_id", "chunk_id", "global_chunk_id",
            "page_idx", "chunk_index", "section_hierarchy", "heading_context",
            "text", "char_count", "word_count",
            # ✅ Add metadata fields from VictorText2
            "Category", "document_type", "ministry", "published_date",
            "source_reference", "version", "language", "semantic_labels"
        ]
    
    def _format_results(self, results, use_distance: bool = False) -> List[Dict]:
        """Format search results"""
        formatted_results = []
        for hits in results:
            for hit in hits:
                score = float(hit.distance) if use_distance else float(hit.score)
                formatted_results.append({
                    "score": score,
                    "global_chunk_id": hit.entity.get("global_chunk_id"),
                    "document_id": hit.entity.get("document_id"),
                    "document_name": hit.entity.get("document_name"),  # ✅ VictorText2 field
                    "chunk_id": hit.entity.get("chunk_id"),
                    "page_idx": hit.entity.get("page_idx"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "section_hierarchy": hit.entity.get("section_hierarchy"),
                    "heading_context": hit.entity.get("heading_context"),  # ✅ VictorText2 field
                    "text": hit.entity.get("text"),
                    "char_count": hit.entity.get("char_count"),
                    "word_count": hit.entity.get("word_count"),
                    # ✅ Add metadata fields
                    "category": hit.entity.get("Category"),
                    "document_type": hit.entity.get("document_type"),
                    "ministry": hit.entity.get("ministry"),
                    "published_date": hit.entity.get("published_date"),
                    "source_reference": hit.entity.get("source_reference"),
                    "version": hit.entity.get("version"),
                    "language": hit.entity.get("language"),
                    "semantic_labels": hit.entity.get("semantic_labels")
                })
        return formatted_results
    
    def search(self, query: str, top_k: int = 5, filter_expr: str = None, 
               method: Literal["vector", "sparse", "hybrid"] = "vector") -> List[Dict]:
        """
        Unified search method (backward compatible with vector-only search)
        
        Args:
            query: Search query text
            top_k: Number of results
            filter_expr: Optional Milvus filter expression
            method: 'vector' (default), 'sparse', or 'hybrid'
        """
        if method == "sparse":
            return self._sparse_search(query, top_k, filter_expr)
        elif method == "hybrid":
            return self._hybrid_search(query, top_k, filter_expr)
        else:
            return self._vector_search(query, top_k, filter_expr)
    
    def _vector_search(self, query: str, top_k: int = 5, filter_expr: str = None) -> List[Dict]:
        """Dense vector search using HNSW"""
        collection = self.get_collection()
        dense_embedding = self.embed_query_dense(query)
        
        search_params = {
            "metric_type": "IP",
            "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
        }
        
        # VictorText2 uses "dense_embedding"
        anns_field = "dense_embedding" if self._has_field("dense_embedding") else "embedding"
        
        results = collection.search(
            data=[dense_embedding],
            anns_field=anns_field,
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=self._get_output_fields()
        )
        
        return self._format_results(results)
    
    def _sparse_search(self, query: str, top_k: int = 5, filter_expr: str = None) -> List[Dict]:
        """Sparse vector search using inverted index"""
        collection = self.get_collection()
        sparse_embedding = self.embed_query_sparse(query)
        
        search_params = {
            "metric_type": "IP",
            "params": {"drop_ratio_search": 0.2}
        }
        
        results = collection.search(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=self._get_output_fields()
        )
        
        return self._format_results(results)
    
    def _hybrid_search(self, query: str, top_k: int = 5, filter_expr: str = None) -> List[Dict]:
        """Hybrid search using RRF fusion of dense + sparse"""
        collection = self.get_collection()
        
        # Get both embeddings
        dense_embedding = self.embed_query_dense(query)
        sparse_embedding = self.embed_query_sparse(query)
        
        # Determine dense field name
        dense_field = "embedding" if self._has_field("embedding") else "dense_embedding"
        
        # Dense search request
        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field=dense_field,
            param={
                "metric_type": "IP",
                "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
            },
            limit=top_k * 2,
            expr=filter_expr
        )
        
        # Sparse search request
        sparse_req = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={
                "metric_type": "IP",
                "params": {"drop_ratio_search": 0.2}
            },
            limit=top_k * 2,
            expr=filter_expr
        )
        
        # RRF Ranker
        ranker = RRFRanker(k=60)
        
        # Execute hybrid search
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=top_k,
            output_fields=self._get_output_fields()
        )
        
        return self._format_results(results, use_distance=True)
    
    def _has_field(self, field_name: str) -> bool:
        """Check if collection has a specific field"""
        try:
            collection = self.get_collection()
            schema = collection.schema
            return any(f.name == field_name for f in schema.fields)
        except:
            return False
    
    def search_with_rerank(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_expr: str = None,
        rerank_top_n: Optional[int] = None,
        alpha: float = 0.5,
        method: Literal["vector", "sparse", "hybrid"] = "vector"
    ) -> List[Dict]:
        """
        Two-stage retrieval with HYBRID SCORING: Search + Re-Ranking
        
        Args:
            query: Search query text
            top_k: Final number of results to return after re-ranking
            filter_expr: Optional Milvus filter expression
            rerank_top_n: Number of candidates to retrieve before re-ranking 
            alpha: Weight for re-ranker score (0.0 = all vector, 1.0 = all rerank)
            method: Search method - 'vector', 'sparse', or 'hybrid'
        
        Returns:
            Re-ranked list of chunks with hybrid scores
        """
        # Stage 1: Retrieve more candidates for re-ranking
        if rerank_top_n is None:
            rerank_top_n = max(top_k * 4, 20)
        else:
            rerank_top_n = max(rerank_top_n, top_k * 2)
        
        candidates = self.search(query, top_k=rerank_top_n, filter_expr=filter_expr, method=method)
        
        if not candidates:
            return []
        
        # Stage 2: Re-rank with cross-encoder
        if self.enable_reranking and self.reranker:
            # Prepare query-document pairs for re-ranker
            pairs = [[query, doc['text']] for doc in candidates]
            
            # Get re-ranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # 🔍 DEBUG: Print score ranges
            print(f"\n🔍 DEBUG Re-ranker Scores:")
            print(f"   Min: {min(rerank_scores):.4f}")
            print(f"   Max: {max(rerank_scores):.4f}")
            print(f"   Mean: {sum(rerank_scores)/len(rerank_scores):.4f}")
            
            # Extract vector scores
            vector_scores = [doc['score'] for doc in candidates]
            
            # Min-max normalization to [0, 1] range
            v_min, v_max = min(vector_scores), max(vector_scores)
            r_min, r_max = min(rerank_scores), max(rerank_scores)
            
            # Avoid division by zero
            v_range = v_max - v_min if v_max > v_min else 1.0
            r_range = r_max - r_min if r_max > r_min else 1.0
            
            for doc, r_score, v_score in zip(candidates, rerank_scores, vector_scores):
                # Normalize both scores to 0-1
                v_normalized = (v_score - v_min) / v_range
                r_normalized = (r_score - r_min) / r_range
                
                # Hybrid score: weighted combination
                hybrid_score = (alpha * r_normalized) + ((1 - alpha) * v_normalized)
                
                # Store all scores for transparency
                doc['vector_score'] = v_score
                doc['vector_normalized'] = round(v_normalized, 4)
                doc['rerank_score'] = float(r_score)
                doc['rerank_normalized'] = round(r_normalized, 4)
                doc['hybrid_score'] = round(hybrid_score, 4)
                doc['score'] = hybrid_score  # Use hybrid as primary score
            
            # Sort by hybrid score
            reranked = sorted(candidates, key=lambda x: x['hybrid_score'], reverse=True)
            return reranked[:top_k]
        else:
            # If re-ranking disabled, return original results
            return candidates[:top_k]
    
    def get_comparison_data(
        self, 
        query: str, 
        top_k: int = 5,
        filter_expr: str = None,
        rerank_top_n: Optional[int] = None,
        method: Literal["vector", "sparse", "hybrid"] = "vector"
    ) -> Dict:
        """
        Get structured comparison data for API response
        """
        import time
        start_time = time.time()
        
        # BEFORE: Search only
        search_results = self.search(query, top_k=top_k, filter_expr=filter_expr, method=method)
        
        # AFTER: With re-ranking
        if rerank_top_n is None:
            rerank_top_n = top_k * 4
        
        reranked_results = self.search_with_rerank(
            query, 
            top_k=top_k, 
            filter_expr=filter_expr,
            rerank_top_n=rerank_top_n,
            method=method
        )
        
        # Calculate metrics
        search_chunk_ids = [r['chunk_id'] for r in search_results]
        reranked_chunk_ids = [r['chunk_id'] for r in reranked_results]
        
        overlap = set(search_chunk_ids) & set(reranked_chunk_ids)
        new_results = [cid for cid in reranked_chunk_ids if cid not in search_chunk_ids]
        demoted_results = [cid for cid in search_chunk_ids if cid not in reranked_chunk_ids]
        
        avg_search = sum(r.get('vector_score', r.get('score', 0)) for r in reranked_results) / len(reranked_results) if reranked_results else 0
        avg_rerank = sum(r.get('rerank_score', r.get('score', 0)) for r in reranked_results) / len(reranked_results) if reranked_results else 0
        
        improvement_percent = ((avg_rerank - avg_search) / avg_search * 100) if avg_search > 0 else 0
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "top_k": top_k,
            "before_reranking": search_results,
            "after_reranking": reranked_results,
            "metrics": {
                "kept_count": len(overlap),
                "promoted_count": len(new_results),
                "demoted_count": len(demoted_results),
                "avg_score_before": round(avg_search, 4),
                "avg_score_after": round(avg_rerank, 4),
                "improvement_percent": round(improvement_percent, 2)
            },
            "latency_ms": round(latency_ms, 2)
        }
    
    def search_by_document(self, query: str, document_name: str, top_k: int = 5, 
                          rerank: bool = False, method: str = "vector") -> List[Dict]:
        """Search within a specific document"""
        filter_expr = f'document_name == "{document_name}"'
        if rerank:
            return self.search_with_rerank(query, top_k, filter_expr, method=method)
        return self.search(query, top_k, filter_expr, method=method)
   
    def search_by_page(self, query: str, document_name: str, page_idx: int, 
                      top_k: int = 3, rerank: bool = False, method: str = "vector") -> List[Dict]:
        """Search within a specific page of a document"""
        filter_expr = f'document_name == "{document_name}" && page_idx == {page_idx}'
        if rerank:
            return self.search_with_rerank(query, top_k, filter_expr, method=method)
        return self.search(query, top_k, filter_expr, method=method)
   
    def get_chunk_by_id(self, chunk_id: str) -> Dict:
        """Retrieve a specific chunk by global_chunk_id"""
        collection = self.get_collection()
       
        results = collection.query(
            expr=f'global_chunk_id == "{chunk_id}"',
            output_fields=[
                "global_chunk_id", "document_id", "source_file", "page_idx",
                "chunk_index", "section_hierarchy", "text", "char_count", "word_count"
            ]
        )
       
        if results:
            return results[0]
        return None
   
    def get_document_chunks(self, document_id: str, limit: int = 100) -> List[Dict]:
        """Get all chunks from a specific document"""
        collection = self.get_collection()
       
        results = collection.query(
            expr=f'document_id == "{document_id}"',
            limit=limit,
            output_fields=[
                "global_chunk_id", "document_id", "source_file", "page_idx",
                "chunk_index", "section_hierarchy", "text"
            ]
        )
       
        return results
   
    def list_documents(self) -> List[str]:
        """Get list of all unique documents in collection"""
        collection = self.get_collection()
       
        results = collection.query(
            expr="chunk_index >= 0",
            limit=10000,
            output_fields=["document_id"]
        )
       
        # Extract unique document IDs
        documents = list(set(r["document_id"] for r in results if r.get("document_id")))
        return sorted(documents)
   
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        collection = self.get_collection()
       
        return {
            "collection_name": self.collection_name,
            "total_entities": collection.num_entities,
            "schema": {
                "embedding_dim": 1024,
                "fields": [
                    "global_chunk_id", "document_id", "source_file", "page_idx",
                    "chunk_index", "section_hierarchy", "text", "char_count", "word_count"
                ]
            }
        }
   
    def health_check(self) -> Dict:
        """Check Milvus health and capabilities"""
        try:
            collection = self.get_collection()
            documents = self.list_documents()
            
            schema = collection.schema
            has_sparse = self._has_field("sparse_embedding")
            has_dense = self._has_field("dense_embedding") or self._has_field("embedding")
           
            return {
                "milvus_connected": True,
                "collection_exists": True,
                "collection_name": self.collection_name,
                "total_vectors": collection.num_entities,
                "total_documents": len(documents),
                "embedding_model": self.embedding_model_name,
                "has_dense_field": has_dense,
                "has_sparse_field": has_sparse,
                "hybrid_enabled": has_dense and has_sparse,
                "reranker_enabled": self.enable_reranking,
                "reranker_model": self.reranker_model_name if self.enable_reranking else None
            }
        except Exception as e:
            return {
                "milvus_connected": False,
                "collection_exists": False,
                "total_vectors": 0,
                "hybrid_enabled": False,
                "error": str(e)
            }


# Global instance
milvus_client = None


def get_milvus_client() -> MilvusClient:
    """Get or create Milvus client singleton"""
    global milvus_client
    if milvus_client is None:
        milvus_client = MilvusClient()
    return milvus_client


# Example usage
if __name__ == "__main__":
    client = get_milvus_client()
   
    # Health check
    print("\n🏥 Health Check:")
    health = client.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")
   
    query = "What is RUSA?"
    
    # Test different search methods
    print("\n🔍 Vector Search:")
    results = client.search(query, top_k=3, method="vector")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text'][:80]}...")
    
    # If hybrid is enabled
    if health.get('hybrid_enabled'):
        print("\n🔍 Hybrid Search (RRF):")
        results = client.search(query, top_k=3, method="hybrid")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['text'][:80]}...")
    
    # With reranking
    print("\n🔍 Vector Search + Reranking:")
    results = client.search_with_rerank(query, top_k=3, method="vector")
    for r in results:
        print(f"  [{r['score']:.4f}] Vector: {r.get('vector_score', 0):.4f} | Rerank: {r.get('rerank_score', 0):.4f}")
        print(f"      {r['text'][:80]}...")
