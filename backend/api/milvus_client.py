from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

class MilvusClient:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "VictorText")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize re-ranker (lazy loading)
        self.reranker_model_name = os.getenv("RERANKER_MODEL")
        self._reranker = None
        self.enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        
        # Connect to Milvus
        self.connect()
    
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
        """Generate embedding for query"""
        embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=False
        )
        return embedding[0].tolist()
   
    def search(self, query: str, top_k: int = 5, filter_expr: str = None) -> List[Dict]:
        """
        Search for similar vectors in VictorText collection
       
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_expr: Optional Milvus filter expression (e.g., 'document_name == "RUSA_final090913"')
       
        Returns:
            List of matching chunks with metadata
        """
        collection = self.get_collection()
       
        # Generate query embedding
        query_embedding = self.embed_query(query)
       
        # Search parameters
        search_params = {
            "metric_type": "IP",
            "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
        }
       
        # Output fields matching VictorText schema
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
            limit=top_k,
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
       
        return formatted_results
    
    def search_with_rerank(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_expr: str = None,
        rerank_top_n: Optional[int] = None,
        alpha: float = 0.5  # 50% vector, 50% rerank - BALANCED
    ) -> List[Dict]:
        """
        Two-stage retrieval with HYBRID SCORING: Dense vector search + Re-Ranking
        
        Args:
            query: Search query text
            top_k: Final number of results to return after re-ranking
            filter_expr: Optional Milvus filter expression
            rerank_top_n: Number of candidates to retrieve before re-ranking 
            alpha: Weight for re-ranker score (0.0 = all vector, 1.0 = all rerank)
                  Default 0.5 = balanced hybrid
        
        Returns:
            Re-ranked list of chunks with hybrid scores
        """
        # Stage 1: Retrieve more candidates for re-ranking
        if rerank_top_n is None:
            rerank_top_n = max(top_k * 4, 20)
        else:
            rerank_top_n = max(rerank_top_n, top_k * 2)
        
        candidates = self.search(query, top_k=rerank_top_n, filter_expr=filter_expr)
        
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
    
    def compare_search_methods(
        self, 
        query: str, 
        top_k: int = 5,
        filter_expr: str = None,
        rerank_top_n: Optional[int] = None
    ) -> Dict:
        """
        Compare vector search vs re-ranked search side-by-side
        
        Args:
            query: Search query text
            top_k: Number of results to show for each method
            filter_expr: Optional Milvus filter expression
            rerank_top_n: Candidate pool size for re-ranking
        
        Returns:
            Dictionary with comparison data and statistics
        """
        print(f"\n{'='*80}")
        print(f"🔬 COMPARING: Vector Search vs Re-ranked Search")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Top-K: {top_k}")
        print(f"{'='*80}\n")
        
        # Get vector-only results
        vector_results = self.search(query, top_k=top_k, filter_expr=filter_expr)
        
        # Get re-ranked results
        reranked_results = self.search_with_rerank(
            query, 
            top_k=top_k, 
            filter_expr=filter_expr,
            rerank_top_n=rerank_top_n
        )
        
        # Print side-by-side comparison
        print("\n📊 SIDE-BY-SIDE COMPARISON\n")
        
        for i in range(top_k):
            print(f"{'─'*80}")
            print(f"RANK #{i+1}")
            print(f"{'─'*80}")
            
            # Vector search result
            if i < len(vector_results):
                v_result = vector_results[i]
                print(f"\n🔵 VECTOR SEARCH:")
                print(f"   Score: {v_result['score']:.4f}")
                print(f"   Document: {v_result['document_name']}")
                print(f"   Page: {v_result['page_idx']}")
                print(f"   Chunk ID: {v_result['chunk_id']}")
                print(f"   Text: {v_result['text'][:150]}...")
            
            # Re-ranked result
            if i < len(reranked_results):
                r_result = reranked_results[i]
                print(f"\n🟢 RE-RANKED SEARCH:")
                print(f"   Rerank Score: {r_result.get('rerank_score', 0):.4f}")
                print(f"   Vector Score: {r_result.get('vector_score', 0):.4f}")
                print(f"   Score Delta: {r_result.get('rerank_score', 0) - r_result.get('vector_score', 0):+.4f}")
                print(f"   Document: {r_result['document_name']}")
                print(f"   Page: {r_result['page_idx']}")
                print(f"   Chunk ID: {r_result['chunk_id']}")
                print(f"   Text: {r_result['text'][:150]}...")
            
            print()
        
        # Calculate ranking changes
        print(f"\n{'='*80}")
        print(f"📈 RANKING ANALYSIS")
        print(f"{'='*80}\n")
        
        # Track which chunks appear in both results
        vector_chunk_ids = [r['chunk_id'] for r in vector_results]
        reranked_chunk_ids = [r['chunk_id'] for r in reranked_results]
        
        # Calculate overlap
        overlap = set(vector_chunk_ids) & set(reranked_chunk_ids)
        overlap_percent = (len(overlap) / top_k) * 100 if top_k > 0 else 0
        
        print(f"Results Overlap: {len(overlap)}/{top_k} ({overlap_percent:.1f}%)")
        print(f"New Results (Re-ranking): {top_k - len(overlap)}")
        
        # Show position changes for overlapping results
        position_changes = []
        for chunk_id in overlap:
            v_pos = vector_chunk_ids.index(chunk_id) + 1
            r_pos = reranked_chunk_ids.index(chunk_id) + 1
            change = v_pos - r_pos  # Positive = moved up, Negative = moved down
            position_changes.append({
                'chunk_id': chunk_id,
                'vector_rank': v_pos,
                'rerank_rank': r_pos,
                'change': change
            })
        
        if position_changes:
            print(f"\n📊 Position Changes:")
            position_changes.sort(key=lambda x: abs(x['change']), reverse=True)
            for pc in position_changes[:5]:  # Show top 5 changes
                direction = "⬆️" if pc['change'] > 0 else "⬇️" if pc['change'] < 0 else "➡️"
                print(f"   {direction} Chunk {pc['chunk_id'][:12]}... : Rank {pc['vector_rank']} → {pc['rerank_rank']} ({pc['change']:+d})")
        
        # Score statistics
        if reranked_results:
            avg_vector_score = sum(r.get('vector_score', 0) for r in reranked_results) / len(reranked_results)
            avg_rerank_score = sum(r.get('rerank_score', 0) for r in reranked_results) / len(reranked_results)
            
            print(f"\n📊 Score Statistics:")
            print(f"   Avg Vector Score: {avg_vector_score:.4f}")
            print(f"   Avg Rerank Score: {avg_rerank_score:.4f}")
            print(f"   Avg Delta: {avg_rerank_score - avg_vector_score:+.4f}")
        
        print(f"\n{'='*80}\n")
        
        # Return structured comparison data
        return {
            "query": query,
            "top_k": top_k,
            "vector_results": vector_results,
            "reranked_results": reranked_results,
            "overlap_count": len(overlap),
            "overlap_percent": overlap_percent,
            "position_changes": position_changes,
            "new_results_count": top_k - len(overlap)
        }
    
    def compare_before_after_reranking(
        self, 
        query: str, 
        top_k: int = 5,
        filter_expr: str = None,
        show_full_text: bool = False
    ) -> Dict:
        """
        Simple before/after comparison: Vector Search → Re-ranking
        Shows how re-ranking improves the initial vector search results
        
        Args:
            query: Search query text
            top_k: Number of results to compare
            filter_expr: Optional Milvus filter expression
            show_full_text: If True, show full text instead of preview
        
        Returns:
            Dictionary with before/after results and improvement metrics
        """
        print(f"\n{'='*100}")
        print(f"📊 BEFORE vs AFTER RE-RANKING")
        print(f"{'='*100}")
        print(f"Query: '{query}'")
        print(f"{'='*100}\n")
        
        # BEFORE: Vector search only
        print("⏳ Getting initial vector search results...\n")
        vector_results = self.search(query, top_k=top_k, filter_expr=filter_expr)
        
        # AFTER: With re-ranking
        print("⏳ Applying re-ranker...\n")
        reranked_results = self.search_with_rerank(
            query, 
            top_k=top_k, 
            filter_expr=filter_expr,
            rerank_top_n=top_k * 4  # Retrieve 4x candidates for re-ranking
        )
        
        print(f"{'='*100}")
        print(f"🔵 BEFORE RE-RANKING (Vector Similarity Only)")
        print(f"{'='*100}\n")
        
        for i, result in enumerate(vector_results, 1):
            text_preview = result['text'] if show_full_text else result['text'][:200] + "..."
            print(f"#{i} | Score: {result['score']:.4f} | Doc: {result['document_name']} | Page: {result['page_idx']}")
            print(f"    Text: {text_preview}")
            print()
        
        print(f"\n{'='*100}")
        print(f"🟢 AFTER RE-RANKING (Re-ranked by Relevance)")
        print(f"{'='*100}\n")
        
        for i, result in enumerate(reranked_results, 1):
            text_preview = result['text'] if show_full_text else result['text'][:200] + "..."
            vector_score = result.get('vector_score', 0)
            rerank_score = result.get('rerank_score', 0)
            improvement = rerank_score - vector_score
            
            print(f"#{i} | Rerank: {rerank_score:.4f} | Vector: {vector_score:.4f} | Δ {improvement:+.4f}")
            print(f"    Doc: {result['document_name']} | Page: {result['page_idx']}")
            print(f"    Text: {text_preview}")
            print()
        
        # Calculate metrics
        vector_chunk_ids = [r['chunk_id'] for r in vector_results]
        reranked_chunk_ids = [r['chunk_id'] for r in reranked_results]
        
        # How many results stayed in top-k
        overlap = set(vector_chunk_ids) & set(reranked_chunk_ids)
        
        # How many new results were promoted
        new_results = [cid for cid in reranked_chunk_ids if cid not in vector_chunk_ids]
        
        # How many old results were demoted
        demoted_results = [cid for cid in vector_chunk_ids if cid not in reranked_chunk_ids]
        
        print(f"{'='*100}")
        print(f"📈 IMPROVEMENT SUMMARY")
        print(f"{'='*100}\n")
        
        print(f"✅ Results kept in top-{top_k}: {len(overlap)}/{top_k} ({len(overlap)/top_k*100:.0f}%)")
        print(f"⬆️  New results promoted: {len(new_results)}")
        print(f"⬇️  Results demoted out of top-{top_k}: {len(demoted_results)}")
        
        # Show which results changed
        if new_results:
            print(f"\n🆕 NEW RESULTS PROMOTED BY RE-RANKER:")
            for cid in new_results:
                idx = reranked_chunk_ids.index(cid)
                result = reranked_results[idx]
                print(f"    → Rank #{idx+1}: {result['document_name']} (Page {result['page_idx']}) - Score: {result.get('rerank_score', 0):.4f}")
        
        if demoted_results:
            print(f"\n📉 RESULTS DEMOTED (were in top-{top_k}, now removed):")
            for cid in demoted_results:
                idx = vector_chunk_ids.index(cid)
                result = vector_results[idx]
                print(f"    → Was Rank #{idx+1}: {result['document_name']} (Page {result['page_idx']}) - Score: {result['score']:.4f}")
        
        # Show position changes for results that stayed
        print(f"\n🔄 POSITION CHANGES (for results that stayed in top-{top_k}):")
        for cid in overlap:
            old_rank = vector_chunk_ids.index(cid) + 1
            new_rank = reranked_chunk_ids.index(cid) + 1
            change = old_rank - new_rank
            
            if change != 0:
                direction = "⬆️ UP" if change > 0 else "⬇️ DOWN"
                result = reranked_results[new_rank - 1]
                print(f"    {direction} {abs(change)} position(s): Rank {old_rank} → {new_rank} | {result['document_name']} (Page {result['page_idx']})")
        
        # Score improvements
        if reranked_results:
            avg_vector = sum(r.get('vector_score', 0) for r in reranked_results) / len(reranked_results)
            avg_rerank = sum(r.get('rerank_score', 0) for r in reranked_results) / len(reranked_results)
            
            print(f"\n📊 SCORE ANALYSIS:")
            print(f"    Average Vector Score (BEFORE): {avg_vector:.4f}")
            print(f"    Average Rerank Score (AFTER):  {avg_rerank:.4f}")
            print(f"    Average Improvement: {avg_rerank - avg_vector:+.4f} ({(avg_rerank - avg_vector)/avg_vector*100:+.1f}%)")
        
        print(f"\n{'='*100}\n")
        
        return {
            "query": query,
            "top_k": top_k,
            "before": vector_results,
            "after": reranked_results,
            "kept_count": len(overlap),
            "promoted_count": len(new_results),
            "demoted_count": len(demoted_results),
            "avg_score_before": avg_vector if reranked_results else 0,
            "avg_score_after": avg_rerank if reranked_results else 0,
            "improvement_percent": ((avg_rerank - avg_vector)/avg_vector*100) if (reranked_results and avg_vector > 0) else 0
        }
    
    def get_comparison_data(
        self, 
        query: str, 
        top_k: int = 5,
        filter_expr: str = None,
        rerank_top_n: Optional[int] = None
    ) -> Dict:
        """
        Get structured comparison data for API response
        (Returns data instead of printing to console)
        
        Args:
            query: Search query text
            top_k: Number of results to compare
            filter_expr: Optional Milvus filter expression
            rerank_top_n: Number of candidates for re-ranking
        
        Returns:
            Dictionary with before/after results and metrics
        """
        import time
        start_time = time.time()
        
        # BEFORE: Vector search only
        vector_results = self.search(query, top_k=top_k, filter_expr=filter_expr)
        
        # AFTER: With re-ranking
        if rerank_top_n is None:
            rerank_top_n = top_k * 4
        
        reranked_results = self.search_with_rerank(
            query, 
            top_k=top_k, 
            filter_expr=filter_expr,
            rerank_top_n=rerank_top_n
        )
        
        # Calculate metrics
        vector_chunk_ids = [r['chunk_id'] for r in vector_results]
        reranked_chunk_ids = [r['chunk_id'] for r in reranked_results]
        
        overlap = set(vector_chunk_ids) & set(reranked_chunk_ids)
        new_results = [cid for cid in reranked_chunk_ids if cid not in vector_chunk_ids]
        demoted_results = [cid for cid in vector_chunk_ids if cid not in reranked_chunk_ids]
        
        avg_vector = sum(r.get('vector_score', r.get('score', 0)) for r in reranked_results) / len(reranked_results) if reranked_results else 0
        avg_rerank = sum(r.get('rerank_score', r.get('score', 0)) for r in reranked_results) / len(reranked_results) if reranked_results else 0
        
        improvement_percent = ((avg_rerank - avg_vector) / avg_vector * 100) if avg_vector > 0 else 0
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "top_k": top_k,
            "before_reranking": vector_results,
            "after_reranking": reranked_results,
            "metrics": {
                "kept_count": len(overlap),
                "promoted_count": len(new_results),
                "demoted_count": len(demoted_results),
                "avg_score_before": round(avg_vector, 4),
                "avg_score_after": round(avg_rerank, 4),
                "improvement_percent": round(improvement_percent, 2)
            },
            "latency_ms": round(latency_ms, 2)
        }
    
    def search_by_document(self, query: str, document_name: str, top_k: int = 5, rerank: bool = False) -> List[Dict]:
        """Search within a specific document"""
        filter_expr = f'document_name == "{document_name}"'
        if rerank:
            return self.search_with_rerank(query, top_k, filter_expr)
        return self.search(query, top_k, filter_expr)
   
    def search_by_page(self, query: str, document_name: str, page_idx: int, top_k: int = 3, rerank: bool = False) -> List[Dict]:
        """Search within a specific page of a document"""
        filter_expr = f'document_name == "{document_name}" && page_idx == {page_idx}'
        if rerank:
            return self.search_with_rerank(query, top_k, filter_expr)
        return self.search(query, top_k, filter_expr)
   
    def get_chunk_by_id(self, chunk_id: str) -> Dict:
        """Retrieve a specific chunk by chunk_id"""
        collection = self.get_collection()
       
        results = collection.query(
            expr=f'chunk_id == "{chunk_id}"',
            output_fields=[
                "document_name", "document_id", "chunk_id", "global_chunk_id",
                "page_idx", "chunk_index", "section_hierarchy", "heading_context",
                "text", "char_count", "word_count"
            ]
        )
       
        if results:
            return results[0]
        return None
   
    def get_document_chunks(self, document_name: str, limit: int = 100) -> List[Dict]:
        """Get all chunks from a specific document"""
        collection = self.get_collection()
       
        results = collection.query(
            expr=f'document_name == "{document_name}"',
            limit=limit,
            output_fields=[
                "document_name", "document_id", "chunk_id", "page_idx",
                "chunk_index", "heading_context", "text"
            ]
        )
       
        return results
   
    def list_documents(self) -> List[str]:
        """Get list of all unique documents in collection"""
        collection = self.get_collection()
       
        # Query all document names (Milvus doesn't have distinct, so we get all and deduplicate)
        results = collection.query(
            expr="chunk_index >= 0",
            limit=10000,
            output_fields=["document_name"]
        )
       
        # Extract unique document names
        documents = list(set(r["document_name"] for r in results))
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
                    "document_name", "document_id", "chunk_id", "global_chunk_id",
                    "page_idx", "chunk_index", "section_hierarchy", "heading_context",
                    "text", "char_count", "word_count"
                ]
            }
        }
   
    def health_check(self) -> Dict:
        """Check Milvus health"""
        try:
            collection = self.get_collection()
            documents = self.list_documents()
           
            return {
                "milvus_connected": True,
                "collection_exists": True,
                "collection_name": self.collection_name,
                "total_vectors": collection.num_entities,
                "total_documents": len(documents),
                "embedding_model": self.embedding_model_name,
                "reranker_enabled": self.enable_reranking,
                "reranker_model": self.reranker_model_name if self.enable_reranking else None
            }
        except Exception as e:
            return {
                "milvus_connected": False,
                "collection_exists": False,
                "total_vectors": 0,
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
   
    # Simple before/after comparison
    query = "What is RUSA?"
    comparison = client.compare_before_after_reranking(query, top_k=5)
    
    # Access improvement metrics
    print(f"\n💡 Quick Stats:")
    print(f"   • {comparison['promoted_count']} new results found by re-ranker")
    print(f"   • {comparison['improvement_percent']:.1f}% score improvement on average")