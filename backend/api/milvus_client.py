from pymilvus import connections, Collection, utility, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import os
from typing import List, Dict, Literal, Optional
from dotenv import load_dotenv

load_dotenv()


class MilvusClient:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("COLLECTION_NAME", "VictorText2")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        
        # ✅ CORRECT FIELD NAMES from schema
        self.dense_field = "dense_embedding"  # NOT "dense_vector"
        self.sparse_field = "sparse_embedding"  # NOT "sparse_vector"
        
        # Dense embeddings - SentenceTransformer (cached)
        print(f"Loading dense embedding model: {self.embedding_model_name}")
        self.dense_model = SentenceTransformer(self.embedding_model_name)
        
        # Sparse embeddings - FlagEmbedding (only for sparse)
        print(f"Loading sparse embedding model: {self.embedding_model_name}")
        self.sparse_model = BGEM3FlagModel(
            self.embedding_model_name,
            use_fp16=False  # No fp16
        )
        
        self.connect()
        self.collection = self.get_collection()

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
    
    def embed_query_dense(self, query: str) -> List[float]:
        """Generate dense embedding using SentenceTransformer"""
        embedding = self.dense_model.encode(
            [query],
            normalize_embeddings=True
        )
        return embedding[0].tolist()
    
    def embed_query_sparse(self, query: str) -> Dict[int, float]:
        """
        Generate sparse embedding using FlagEmbedding BGEM3
        CORRECTLY uses the encode method with proper parameters
        """
        try:
            # ✅ CORRECT METHOD: Use encode with batch_size=1 for single query
            output = self.sparse_model.encode(
                sentences=[query],  # Must be list of strings
                batch_size=1,
                max_length=8192,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            # Debug output structure
            print(f"   🔍 Sparse output type: {type(output)}")
            if isinstance(output, dict):
                print(f"   🔍 Sparse output keys: {output.keys()}")
            
            # Handle different output formats
            if output is None:
                raise ValueError("Sparse model returned None - model may not be loaded correctly")
            
            # Check for lexical_weights in output
            if isinstance(output, dict) and 'lexical_weights' in output:
                sparse_weights = output['lexical_weights']
                
                # If it's a list, take first element
                if isinstance(sparse_weights, list) and len(sparse_weights) > 0:
                    sparse_dict = sparse_weights[0]
                else:
                    sparse_dict = sparse_weights
                
                # Convert to proper format
                result = {int(k): float(v) for k, v in sparse_dict.items()}
                print(f"   ✅ Generated sparse vector with {len(result)} tokens")
                return result
            else:
                raise ValueError(f"Unexpected sparse output format. Type: {type(output)}, Keys: {output.keys() if isinstance(output, dict) else 'N/A'}")
            
        except Exception as e:
            print(f"❌ CRITICAL: Sparse embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # ❌ DO NOT FALLBACK - Raise the error to maintain hybrid search integrity
            raise RuntimeError(f"Sparse embedding failed - hybrid search cannot proceed: {e}")

    def _get_output_fields(self) -> List[str]:
        """Essential output fields for VictorText2 schema with selective metadata"""
        return [
            # Core document identification
            "document_name", "document_id", "chunk_id", "global_chunk_id",
            "page_idx", "chunk_index", 
            # Hierarchical context
            "section_hierarchy", "heading_context",
            # Content
            "text", "char_count", "word_count",
            # Key metadata for filtering and context
            "published_date", "language", "Category", "document_type", 
            "ministry", "source_reference"
        ]
    
    def _format_results(self, results, use_distance: bool = False) -> List[Dict]:
        """Format search results with VictorText2 schema and selective metadata"""
        formatted_results = []
        for hits in results:
            for hit in hits:
                score = float(hit.distance) if use_distance else float(hit.score)
                formatted_results.append({
                    "score": score,
                    # Core fields
                    "document_name": hit.entity.get("document_name"),
                    "document_id": hit.entity.get("document_id"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "global_chunk_id": hit.entity.get("global_chunk_id"),
                    "source_file": hit.entity.get("document_name"),
                    "page_idx": hit.entity.get("page_idx"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "section_hierarchy": hit.entity.get("section_hierarchy"),
                    "heading_context": hit.entity.get("heading_context"),
                    "text": hit.entity.get("text"),
                    "char_count": hit.entity.get("char_count"),
                    "word_count": hit.entity.get("word_count"),
                    # Essential metadata
                    "published_date": hit.entity.get("published_date"),
                    "language": hit.entity.get("language"),
                    "category": hit.entity.get("Category"),
                    "document_type": hit.entity.get("document_type"),
                    "ministry": hit.entity.get("ministry"),
                    "source_reference": hit.entity.get("source_reference")
                })
        return formatted_results
    
    # Simplified search - no fallback, clean filtering
    def search(self, query: str, top_k: int = 5, method: str = "hybrid", 
               filter_expr: str = None):
        """
        Hybrid search with optional metadata filtering
        """
        print(f"\n🔍 MILVUS SEARCH")
        print(f"   Method: {method}")
        print(f"   Top-K: {top_k}")
        print(f"   Filter: {filter_expr or 'None'}")
        
        try:
            # ✅ Add quality filter (always exclude junk chunks)
            quality_filter = "char_count >= 50"
            
            # Combine with user filter
            if filter_expr:
                combined_filter = f"({filter_expr}) && {quality_filter}"
            else:
                combined_filter = quality_filter
            
            print(f"   🔍 Final filter: {combined_filter}")
            
            if method == "hybrid":
                # Generate embeddings
                print(f"   🔄 Generating embeddings...")
                dense_vec = self.dense_model.encode(query).tolist()
                sparse_vec = self.embed_query_sparse(query)
                print(f"   ✅ Dense: {len(dense_vec)}D, Sparse: {len(sparse_vec)} tokens")
                
                # Search params
                dense_params = {"metric_type": "IP", "params": {"nprobe": 10}}
                sparse_params = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
                
                # Build requests
                dense_req = AnnSearchRequest(
                    data=[dense_vec],
                    anns_field=self.dense_field,
                    param=dense_params,
                    limit=top_k,
                    expr=combined_filter
                )
                
                sparse_req = AnnSearchRequest(
                    data=[sparse_vec],
                    anns_field=self.sparse_field,
                    param=sparse_params,
                    limit=top_k,
                    expr=combined_filter
                )
                
                # ✅ Hybrid search
                print(f"   🔄 Executing hybrid search...")
                results = self.collection.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=RRFRanker(),
                    limit=top_k,
                    output_fields=["*"]
                )
                
                formatted_results = self._format_results(results)
                filtered_results = self._filter_quality_results(formatted_results, top_k)
                
                print(f"   ✅ Results: {len(filtered_results)}")
                return filtered_results
            
            elif method == "vector":
                dense_vec = self.dense_model.encode(query).tolist()
                
                results = self.collection.search(
                    data=[dense_vec],
                    anns_field=self.dense_field,
                    param={"metric_type": "IP", "params": {"nprobe": 10}},
                    limit=top_k,
                    expr=combined_filter,
                    output_fields=["*"]
                )
                
                formatted_results = self._format_results(results)
                filtered_results = self._filter_quality_results(formatted_results, top_k)
                
                print(f"   ✅ Results: {len(filtered_results)}")
                return filtered_results
            
            elif method == "sparse":
                sparse_vec = self.embed_query_sparse(query)
                
                results = self.collection.search(
                    data=[sparse_vec],
                    anns_field=self.sparse_field,
                    param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                    limit=top_k,
                    expr=combined_filter,
                    output_fields=["*"]
                )
                
                formatted_results = self._format_results(results)
                filtered_results = self._filter_quality_results(formatted_results, top_k)
                
                print(f"   ✅ Results: {len(filtered_results)}")
                return filtered_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _filter_quality_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        """
        Filter out low-quality chunks (too short, single characters, etc.)
        """
        quality_results = []
        
        for result in results:
            text = result.get('text', '')
            char_count = result.get('char_count', 0)
            word_count = result.get('word_count', 0)
            
            # ✅ QUALITY CRITERIA:
            # 1. At least 50 characters
            # 2. At least 5 words
            # 3. Not just whitespace
            # 4. Has meaningful content (not just single chars like "a", "i")
            
            if (char_count >= 50 and 
                word_count >= 5 and 
                text.strip() and
                len(text.strip()) >= 50):
                
                quality_results.append(result)
                
                # Stop when we have enough quality results
                if len(quality_results) >= top_k:
                    break
        
        # ✅ Log quality filtering
        if len(quality_results) < len(results):
            filtered_out = len(results) - len(quality_results)
            print(f"   🧹 Filtered out {filtered_out} low-quality chunks")
            print(f"   📊 Quality results: {len(quality_results)}/{top_k} requested")
        
        return quality_results[:top_k]
    
    def _vector_search(self, query: str, top_k: int = 5, filter_expr: str = None) -> List[Dict]:
        """Dense vector search using HNSW - FIXED METRIC TYPE"""
        collection = self.get_collection()
        dense_embedding = self.embed_query_dense(query)
        
        search_params = {
            "metric_type": "IP",  # ✅ FIXED: IP not COSINE (matches your index)
            "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
        }
        
        results = collection.search(
            data=[dense_embedding],
            anns_field=self.dense_field,  # "dense_embedding"
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
            anns_field=self.sparse_field,  # "sparse_embedding"
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
        
        # ✅ Dense search request with IP metric
        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field=self.dense_field,  # "dense_embedding"
            param={
                "metric_type": "IP",  # ✅ FIXED: IP not COSINE
                "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
            },
            limit=top_k * 2,
            expr=filter_expr
        )
        
        # Sparse search request
        sparse_req = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field=self.sparse_field,  # "sparse_embedding"
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

    # Specialized search methods (unchanged - they use the fixed search method)
    def search_by_document(self, query: str, document_id: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific document"""
        filter_expr = f'document_id == "{document_id}"'
        return self.search(query, top_k, method="hybrid", filter_expr=filter_expr)
    
    def search_by_category(self, query: str, category: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific category"""
        filter_expr = f'Category == "{category}"'
        return self.search(query, top_k, method="hybrid", filter_expr=filter_expr)
    
    def search_by_ministry(self, query: str, ministry: str, top_k: int = 5) -> List[Dict]:
        """Search within documents from a specific ministry"""
        filter_expr = f'ministry == "{ministry}"'
        return self.search(query, top_k, method="hybrid", filter_expr=filter_expr)
    
    def search_by_language(self, query: str, language: str, top_k: int = 5) -> List[Dict]:
        """Search within documents of a specific language"""
        filter_expr = f'language == "{language}"'
        return self.search(query, top_k, method="hybrid", filter_expr=filter_expr)
    
    def search_by_date_range(self, query: str, start_date: str, end_date: str, top_k: int = 5) -> List[Dict]:
        """Search within a date range (format: YYYY-MM-DD)"""
        filter_expr = f'published_date >= "{start_date}" && published_date <= "{end_date}"'
        return self.search(query, top_k, method="hybrid", filter_expr=filter_expr)
    
    def search_by_document_type(self, query: str, doc_type: str, top_k: int = 5) -> List[Dict]:
        """Search within documents of a specific type"""
        filter_expr = f'document_type == "{doc_type}"'
        return self.search(query, top_k, method="hybrid", filter_expr=filter_expr)
    
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
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get all available filter values from collection"""
        collection = self.get_collection()
        
        # Query all documents to get unique filter values
        results = collection.query(
            expr="chunk_index >= 0",
            limit=10000,
            output_fields=["Category", "ministry", "document_type", "language", "published_date"]
        )
        
        # Extract unique values for each filter
        categories = list(set(r.get("Category") for r in results if r.get("Category")))
        ministries = list(set(r.get("ministry") for r in results if r.get("ministry")))
        doc_types = list(set(r.get("document_type") for r in results if r.get("document_type")))
        languages = list(set(r.get("language") for r in results if r.get("language")))
        dates = [r.get("published_date") for r in results if r.get("published_date")]
        
        return {
            "categories": sorted(categories),
            "ministries": sorted(ministries),
            "document_types": sorted(doc_types),
            "languages": sorted(languages),
            "date_range": {
                "min": min(dates) if dates else None,
                "max": max(dates) if dates else None
            }
        }
    
    def health_check(self) -> Dict:
        """Check Milvus health and hybrid search support for VictorText2"""
        try:
            collection = self.get_collection()
            
            schema = collection.schema
            # Check for CORRECT field names
            has_sparse = any(f.name == self.sparse_field for f in schema.fields)
            has_dense = any(f.name == self.dense_field for f in schema.fields)
            
            documents = self.list_documents()
            
            # Test sparse embedding generation
            sparse_working = False
            try:
                test_sparse = self.embed_query_sparse("test query")
                sparse_working = len(test_sparse) > 0
            except:
                pass
            
            return {
                "milvus_connected": True,
                "collection_exists": True,
                "collection_name": self.collection_name,
                "schema_version": "VictorText2",
                "total_vectors": collection.num_entities,
                "total_documents": len(documents),
                "embedding_model": self.embedding_model_name,
                "dense_field_name": self.dense_field,
                "sparse_field_name": self.sparse_field,
                "has_dense_field": has_dense,
                "has_sparse_field": has_sparse,
                "sparse_model_working": sparse_working,
                "hybrid_search_enabled": has_dense and has_sparse and sparse_working
            }
        except Exception as e:
            return {
                "milvus_connected": False,
                "collection_exists": False,
                "total_vectors": 0,
                "error": str(e)
            }


# Global singleton
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
    
    print("\n🏥 Health Check:")
    health = client.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    if not health.get("sparse_model_working"):
        print("\n❌ WARNING: Sparse model not working - hybrid search will fail!")
        print("   Check BGEM3FlagModel initialization and encode() method")
    
    print("\n📚 Available Documents:")
    docs = client.list_documents()
    print(f"Found {len(docs)} documents")
    
    print("\n🎯 Available Filters:")
    filters = client.get_available_filters()
    for filter_type, values in filters.items():
        if isinstance(values, dict):
            print(f"  {filter_type}:")
            for k, v in values.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {filter_type}: {len(values)} options")
            print(f"    {values[:5]}{'...' if len(values) > 5 else ''}")
    
    query = "What is RUSA?"
    
    print("\n🔍 Testing Hybrid Search:")
    try:
        results = client.search(query, top_k=3, method="hybrid")
        print(f"✅ SUCCESS: {len(results)} results")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['text'][:80]}...")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("   Hybrid search requires both dense AND sparse embeddings to work")