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
        """Generate sparse embedding using FlagEmbedding"""
        output = self.sparse_model.encode(
            [query],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        sparse_weights = output['lexical_weights'][0]
        return {int(k): float(v) for k, v in sparse_weights.items()}
    
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
                    "source_file": hit.entity.get("document_name"),  # For compatibility
                    "page_idx": hit.entity.get("page_idx"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    # Hierarchical context
                    "section_hierarchy": hit.entity.get("section_hierarchy"),
                    "heading_context": hit.entity.get("heading_context"),
                    # Content
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
    
    def search(self, query: str, top_k: int = 5, filter_expr: str = None, 
               method: Literal["vector", "sparse", "hybrid"] = "hybrid",
               dense_weight: float = 0.7, sparse_weight: float = 0.3) -> List[Dict]:
        """
        Unified search method for VictorText2 collection
        
        Args:
            query: Search query text
            top_k: Number of results
            filter_expr: Optional Milvus filter expression
            method: 'vector', 'sparse', or 'hybrid'
            dense_weight: Weight for dense search (used in hybrid mode)
            sparse_weight: Weight for sparse search (used in hybrid mode)
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
        
        results = collection.search(
            data=[dense_embedding],
            anns_field="dense_embedding",
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
        
        # Dense search request
        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={
                "metric_type": "IP",
                "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
            },
            limit=top_k * 2,
            expr=filter_expr  # Filter goes here
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
            expr=filter_expr  # Filter goes here
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
    
    # Specialized search methods with filters
    def search_by_document(self, query: str, document_id: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific document"""
        filter_expr = f'document_id == "{document_id}"'
        return self.search(query, top_k, filter_expr)
    
    def search_by_category(self, query: str, category: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific category"""
        filter_expr = f'Category == "{category}"'
        return self.search(query, top_k, filter_expr)
    
    def search_by_ministry(self, query: str, ministry: str, top_k: int = 5) -> List[Dict]:
        """Search within documents from a specific ministry"""
        filter_expr = f'ministry == "{ministry}"'
        return self.search(query, top_k, filter_expr)
    
    def search_by_language(self, query: str, language: str, top_k: int = 5) -> List[Dict]:
        """Search within documents of a specific language"""
        filter_expr = f'language == "{language}"'
        return self.search(query, top_k, filter_expr)
    
    def search_by_date_range(self, query: str, start_date: str, end_date: str, top_k: int = 5) -> List[Dict]:
        """Search within a date range (format: YYYY-MM-DD)"""
        filter_expr = f'published_date >= "{start_date}" && published_date <= "{end_date}"'
        return self.search(query, top_k, filter_expr)
    
    def search_by_document_type(self, query: str, doc_type: str, top_k: int = 5) -> List[Dict]:
        """Search within documents of a specific type"""
        filter_expr = f'document_type == "{doc_type}"'
        return self.search(query, top_k, filter_expr)
    
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
        """Get available filter values for different metadata fields"""
        collection = self.get_collection()
        
        # Sample documents to get filter options
        results = collection.query(
            expr="chunk_index >= 0",
            limit=1000,
            output_fields=["Category", "ministry", "document_type", "language"]
        )
        
        filters = {
            "categories": list(set(r["Category"] for r in results if r.get("Category"))),
            "ministries": list(set(r["ministry"] for r in results if r.get("ministry"))),
            "document_types": list(set(r["document_type"] for r in results if r.get("document_type"))),
            "languages": list(set(r["language"] for r in results if r.get("language")))
        }
        
        return {k: sorted(v) for k, v in filters.items()}
    
    def health_check(self) -> Dict:
        """Check Milvus health and hybrid search support for VictorText2"""
        try:
            collection = self.get_collection()
            
            schema = collection.schema
            has_sparse = any(f.name == "sparse_embedding" for f in schema.fields)
            has_dense = any(f.name == "dense_embedding" for f in schema.fields)
            
            documents = self.list_documents()
            
            return {
                "milvus_connected": True,
                "collection_exists": True,
                "collection_name": self.collection_name,
                "schema_version": "VictorText2",
                "total_vectors": collection.num_entities,
                "total_documents": len(documents),
                "embedding_model": self.embedding_model_name,
                "has_dense_field": has_dense,
                "has_sparse_field": has_sparse,
                "hybrid_search_enabled": has_dense and has_sparse
            }
        except Exception as e:
            return {
                "milvus_connected": False,
                "collection_exists": False,
                "total_vectors": 0,
                "hybrid_search_enabled": False,
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
    print(client.health_check())
    
    print("\n📚 Available Documents:")
    docs = client.list_documents()
    print(f"Found {len(docs)} documents")
    
    print("\n🎯 Available Filters:")
    filters = client.get_available_filters()
    for filter_type, values in filters.items():
        print(f"  {filter_type}: {len(values)} options")
        print(f"    {values[:5]}{'...' if len(values) > 5 else ''}")
    
    query = "What is RUSA?"
    
    print("\n🔍 Vector Search:")
    results = client.search(query, top_k=3, method="vector")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text'][:80]}...")
    
    print("\n🔍 Sparse Search:")
    results = client.search(query, top_k=3, method="sparse")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text'][:80]}...")
    
    print("\n🔍 Hybrid Search (RRF):")
    results = client.search(query, top_k=3, method="hybrid")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text'][:80]}...")