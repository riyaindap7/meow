# from pymilvus import connections, Collection, utility
# from sentence_transformers import SentenceTransformer
# import os
# from typing import List, Dict
# from dotenv import load_dotenv

# load_dotenv()

# class MilvusClient:
#     def __init__(self):
#         self.host = os.getenv("MILVUS_HOST", "localhost")
#         self.port = os.getenv("MILVUS_PORT", "19530")
#         self.collection_name = os.getenv("COLLECTION_NAME", "pdf_vectors")
#         self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        
#         # Initialize embedding model
#         print(f"Loading embedding model: {self.embedding_model_name}")
#         self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
#         # Connect to Milvus
#         self.connect()
        
#     def connect(self):
#         """Connect to Milvus"""
#         try:
#             connections.connect("default", host=self.host, port=self.port)
#             print(f"✅ Connected to Milvus at {self.host}:{self.port}")
#         except Exception as e:
#             print(f"❌ Failed to connect to Milvus: {e}")
#             raise
    
#     def get_collection(self) -> Collection:
#         """Get collection instance"""
#         if not utility.has_collection(self.collection_name):
#             raise ValueError(f"Collection '{self.collection_name}' does not exist")
        
#         collection = Collection(self.collection_name)
#         collection.load()
#         return collection
    
#     def embed_query(self, query: str) -> List[float]:
#         """Generate embedding for query"""
#         embedding = self.embedding_model.encode(
#             [query],
#             normalize_embeddings=True
#         )
#         return embedding[0].tolist()
    
#     def search(self, query: str, top_k: int = 3) -> List[Dict]:
#         """Search for similar vectors"""
#         collection = self.get_collection()
        
#         # Generate query embedding
#         query_embedding = self.embed_query(query)
        
#         # Search parameters
#         search_params = {
#             "metric_type": "IP",
#             "params": {"ef": int(os.getenv("SEARCH_EF", 64))}
#         }
        
#         # Execute search
#         results = collection.search(
#             data=[query_embedding],
#             anns_field="embedding",
#             param=search_params,
#             limit=top_k,
#             output_fields=["text", "source", "page"]
#         )
        
#         # Format results
#         formatted_results = []
#         for hits in results:
#             for hit in hits:
#                 formatted_results.append({
#                     "text": hit.entity.get("text"),
#                     "source": hit.entity.get("source"),
#                     "page": hit.entity.get("page"),
#                     "score": float(hit.score)
#                 })
        
#         return formatted_results
    
#     def health_check(self) -> Dict:
#         """Check Milvus health"""
#         try:
#             collection = self.get_collection()
#             return {
#                 "milvus_connected": True,
#                 "collection_exists": True,
#                 "total_vectors": collection.num_entities
#             }
#         except Exception as e:
#             return {
#                 "milvus_connected": False,
#                 "collection_exists": False,
#                 "total_vectors": 0,
#                 "error": str(e)
#             }

# # Global instance
# milvus_client = None

# def get_milvus_client() -> MilvusClient:
#     """Get or create Milvus client singleton"""
#     global milvus_client
#     if milvus_client is None:
#         milvus_client = MilvusClient()
#     return milvus_client




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
        self.collection_name = os.getenv("MILVUS_COLLECTION", "VictorText")
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
        """Common output fields for all searches"""
        return [
            "document_name", "document_id", "chunk_id", "global_chunk_id",
            "page_idx", "chunk_index", "section_hierarchy", "heading_context",
            "text", "char_count", "word_count"
        ]
    
    def _format_results(self, results, use_distance: bool = False) -> List[Dict]:
        """Format search results"""
        formatted_results = []
        for hits in results:
            for hit in hits:
                score = float(hit.distance) if use_distance else float(hit.score)
                formatted_results.append({
                    "score": score,
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
    
    def search(self, query: str, top_k: int = 5, filter_expr: str = None, 
               method: Literal["vector", "sparse", "hybrid"] = "hybrid") -> List[Dict]:
        """
        Unified search method
        
        Args:
            query: Search query text
            top_k: Number of results
            filter_expr: Optional Milvus filter expression
            method: 'vector', 'sparse', or 'hybrid'
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
        
        # Execute hybrid search - removed expr from here
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=top_k,
            output_fields=self._get_output_fields()
        )
        
        return self._format_results(results, use_distance=True)
    
    def health_check(self) -> Dict:
        """Check Milvus health and hybrid search support"""
        try:
            collection = self.get_collection()
            
            schema = collection.schema
            has_sparse = any(f.name == "sparse_embedding" for f in schema.fields)
            has_dense = any(f.name == "dense_embedding" for f in schema.fields)
            
            return {
                "milvus_connected": True,
                "collection_exists": True,
                "collection_name": self.collection_name,
                "total_vectors": collection.num_entities,
                "embedding_model": self.embedding_model_name,
                "has_dense_field": has_dense,
                "has_sparse_field": has_sparse,
                "hybrid_enabled": has_dense and has_sparse
            }
        except Exception as e:
            return {
                "milvus_connected": False,
                "collection_exists": False,
                "total_vectors": 0,
                "hybrid_enabled": False,
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