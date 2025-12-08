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




from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class MilvusClient:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("COLLECTION_NAME", "Vtext")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
       
        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
       
        # Connect to Milvus
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
   
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=False
        )
        return embedding[0].tolist()
   
    def search(self, query: str, top_k: int = 5, filter_expr: str = None) -> List[Dict]:
        """
        Search for similar vectors in Vtext collection
       
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_expr: Optional Milvus filter expression
       
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
       
        # Output fields matching new Vtext schema
        output_fields = [
            "global_chunk_id",
            "document_id",
            "source_file",
            "page_idx",
            "chunk_index",
            "section_hierarchy",
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
                    "global_chunk_id": hit.entity.get("global_chunk_id"),
                    "document_id": hit.entity.get("document_id"),
                    "source_file": hit.entity.get("source_file"),
                    "page_idx": hit.entity.get("page_idx"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "section_hierarchy": hit.entity.get("section_hierarchy"),
                    "text": hit.entity.get("text"),
                    "char_count": hit.entity.get("char_count"),
                    "word_count": hit.entity.get("word_count")
                })
       
        return formatted_results
   
    def search_by_document(self, query: str, document_id: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific document"""
        filter_expr = f'document_id == "{document_id}"'
        return self.search(query, top_k, filter_expr)
   
    def search_by_page(self, query: str, document_id: str, page_idx: int, top_k: int = 3) -> List[Dict]:
        """Search within a specific page of a document"""
        filter_expr = f'document_id == "{document_id}" && page_idx == {page_idx}'
        return self.search(query, top_k, filter_expr)
   
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
                "embedding_model": self.embedding_model_name
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
    print(client.health_check())
   
    # List documents
    print("\n📚 Available Documents:")
    docs = client.list_documents()
    print(f"Found {len(docs)} documents")
    for doc in docs[:10]:
        print(f"  - {doc}")
   
    # Search example
    print("\n🔍 Search Example:")
    query = "What is RUSA?"
    results = client.search(query, top_k=3)
   
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['score']:.4f}) ---")
        print(f"Document: {result['document_id']}")
        print(f"Source File: {result['source_file']}")
        print(f"Page: {result['page_idx']}")
        print(f"Section: {result['section_hierarchy'][:100]}")
        print(f"Text: {result['text'][:200]}...")