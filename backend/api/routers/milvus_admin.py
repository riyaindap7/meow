"""
Milvus Admin Router - Collection & Data Management
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pymilvus import connections, utility, Collection
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Milvus connection details
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.65.80")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")


def ensure_connection():
    """Ensure Milvus connection is active"""
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        print(f"Milvus connection: {e}")


# ==================== MODELS ====================

class CollectionInfo(BaseModel):
    name: str
    description: str
    num_entities: int
    schema: Dict[str, Any]
    indexes: List[Dict[str, Any]]


class CollectionStats(BaseModel):
    name: str
    row_count: int
    index_info: List[Dict[str, Any]]
    memory_size: Optional[int] = None


class QueryRequest(BaseModel):
    collection_name: str
    expr: Optional[str] = None
    output_fields: List[str] = ["*"]
    limit: int = 100


# ==================== ENDPOINTS ====================

@router.get("/collections", response_model=List[str])
async def list_collections():
    """List all Milvus collections"""
    try:
        ensure_connection()
        all_collections = utility.list_collections()
        
        # Filter out collections that don't actually exist or can't be loaded
        valid_collections = []
        for col_name in all_collections:
            try:
                if utility.has_collection(col_name):
                    # Try to actually access the collection to verify it exists
                    col = Collection(col_name)
                    _ = col.num_entities  # This will fail if collection is invalid
                    valid_collections.append(col_name)
            except Exception as e:
                print(f"⚠️ Skipping invalid collection '{col_name}': {str(e)}")
                continue
        
        return valid_collections
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@router.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    """Get detailed information about a collection"""
    try:
        ensure_connection()
        
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        collection = Collection(collection_name)
        collection.load()
        
        # Get schema
        schema = collection.schema
        schema_dict = {
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.dtype),
                    "is_primary": field.is_primary,
                    "auto_id": field.auto_id if hasattr(field, 'auto_id') else False,
                    "dim": field.params.get("dim") if hasattr(field, 'params') and field.params else None,
                    "max_length": field.params.get("max_length") if hasattr(field, 'params') and field.params else None
                }
                for field in schema.fields
            ],
            "description": schema.description
        }
        
        # Get indexes
        indexes = []
        try:
            for field in schema.fields:
                if field.dtype in [100, 101]:  # Vector types
                    index_info = collection.index(field.name)
                    if index_info:
                        indexes.append({
                            "field": field.name,
                            "index_type": index_info.params.get("index_type"),
                            "metric_type": index_info.params.get("metric_type"),
                            "params": index_info.params
                        })
        except:
            pass
        
        return {
            "name": collection_name,
            "description": schema.description,
            "num_entities": collection.num_entities,
            "schema": schema_dict,
            "indexes": indexes
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


@router.get("/collections/{collection_name}/stats")
async def get_collection_stats(collection_name: str):
    """Get collection statistics"""
    try:
        ensure_connection()
        
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        collection = Collection(collection_name)
        collection.load()
        
        stats = {
            "name": collection_name,
            "row_count": collection.num_entities,
            "index_info": [],
            "partitions": utility.list_partitions(collection_name)
        }
        
        # Try to get index information
        try:
            schema = collection.schema
            for field in schema.fields:
                if field.dtype in [100, 101]:  # Vector types
                    index_info = collection.index(field.name)
                    if index_info:
                        stats["index_info"].append({
                            "field": field.name,
                            "index_type": index_info.params.get("index_type"),
                            "metric_type": index_info.params.get("metric_type")
                        })
        except:
            pass
        
        return stats
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/collections/{collection_name}/query")
async def query_collection(collection_name: str, request: QueryRequest):
    """Query data from a collection"""
    try:
        ensure_connection()
        
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        collection = Collection(collection_name)
        collection.load()
        
        # Get schema to determine output fields if "*" is requested
        schema = collection.schema
        all_field_names = [field.name for field in schema.fields]
        
        # Determine which fields to output
        output_fields = request.output_fields
        if "*" in output_fields:
            # Get all non-vector fields by default, but we'll include vectors separately
            output_fields = [f.name for f in schema.fields if f.dtype not in [100, 101]]
        
        # Always include vector fields explicitly
        vector_fields = [f.name for f in schema.fields if f.dtype in [100, 101]]
        output_fields_with_vectors = list(set(output_fields + vector_fields))
        
        # Query
        results = collection.query(
            expr=request.expr or "",
            output_fields=output_fields_with_vectors,
            limit=request.limit
        )
        
        # Process results to make embeddings more readable
        processed_results = []
        for result in results[:100]:  # Limit to 100 for display
            processed_result = {}
            for key, value in result.items():
                # Check if this is an embedding field
                if key in vector_fields and isinstance(value, (list, tuple)):
                    # Store embedding info
                    processed_result[key] = {
                        "type": "embedding",
                        "dimension": len(value),
                        "preview": value[:5] if len(value) > 5 else value,  # First 5 values
                        "full": value  # Keep full embedding
                    }
                else:
                    processed_result[key] = value
            processed_results.append(processed_result)
        
        return {
            "collection": collection_name,
            "total_results": len(results),
            "results": processed_results,
            "vector_fields": vector_fields,
            "schema_fields": all_field_names
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection (use with caution!)"""
    try:
        ensure_connection()
        
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        utility.drop_collection(collection_name)
        
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")


@router.get("/server/status")
async def get_server_status():
    """Get Milvus server status"""
    try:
        ensure_connection()
        
        collections = utility.list_collections()
        
        total_entities = 0
        for coll_name in collections:
            try:
                coll = Collection(coll_name)
                total_entities += coll.num_entities
            except:
                pass
        
        return {
            "status": "connected",
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
            "collections_count": len(collections),
            "total_entities": total_entities
        }
    
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e)
        }