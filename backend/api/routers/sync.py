# backend/api/routers/sync.py

"""
Sync & Document Management API Router

This router provides endpoints for:
- Triggering Google Drive sync
- Checking sync status
- Managing document processing
- Retrieving sync statistics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from backend.services.sync_service import create_sync_service
# from backend.services.document_processor import create_document_processor
from backend.services.mongodb_service import (
    get_stats,
    get_recent_sync_logs,
    find_documents,
    find_document,
    update_document,
    get_document_by_drive_id
)
from backend.services.local_storage_service import get_storage_stats

router = APIRouter(prefix="/sync", tags=["sync"])


class SyncRequest(BaseModel):
    """Request to trigger sync"""
    force: bool = False
    category: Optional[str] = None


class ProcessRequest(BaseModel):
    """Request to process documents"""
    category: Optional[str] = None
    limit: int = 100


class ProcessDocumentRequest(BaseModel):
    """Request to process a single document"""
    document_id: Optional[str] = None
    google_drive_id: Optional[str] = None


@router.post("/trigger")
async def trigger_sync(background_tasks: BackgroundTasks, request: SyncRequest = SyncRequest()):
    """
    Trigger Google Drive to local storage sync
    
    This can be run in the background to avoid blocking the request.
    
    Request body (all optional):
    {
        "force": false,     // Re-download all files
        "category": null    // Sync specific category only
    }
    """
    try:
        sync_service = create_sync_service()
        
        # Run sync in background
        def run_sync():
            if request.category:
                return sync_service.sync_category(request.category, force=request.force)
            else:
                return sync_service.sync_all(force=request.force)
        
        # Add to background tasks
        background_tasks.add_task(run_sync)
        
        return {
            "status": "started",
            "message": "Sync started in background",
            "force": request.force,
            "category": request.category
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start sync: {str(e)}")


@router.get("/status")
async def get_sync_status():
    """
    Get current sync status and statistics
    """
    try:
        # Get MongoDB stats
        db_stats = get_stats()
        
        # Get storage stats
        storage_stats = get_storage_stats()
        
        # Get recent sync logs
        recent_syncs = get_recent_sync_logs(limit=5)
        
        return {
            "database": {
                "total_documents": db_stats["total_documents"],
                "documents_by_status": db_stats["documents_by_status"],
                "total_collections": db_stats["total_collections"],
                "last_sync_time": db_stats["last_sync_time"]
            },
            "storage": {
                "total_files": storage_stats["total_files"],
                "total_size_mb": storage_stats["total_size_bytes"] / 1024 / 1024,
                "categories": storage_stats["categories"]
            },
            "recent_syncs": recent_syncs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sync status: {str(e)}")


@router.get("/logs")
async def get_sync_logs(limit: int = 10):
    """
    Get recent sync logs
    """
    try:
        logs = get_recent_sync_logs(limit=limit)
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sync logs: {str(e)}")


@router.post("/process-documents")
async def process_documents(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process synced PDF documents (extract text and images)
    
    Runs in background to avoid blocking.
    """
    try:
        processor = create_document_processor()
        
        # Run processing in background
        def run_processing():
            return processor.batch_process_documents(
                category=request.category,
                status_filter="synced",
                limit=request.limit
            )
        
        background_tasks.add_task(run_processing)
        
        return {
            "status": "started",
            "message": "Document processing started in background",
            "category": request.category,
            "limit": request.limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@router.post("/process-document")
async def process_single_document(request: ProcessDocumentRequest):
    """
    Process a single document by ID or Google Drive ID
    """
    try:
        processor = create_document_processor()
        
        # Find document
        doc = None
        if request.document_id:
            doc = find_document({"_id": request.document_id})
        elif request.google_drive_id:
            doc = get_document_by_drive_id(request.google_drive_id)
        else:
            raise HTTPException(status_code=400, detail="Must provide document_id or google_drive_id")
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Process document
        result = processor.process_document_from_db(str(doc["_id"]))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents")
async def get_documents(
    category: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    skip: int = 0
):
    """
    Get documents with optional filters
    """
    try:
        query = {}
        
        if category:
            query["category"] = category
        
        if status:
            query["status"] = status
        
        documents = find_documents(query, limit=limit, skip=skip)
        
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            doc["_id"] = str(doc["_id"])
        
        return {
            "documents": documents,
            "count": len(documents),
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get a single document by ID
    """
    try:
        doc = find_document({"_id": document_id}) or get_document_by_drive_id(document_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Convert ObjectId to string
        doc["_id"] = str(doc["_id"])
        
        return doc
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.patch("/documents/{document_id}")
async def update_document_metadata(document_id: str, metadata: Dict[str, Any]):
    """
    Update document metadata
    """
    try:
        # Find document
        doc = find_document({"_id": document_id})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update document
        success = update_document({"_id": document_id}, metadata)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update document")
        
        return {"status": "updated", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")


@router.get("/categories")
async def get_categories():
    """
    Get list of all categories with document counts
    """
    try:
        storage_stats = get_storage_stats()
        
        categories = []
        for category_name, info in storage_stats["categories"].items():
            categories.append({
                "name": category_name,
                "file_count": info["file_count"],
                "size_mb": info["size_bytes"] / 1024 / 1024
            })
        
        return {"categories": categories}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")
