from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

router = APIRouter()

class DocumentMetadata(BaseModel):
    filename: str
    size: int
    mime_type: str

# In-memory documents storage for demo
documents_db = []

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document"""
    contents = await file.read()
    
    doc = {
        "id": len(documents_db) + 1,
        "filename": file.filename,
        "size": len(contents),
        "mime_type": file.content_type
    }
    
    documents_db.append(doc)
    
    return {
        "status": "success",
        "document_id": doc["id"],
        "filename": file.filename,
        "size": len(contents)
    }

@router.get("/")
async def list_documents():
    """List all documents"""
    return {"documents": documents_db, "total": len(documents_db)}




@router.post("/upload-and-parse")
async def upload_and_parse(file: UploadFile = File(...)):
    # Step 1: Read uploaded file
    file_bytes = await file.read()
    
    # Step 2: Save to local storage
    from backend.services.local_storage_service import save_file, compute_bytes_hash
    from backend.services.mongodb_service import insert_document
    
    # Determine category (default to 'uploads')
    category = "uploads"
    local_path = save_file(file_bytes, category, file.filename)
    file_hash = compute_bytes_hash(file_bytes)
    
    # Save metadata to MongoDB
    doc_metadata = {
        "filename": file.filename,
        "file_hash": file_hash,
        "file_size": len(file_bytes),
        "category": category,
        "local_path": str(local_path),
        "mime_type": file.content_type,
        "status": "uploaded"
    }
    insert_document(doc_metadata)
    
    return {
        "local_path": str(local_path),
        "filename": file.filename,
        "size": len(file_bytes),
        "hash": file_hash
    }
