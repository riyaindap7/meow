# from fastapi import APIRouter, UploadFile, File
# from pydantic import BaseModel

# router = APIRouter()

# class DocumentMetadata(BaseModel):
#     filename: str
#     size: int
#     mime_type: str

# # In-memory documents storage for demo
# documents_db = []

# @router.post("/upload")
# async def upload_document(file: UploadFile = File(...)):
#     """Upload a document"""
#     contents = await file.read()
    
#     doc = {
#         "id": len(documents_db) + 1,
#         "filename": file.filename,
#         "size": len(contents),
#         "mime_type": file.content_type
#     }
    
#     documents_db.append(doc)
    
#     return {
#         "status": "success",
#         "document_id": doc["id"],
#         "filename": file.filename,
#         "size": len(contents)
#     }

# @router.get("/")
# async def list_documents():
#     """List all documents"""
#     return {"documents": documents_db, "total": len(documents_db)}




# @router.post("/upload-and-parse")
# async def upload_and_parse(file: UploadFile = File(...)):
#     # Step 1: Read uploaded file
#     file_bytes = await file.read()
    
#     # Step 2: Save to local storage
#     from backend.services.local_storage_service import save_file, compute_bytes_hash
#     from backend.services.mongodb_service import insert_document
    
#     # Determine category (default to 'uploads')
#     category = "uploads"
#     local_path = save_file(file_bytes, category, file.filename)
#     file_hash = compute_bytes_hash(file_bytes)
    
#     # Save metadata to MongoDB
#     doc_metadata = {
#         "filename": file.filename,
#         "file_hash": file_hash,
#         "file_size": len(file_bytes),
#         "category": category,
#         "local_path": str(local_path),
#         "mime_type": file.content_type,
#         "status": "uploaded"
#     }
#     insert_document(doc_metadata)
    
#     return {
#         "local_path": str(local_path),
#         "filename": file.filename,
#         "size": len(file_bytes),
#         "hash": file_hash
#     }



from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool

from backend.services.local_storage_service import save_file, compute_bytes_hash
from backend.services.mongodb_service import insert_document
from backend.services.pdf_parser_service import parse_single_pdf  # <- new import

router = APIRouter()


class DocumentMetadata(BaseModel):
    filename: str
    size: int
    mime_type: str


# In-memory documents storage for demo
documents_db = []


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Simple upload without parsing"""
    contents = await file.read()

    doc = {
        "id": len(documents_db) + 1,
        "filename": file.filename,
        "size": len(contents),
        "mime_type": file.content_type,
    }

    documents_db.append(doc)

    return {
        "status": "success",
        "document_id": doc["id"],
        "filename": file.filename,
        "size": len(contents),
    }


@router.get("/")
async def list_documents():
    """List all documents"""
    return {"documents": documents_db, "total": len(documents_db)}


@router.post("/upload-and-parse")
async def upload_and_parse(file: UploadFile = File(...)):
    """
    Upload a single document from the UI, save it locally, and run MinerU parsing
    using the same logic as the batch script but for one file.
    """
    # Step 1: Read uploaded file
    file_bytes = await file.read()

    # Step 2: Save to local storage
    category = "uploads"  # or infer based on user / path / etc.
    local_path = save_file(file_bytes, category, file.filename)
    file_hash = compute_bytes_hash(file_bytes)

    # Step 3: Save metadata to MongoDB
    doc_metadata = {
        "filename": file.filename,
        "file_hash": file_hash,
        "file_size": len(file_bytes),
        "category": category,
        "local_path": str(local_path),
        "mime_type": file.content_type,
        "status": "uploaded",
    }
    insert_document(doc_metadata)

    # Step 4: Parse the single PDF using MinerU
    # Run the blocking subprocess in a thread so we don't block the event loop
    parse_result = await run_in_threadpool(
        parse_single_pdf,
        str(local_path),
    )

    # (Optional) you could update Mongo here with parse status / output_dir

    # Step 5: Return combined info
    return {
        "upload": {
            "local_path": str(local_path),
            "filename": file.filename,
            "size": len(file_bytes),
            "hash": file_hash,
        },
        "parse": parse_result,
    }
