"""
File serving endpoints for team access
Serves files from local storage via HTTP
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from typing import Optional, List
import mimetypes
import zipfile
import io
from datetime import datetime

from backend.services.local_storage_service import (
    get_file_path,
    get_file_info,
    list_files_in_category,
    get_all_categories
)
from backend.services.mongodb_service import (
    get_document_by_hash,
    find_documents
)

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/categories")
async def get_categories():
    """Get all available categories"""
    categories = get_all_categories()
    return {
        "categories": categories,
        "count": len(categories)
    }


@router.get("/category/{category}")
async def list_category_files(category: str):
    """List all files in a category"""
    try:
        files = list_files_in_category(category)
        
        file_list = []
        for file_path in files:
            if file_path.is_file():
                info = get_file_info(file_path)
                file_list.append({
                    "filename": file_path.name,
                    "size": info["size"],
                    "hash": info["hash"],
                    "modified_time": info["modified_time"],
                    "category": category,
                    "download_url": f"/api/files/download/{category}/{file_path.name}"
                })
        
        return {
            "category": category,
            "files": file_list,
            "count": len(file_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Category not found: {str(e)}")


@router.get("/download/{category}/{filename}")
async def download_file(category: str, filename: str):
    """
    Download file from local storage
    Team members can access this endpoint to get files
    """
    try:
        file_path = get_file_path(category, filename)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"
        
        return FileResponse(
            path=str(file_path),
            media_type=mime_type,
            filename=filename,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@router.get("/view/{category}/{filename}")
async def view_file(category: str, filename: str):
    """
    View/preview file in browser (for PDFs, images, etc.)
    Opens in browser instead of downloading
    """
    try:
        file_path = get_file_path(category, filename)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"
        
        return FileResponse(
            path=str(file_path),
            media_type=mime_type,
            filename=filename,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error viewing file: {str(e)}")


@router.get("/stream/{category}/{filename}")
async def stream_file(category: str, filename: str):
    """
    Stream large files in chunks
    Better for large PDFs or videos
    """
    try:
        file_path = get_file_path(category, filename)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        def iterfile():
            with open(file_path, mode="rb") as file:
                while chunk := file.read(1024 * 1024):  # 1MB chunks
                    yield chunk
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"
        
        return StreamingResponse(
            iterfile(),
            media_type=mime_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming file: {str(e)}")


@router.get("/info/{category}/{filename}")
async def get_file_metadata(category: str, filename: str):
    """
    Get file metadata without downloading
    """
    try:
        file_path = get_file_path(category, filename)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        info = get_file_info(file_path)
        
        # Get from MongoDB if exists
        mongo_doc = get_document_by_hash(info["hash"])
        
        return {
            "filename": filename,
            "category": category,
            "size": info["size"],
            "hash": info["hash"],
            "modified_time": info["modified_time"],
            "download_url": f"/api/files/download/{category}/{filename}",
            "view_url": f"/api/files/view/{category}/{filename}",
            "mongodb_metadata": mongo_doc if mongo_doc else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting file info: {str(e)}")


@router.get("/search")
async def search_files(
    query: Optional[str] = None,
    category: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None
):
    """
    Search files across all categories or specific category
    """
    try:
        categories = [category] if category else get_all_categories()
        results = []
        
        for cat in categories:
            files = list_files_in_category(cat)
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                # Apply filters
                if query and query.lower() not in file_path.name.lower():
                    continue
                
                info = get_file_info(file_path)
                
                if min_size and info["size"] < min_size:
                    continue
                if max_size and info["size"] > max_size:
                    continue
                
                results.append({
                    "filename": file_path.name,
                    "category": cat,
                    "size": info["size"],
                    "hash": info["hash"],
                    "modified_time": info["modified_time"],
                    "download_url": f"/api/files/download/{cat}/{file_path.name}",
                    "view_url": f"/api/files/view/{cat}/{file_path.name}"
                })
        
        return {
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching files: {str(e)}")


@router.get("/bulk-download")
async def bulk_download_all_pdfs(
    categories: Optional[str] = None,  # Comma-separated categories
    format: str = "zip"  # Future: support tar, etc.
):
    """
    Bulk download all PDFs as a ZIP file
    
    Args:
        categories: Optional comma-separated list of categories (e.g., "research,reports")
                   If not provided, downloads from all categories
        format: Archive format (currently only 'zip' supported)
    
    Returns:
        ZIP file containing all PDFs organized by category
    
    Example:
        /api/files/bulk-download
        /api/files/bulk-download?categories=research,reports
    """
    try:
        # Determine which categories to include
        if categories:
            category_list = [c.strip() for c in categories.split(",")]
        else:
            category_list = get_all_categories()
        
        # Create in-memory ZIP file
        zip_buffer = io.BytesIO()
        
        total_files = 0
        total_size = 0
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for category in category_list:
                try:
                    files = list_files_in_category(category, pattern="*.pdf")
                    
                    for file_path in files:
                        if not file_path.is_file():
                            continue
                        
                        # Add file to ZIP with category folder structure
                        # data/local_storage/category/filename.pdf
                        arcname = f"data/local_storage/{category}/{file_path.name}"
                        
                        zip_file.write(file_path, arcname=arcname)
                        
                        total_files += 1
                        total_size += file_path.stat().st_size
                        
                except Exception as e:
                    print(f"Warning: Error processing category {category}: {e}")
                    continue
        
        # Prepare ZIP for download
        zip_buffer.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pdfs_bulk_download_{timestamp}.zip"
        
        # Add metadata as comment in ZIP
        metadata = f"Bulk download - {total_files} files, {total_size / 1024 / 1024:.2f} MB"
        
        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Total-Files": str(total_files),
                "X-Total-Size": str(total_size),
                "X-Categories": ",".join(category_list)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating bulk download: {str(e)}"
        )


@router.get("/bulk-download-by-category/{category}")
async def bulk_download_category(category: str):
    """
    Download all PDFs from a specific category as ZIP
    
    Example:
        /api/files/bulk-download-by-category/research
    """
    try:
        files = list_files_in_category(category, pattern="*.pdf")
        
        if not files:
            raise HTTPException(
                status_code=404,
                detail=f"No PDF files found in category: {category}"
            )
        
        # Create ZIP
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                # Structure: data/local_storage/category/filename.pdf
                arcname = f"data/local_storage/{category}/{file_path.name}"
                zip_file.write(file_path, arcname=arcname)
        
        zip_buffer.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{category}_pdfs_{timestamp}.zip"
        
        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading category: {str(e)}"
        )


@router.post("/bulk-download-selected")
async def bulk_download_selected_files(file_list: List[dict]):
    """
    Download specific files as ZIP
    
    Request body:
    [
        {"category": "research", "filename": "paper1.pdf"},
        {"category": "reports", "filename": "report2.pdf"}
    ]
    """
    try:
        if not file_list:
            raise HTTPException(status_code=400, detail="File list is empty")
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for item in file_list:
                category = item.get("category")
                filename = item.get("filename")
                
                if not category or not filename:
                    continue
                
                try:
                    file_path = get_file_path(category, filename)
                    
                    if file_path.exists() and file_path.is_file():
                        arcname = f"data/local_storage/{category}/{filename}"
                        zip_file.write(file_path, arcname=arcname)
                        
                except Exception as e:
                    print(f"Warning: Could not add {category}/{filename}: {e}")
                    continue
        
        zip_buffer.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selected_pdfs_{timestamp}.zip"
        
        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading selected files: {str(e)}"
        )


@router.get("/bulk-download-info")
async def get_bulk_download_info(categories: Optional[str] = None):
    """
    Get information about bulk download without downloading
    Shows how many files and total size
    """
    try:
        if categories:
            category_list = [c.strip() for c in categories.split(",")]
        else:
            category_list = get_all_categories()
        
        info = {
            "categories": [],
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0
        }
        
        for category in category_list:
            files = list_files_in_category(category, pattern="*.pdf")
            
            category_size = 0
            file_count = 0
            
            for file_path in files:
                if file_path.is_file():
                    file_count += 1
                    category_size += file_path.stat().st_size
            
            info["categories"].append({
                "name": category,
                "file_count": file_count,
                "size_bytes": category_size,
                "size_mb": round(category_size / 1024 / 1024, 2)
            })
            
            info["total_files"] += file_count
            info["total_size_bytes"] += category_size
        
        info["total_size_mb"] = round(info["total_size_bytes"] / 1024 / 1024, 2)
        info["estimated_zip_size_mb"] = round(info["total_size_mb"] * 0.95, 2)  # PDFs don't compress much
        
        return info
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting download info: {str(e)}"
        )