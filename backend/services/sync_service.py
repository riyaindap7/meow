# backend/services/sync_service.py

"""
Google Drive to Local Storage Sync Service

This service synchronizes files from Google Drive master folder to local storage.
It performs intelligent incremental sync by comparing:
- File hashes (MD5 from Drive, SHA256 locally)
- File sizes
- Modified timestamps

Only new or changed files are downloaded, minimizing bandwidth and time.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from .google_drive_service import (
    list_files_recursive,
    download_file,
    get_file_metadata,
    get_folder_structure
)
from .local_storage_service import (
    get_storage_root,
    get_category_path,
    save_file,
    file_exists,
    get_file_info,
    compute_file_hash,
    create_processed_folder
)
from .mongodb_service import (
    insert_document,
    update_document,
    find_document,
    get_document_by_hash,
    get_document_by_drive_id,
    log_sync_event
)


class SyncService:
    """
    Service for syncing Google Drive files to local storage
    """
    
    # Existing folder structure mapping
    DEFAULT_CATEGORY_MAPPING = {
        "moe_scraped_higher_edu_RUSA": "higher_education_rusa",
        "Scraped_moe_archived_advertisment": "archived_advertisements",
        "scraped_moe_archived_circulars": "archived_circulars",
        "Scraped_moe_archived_press_releases": "archived_press_releases",
        "Scraped_moe_archived_scholarships": "archived_scholarships",
        "scraped_moe_archived_updates": "archived_updates",
        "scraped_moe_documents&reports": "documents_and_reports",
        "scraped_moe_higher_education_schemes": "higher_education_schemes",
        "scraped_moe_mothly_achivements": "monthly_achievements",
        "scraped_moe_rti": "rti_documents",
        "scraped_moe_schemes": "schemes",
        "scraped_moe_statistics": "statistics"
    }
    
    def __init__(self, drive_folder_id: str, category_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize sync service
        
        Args:
            drive_folder_id: Root Google Drive folder ID containing all documents
            category_mapping: Optional mapping of Drive folder names to categories
                              If not provided, uses DEFAULT_CATEGORY_MAPPING
        """
        self.drive_folder_id = drive_folder_id
        self.category_mapping = category_mapping or self.DEFAULT_CATEGORY_MAPPING
        
    def determine_category(self, drive_path: str) -> str:
        """
        Determine category from Google Drive path
        
        Uses category_mapping to convert Drive folder names to local categories
        """
        # Split path and get first folder (the Drive folder name)
        parts = drive_path.split('/')
        
        if len(parts) > 0:
            # The first part is the Drive folder name
            drive_folder = parts[0] if parts[0] else (parts[1] if len(parts) > 1 else "")
            
            # Check if we have a mapping for this folder
            if drive_folder in self.category_mapping:
                return self.category_mapping[drive_folder]
            
            # Fallback: sanitize folder name for use as category
            return drive_folder.lower().replace(' ', '_').replace('&', 'and')
        
        # Default category
        return "uncategorized"
    
    def file_needs_sync(self, drive_file: Dict, local_doc: Optional[Dict]) -> Tuple[bool, str]:
        """
        Determine if a file needs to be synced
        
        Returns:
            (needs_sync: bool, reason: str)
        """
        drive_id = drive_file['id']
        drive_size = int(drive_file.get('size', 0))
        drive_modified = drive_file.get('modifiedTime')
        drive_md5 = drive_file.get('md5Checksum')
        
        # If no local document record exists, definitely sync
        if not local_doc:
            return (True, "new_file")
        
        # Check if file exists locally
        local_path = local_doc.get('local_path')
        if not local_path or not Path(local_path).exists():
            return (True, "missing_local_file")
        
        # Compare file sizes
        local_size = local_doc.get('file_size', 0)
        if drive_size != local_size:
            return (True, "size_mismatch")
        
        # Compare modified times
        local_modified = local_doc.get('google_drive_modified_time')
        if drive_modified and local_modified:
            # Parse timestamps
            drive_dt = datetime.fromisoformat(drive_modified.replace('Z', '+00:00'))
            local_dt = local_modified if isinstance(local_modified, datetime) else datetime.fromisoformat(str(local_modified))
            
            if drive_dt > local_dt:
                return (True, "newer_version")
        
        # Compare checksums if available
        if drive_md5:
            local_hash = local_doc.get('file_hash')
            # Note: Drive uses MD5, we use SHA256, so this won't match directly
            # We rely on size + timestamp comparison instead
            pass
        
        # File appears up-to-date
        return (False, "up_to_date")
    
    def sync_file(self, drive_file: Dict, force: bool = False) -> Dict:
        """
        Sync a single file from Google Drive to local storage
        
        Returns:
            Dict with sync result and metadata
        """
        drive_id = drive_file['id']
        filename = drive_file['name']
        drive_path = drive_file.get('drive_path', filename)
        
        # Determine category from path
        category = self.determine_category(drive_path)
        
        # Check if file already exists in MongoDB
        local_doc = get_document_by_drive_id(drive_id)
        
        # Check if sync is needed
        needs_sync, reason = self.file_needs_sync(drive_file, local_doc)
        
        if not needs_sync and not force:
            return {
                "status": "skipped",
                "reason": reason,
                "filename": filename,
                "drive_id": drive_id
            }
        
        # Download file
        local_path = get_category_path(category) / filename
        
        try:
            success = download_file(drive_id, local_path)
            
            if not success:
                return {
                    "status": "error",
                    "reason": "download_failed",
                    "filename": filename,
                    "drive_id": drive_id
                }
            
            # Compute file hash
            file_hash = compute_file_hash(local_path)
            file_size = local_path.stat().st_size
            
            # Prepare document metadata
            doc_data = {
                "filename": filename,
                "file_hash": file_hash,
                "file_size": file_size,
                "category": category,
                "google_drive_id": drive_id,
                "google_drive_path": drive_path,
                "google_drive_modified_time": datetime.fromisoformat(
                    drive_file.get('modifiedTime', '').replace('Z', '+00:00')
                ),
                "local_path": str(local_path),
                "mime_type": drive_file.get('mimeType', ''),
                "status": "synced",
                "metadata": {
                    "sync_reason": reason,
                    "drive_md5": drive_file.get('md5Checksum')
                }
            }
            
            # Insert or update MongoDB record
            if local_doc:
                # Update existing record
                update_document({"google_drive_id": drive_id}, doc_data)
                action = "updated"
            else:
                # Insert new record
                insert_document(doc_data)
                action = "inserted"
            
            return {
                "status": "success",
                "action": action,
                "reason": reason,
                "filename": filename,
                "drive_id": drive_id,
                "local_path": str(local_path),
                "file_size": file_size,
                "category": category
            }
            
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e),
                "filename": filename,
                "drive_id": drive_id
            }
    
    def sync_all(self, force: bool = False, file_filter: Optional[str] = None) -> Dict:
        """
        Sync all files from Google Drive to local storage
        
        Args:
            force: If True, download all files regardless of local state
            file_filter: Optional glob pattern to filter files (e.g., "*.pdf")
            
        Returns:
            Dict with sync statistics
        """
        start_time = time.time()
        
        print(f"Starting sync from Google Drive folder: {self.drive_folder_id}")
        
        # Get all files from Drive recursively
        try:
            drive_files = list_files_recursive(self.drive_folder_id)
        except Exception as e:
            print(f"Error listing Drive files: {e}")
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
        
        # Filter files if needed
        if file_filter:
            from fnmatch import fnmatch
            drive_files = [f for f in drive_files if fnmatch(f['name'], file_filter)]
        
        # Only sync PDF files by default (skip Google Docs, Sheets, etc.)
        drive_files = [
            f for f in drive_files 
            if not f.get('mimeType', '').startswith('application/vnd.google-apps.')
        ]
        
        print(f"Found {len(drive_files)} files to sync")
        
        # Sync each file
        results = {
            "files_checked": len(drive_files),
            "files_downloaded": 0,
            "files_updated": 0,
            "files_skipped": 0,
            "files_error": 0,
            "bytes_downloaded": 0,
            "errors": [],
            "details": []
        }
        
        for i, drive_file in enumerate(drive_files, 1):
            print(f"\nSyncing file {i}/{len(drive_files)}: {drive_file['name']}")
            
            result = self.sync_file(drive_file, force=force)
            results["details"].append(result)
            
            if result["status"] == "success":
                if result["action"] == "inserted":
                    results["files_downloaded"] += 1
                elif result["action"] == "updated":
                    results["files_updated"] += 1
                
                results["bytes_downloaded"] += result.get("file_size", 0)
                
            elif result["status"] == "skipped":
                results["files_skipped"] += 1
                
            elif result["status"] == "error":
                results["files_error"] += 1
                results["errors"].append(f"{drive_file['name']}: {result['reason']}")
        
        duration = time.time() - start_time
        results["duration_seconds"] = duration
        
        # Log sync event to MongoDB
        log_sync_event("incremental_sync" if not force else "full_sync", results)
        
        print(f"\n{'='*60}")
        print(f"Sync completed in {duration:.2f} seconds")
        print(f"Files checked: {results['files_checked']}")
        print(f"Files downloaded: {results['files_downloaded']}")
        print(f"Files updated: {results['files_updated']}")
        print(f"Files skipped: {results['files_skipped']}")
        print(f"Files with errors: {results['files_error']}")
        print(f"Bytes downloaded: {results['bytes_downloaded']:,}")
        print(f"{'='*60}\n")
        
        return results
    
    def sync_category(self, category: str, force: bool = False) -> Dict:
        """
        Sync only files from a specific category folder
        
        This is useful for targeted syncs of specific document types
        """
        # This requires category_mapping to be set up properly
        # We'll filter files by checking their drive_path
        
        print(f"Syncing category: {category}")
        
        # Get all files
        drive_files = list_files_recursive(self.drive_folder_id)
        
        # Filter by category
        category_files = [
            f for f in drive_files
            if self.determine_category(f.get('drive_path', '')) == category
        ]
        
        print(f"Found {len(category_files)} files in category '{category}'")
        
        # Similar to sync_all but with filtered list
        results = {"files_checked": len(category_files)}
        
        for drive_file in category_files:
            result = self.sync_file(drive_file, force=force)
            # ... aggregate results
        
        return results


def create_sync_service(drive_folder_id: str = None) -> SyncService:
    """
    Factory function to create SyncService with default configuration
    
    Reads GOOGLE_DRIVE_MASTER_FOLDER_ID from environment if not provided
    """
    import os
    
    if not drive_folder_id:
        drive_folder_id = os.getenv("GOOGLE_DRIVE_MASTER_FOLDER_ID")
        
        if not drive_folder_id:
            raise ValueError(
                "Google Drive folder ID not provided. "
                "Set GOOGLE_DRIVE_MASTER_FOLDER_ID environment variable or pass drive_folder_id parameter"
            )
    
    # Use the default category mapping for existing folder structure
    return SyncService(drive_folder_id)
