"""
Script to sync local storage files to MongoDB
Scans all files in local storage and creates/updates MongoDB records
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from backend.services.local_storage_service import (
    get_storage_root,
    get_all_categories,
    list_files_in_category,
    get_file_info,
    compute_file_hash
)
from backend.services.mongodb_service import (
    insert_document,
    find_document,
    update_document,
    get_document_by_hash
)
from pymongo.errors import DuplicateKeyError
import mimetypes


def get_mime_type(file_path: Path) -> str:
    """Get MIME type from file extension"""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


def sync_local_storage_to_mongodb(force_update: bool = False):
    """
    Scan local storage and create MongoDB records for all files
    
    Args:
        force_update: If True, update existing records even if they exist
    """
    storage_root = get_storage_root()
    categories = get_all_categories()
    
    stats = {
        "files_scanned": 0,
        "files_inserted": 0,
        "files_updated": 0,
        "files_skipped": 0,
        "errors": []
    }
    
    print(f"\n{'='*70}")
    print(f"Syncing Local Storage to MongoDB")
    print(f"{'='*70}")
    print(f"📁 Storage root: {storage_root}")
    print(f"📂 Categories found: {len(categories)}")
    print(f"{'='*70}\n")
    
    for category in categories:
        print(f"\n📂 Processing category: {category}")
        print(f"   {'─'*60}")
        
        files = list_files_in_category(category)
        category_files = [f for f in files if f.is_file()]
        
        print(f"   Found {len(category_files)} files in category")
        
        for file_path in category_files:
            stats["files_scanned"] += 1
            
            try:
                # Get file info
                file_info = get_file_info(file_path)
                file_hash = file_info["hash"]
                
                # Check if document exists
                existing_doc = get_document_by_hash(file_hash)
                
                if existing_doc and not force_update:
                    print(f"   ⏭️  Skipped (exists): {file_path.name}")
                    stats["files_skipped"] += 1
                    continue
                
                # Prepare document data
                doc_data = {
                    "filename": file_path.name,
                    "file_hash": file_hash,
                    "file_size": file_info["size"],
                    "category": category,
                    "local_path": str(file_path),
                    "mime_type": get_mime_type(file_path),
                    "status": "synced",
                    "org_id": "",
                    "uploader_id": "",
                    "uploaded_at": datetime.utcnow(),
                    "metadata": {
                        "source": "local_storage_sync",
                        "file_modified_time": file_info["modified_time"],
                        "sync_timestamp": datetime.utcnow().isoformat()
                    }
                }
                
                if existing_doc:
                    # Update existing
                    update_document({"file_hash": file_hash}, doc_data)
                    print(f"   ✅ Updated: {file_path.name}")
                    stats["files_updated"] += 1
                else:
                    # Insert new
                    try:
                        insert_document(doc_data)
                        print(f"   ✅ Inserted: {file_path.name}")
                        stats["files_inserted"] += 1
                    except DuplicateKeyError:
                        print(f"   ⚠️  Duplicate detected: {file_path.name}")
                        stats["files_skipped"] += 1
                        
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                print(f"   ❌ {error_msg}")
                stats["errors"].append(error_msg)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✨ Sync Complete")
    print(f"{'='*70}")
    print(f"📊 Statistics:")
    print(f"   Files scanned:  {stats['files_scanned']}")
    print(f"   Files inserted: {stats['files_inserted']} ✅")
    print(f"   Files updated:  {stats['files_updated']} 🔄")
    print(f"   Files skipped:  {stats['files_skipped']} ⏭️")
    print(f"   Errors:         {len(stats['errors'])} ❌")
    print(f"{'='*70}")
    
    if stats['errors']:
        print(f"\n❌ Error Details:")
        for i, error in enumerate(stats['errors'][:10], 1):
            print(f"   {i}. {error}")
        if len(stats['errors']) > 10:
            print(f"   ... and {len(stats['errors']) - 10} more errors")
    
    print()
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sync local storage files to MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sync_local_to_mongo.py                  # Sync new files only
  python sync_local_to_mongo.py --force          # Update all files
        """
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force update existing records"
    )
    
    args = parser.parse_args()
    
    print("\n🚀 Starting MongoDB sync process...\n")
    sync_local_storage_to_mongodb(force_update=args.force)
    print("✅ Process completed!\n")