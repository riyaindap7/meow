# backend/services/local_storage_service.py

from pathlib import Path
from typing import Optional, Dict, List
import hashlib
import shutil
import os
from datetime import datetime
import json

# Default local storage root (can be overridden by environment variable)
DEFAULT_STORAGE_ROOT = Path("backend/data/local_storage")


def get_storage_root() -> Path:
    """Get the root directory for local storage"""
    storage_root = os.getenv("LOCAL_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
    root = Path(storage_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_category_path(category: str) -> Path:
    """
    Get local storage path for a category
    
    Categories: policies, circulars, budgets, notifications, updates, archives, etc.
    """
    category_path = get_storage_root() / category
    category_path.mkdir(parents=True, exist_ok=True)
    return category_path


def save_file(file_bytes: bytes, category: str, filename: str) -> Path:
    """
    Save file bytes to local storage
    
    Args:
        file_bytes: Raw file bytes
        category: Category folder (e.g., 'policies', 'circulars')
        filename: Original filename
        
    Returns:
        Path to saved file
    """
    category_path = get_category_path(category)
    file_path = category_path / filename
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    file_path.write_bytes(file_bytes)
    print(f"File saved to local storage: {file_path}")
    
    return file_path


def get_file_path(category: str, filename: str) -> Path:
    """Get local path for a file"""
    return get_category_path(category) / filename


def file_exists(category: str, filename: str) -> bool:
    """Check if file exists in local storage"""
    return get_file_path(category, filename).exists()


def delete_file(category: str, filename: str) -> bool:
    """Delete file from local storage"""
    file_path = get_file_path(category, filename)
    
    if file_path.exists():
        file_path.unlink()
        print(f"File deleted from local storage: {file_path}")
        return True
    
    return False


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file
    
    Used for detecting file changes and deduplication
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def compute_bytes_hash(file_bytes: bytes) -> str:
    """Compute SHA256 hash of bytes"""
    return hashlib.sha256(file_bytes).hexdigest()


def get_file_info(file_path: Path) -> Dict:
    """
    Get file metadata
    
    Returns:
        Dict with size, hash, modified_time
    """
    if not file_path.exists():
        return None
    
    stat = file_path.stat()
    
    return {
        "size": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime),
        "hash": compute_file_hash(file_path),
        "path": str(file_path)
    }


def list_files_in_category(category: str, pattern: str = "*") -> List[Path]:
    """
    List all files in a category folder
    
    Args:
        category: Category name
        pattern: Glob pattern (default: all files)
        
    Returns:
        List of Path objects
    """
    category_path = get_category_path(category)
    return list(category_path.glob(pattern))


def get_all_categories() -> List[str]:
    """Get list of all category folders in local storage"""
    storage_root = get_storage_root()
    return [d.name for d in storage_root.iterdir() if d.is_dir()]


def get_storage_stats() -> Dict:
    """
    Get statistics about local storage
    
    Returns:
        Dict with total_files, total_size_bytes, categories
    """
    storage_root = get_storage_root()
    
    stats = {
        "total_files": 0,
        "total_size_bytes": 0,
        "categories": {}
    }
    
    for category_dir in storage_root.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        category_files = list(category_dir.rglob("*"))
        category_files = [f for f in category_files if f.is_file()]
        
        category_size = sum(f.stat().st_size for f in category_files)
        
        stats["categories"][category_name] = {
            "file_count": len(category_files),
            "size_bytes": category_size
        }
        
        stats["total_files"] += len(category_files)
        stats["total_size_bytes"] += category_size
    
    return stats


def create_processed_folder(doc_id: str) -> Path:
    """
    Create a processed data folder for a document
    
    This stores parsed.json, images.json, and extracted images
    """
    processed_root = get_storage_root() / "processed" / doc_id
    processed_root.mkdir(parents=True, exist_ok=True)
    return processed_root


def save_processed_data(doc_id: str, parsed_data: Dict, images_data: List[Dict]) -> Dict[str, Path]:
    """
    Save processed document data (parsed text and image metadata)
    
    Returns:
        Dict with paths to parsed.json and images.json
    """
    processed_folder = create_processed_folder(doc_id)
    
    # Save parsed.json
    parsed_json_path = processed_folder / "parsed.json"
    parsed_json_path.write_text(
        json.dumps(parsed_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    # Save images.json
    images_json_path = processed_folder / "images.json"
    images_json_path.write_text(
        json.dumps(images_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    return {
        "parsed_json_path": parsed_json_path,
        "images_json_path": images_json_path,
        "folder_path": processed_folder
    }


def load_processed_data(doc_id: str) -> Optional[Dict]:
    """
    Load processed data for a document
    
    Returns:
        Dict with parsed_data and images_data, or None if not found
    """
    processed_folder = get_storage_root() / "processed" / doc_id
    
    if not processed_folder.exists():
        return None
    
    parsed_json_path = processed_folder / "parsed.json"
    images_json_path = processed_folder / "images.json"
    
    result = {}
    
    if parsed_json_path.exists():
        result["parsed_data"] = json.loads(parsed_json_path.read_text(encoding="utf-8"))
    
    if images_json_path.exists():
        result["images_data"] = json.loads(images_json_path.read_text(encoding="utf-8"))
    
    return result if result else None


def cleanup_old_files(category: str, days: int = 30) -> int:
    """
    Delete files older than specified days
    
    Returns:
        Number of files deleted
    """
    from datetime import timedelta
    
    cutoff_time = datetime.now() - timedelta(days=days)
    deleted_count = 0
    
    category_path = get_category_path(category)
    
    for file_path in category_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        if modified_time < cutoff_time:
            file_path.unlink()
            deleted_count += 1
    
    print(f"Deleted {deleted_count} files older than {days} days from {category}")
    return deleted_count
