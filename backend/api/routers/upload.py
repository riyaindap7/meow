# backend/api/routers/upload.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
import tempfile
import asyncio
import pdfplumber
import fitz
import traceback
import uuid
import json
import zipfile
import os
import shutil
import re
import mimetypes
from pymongo.errors import DuplicateKeyError

# services
from backend.services.mongodb_service import (
    insert_document,
    insert_collection,
    find_documents,
    update_document
)
from backend.services.local_storage_service import (
    save_file,
    get_file_path,
    create_processed_folder,
    save_processed_data,
    compute_bytes_hash
)

router = APIRouter()

class UploadCallback(BaseModel):
    file_path: str         # local storage path
    filename: str
    mime_type: str
    file_size: int
    org_id: str = ""
    uploader_id: str = ""
    category: str = "uploads"  # category folder

# Config
LOCAL_STORAGE_ROOT = Path("backend/data/local_storage")
LOCAL_PROCESSED_ROOT = Path("backend/data/processed")

# ----------------- sync helper functions (run in threads) -----------------
def extract_text(pdf_path: str) -> str:
    text_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_content += txt + "\n"
    return text_content

def extract_images_with_metadata(pdf_path: str):
    """
    Extract actual embedded images from the PDF using PyMuPDF and return a list of metadata:
    [
      {
        "page": int,
        "image_index": int,
        "filename": "xxx.png",
        "local_path": "/tmp/xxx.png",
        "bbox": null
      },
      ...
    ]
    """
    doc = fitz.open(pdf_path)
    saved = []
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            fname = f"{Path(pdf_path).stem}_p{page_num+1}_img{img_index}.{image_ext}"
            out_path = Path(tempfile.gettempdir()) / fname
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            saved.append({
                "page": page_num + 1,
                "image_index": img_index,
                "filename": fname,
                "local_path": str(out_path),
                "bbox": None
            })
    doc.close()
    return saved

def chunk_text(text: str, chunk_size: int = 1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def dehyphenate_text(text: str) -> str:
    """
    Fix common hyphenation where words split across lines like:
    "multi-\nple" -> "multiple"
    """
    # Replace hyphen at end of line followed by newline + lowercase start
    text = re.sub(r"-\n([a-z0-9])", r"\1", text)
    # Join words broken by newline (simple heuristic)
    text = re.sub(r"([a-z0-9])\n([a-z])", r"\1 \2", text)
    return text

def reflow_paragraphs(text: str, maxlen: int = 1000) -> str:
    """
    Collapse repeated newlines, keep paragraph breaks, produce paragraph strings.
    Useful to create chunks for embeddings later.
    """
    # unify CRLF
    text = text.replace("\r\n", "\n")
    # Collapse more than 2 newlines -> paragraph break
    text = re.sub(r'\n{2,}', '\n\n', text)
    # For single-line breaks inside paragraphs, replace with space
    paragraphs = []
    for para in text.split("\n\n"):
        # remove stray newlines inside paragraph
        p = " ".join(line.strip() for line in para.splitlines() if line.strip())
        paragraphs.append(p.strip())
    return "\n\n".join(paragraphs)


# ----------------- direct upload endpoint -----------------
@router.post("/upload-direct")
async def upload_file_direct(
    file: UploadFile,
    org_id: str = Form(""),
    uploader_id: str = Form(""),
    category: str = Form("uploads")
):
    """
    Direct file upload endpoint that:
    1) Receives file bytes from frontend
    2) Saves to local storage
    3) Triggers processing flow
    """
    # Read file bytes
    file_bytes = await file.read()
    
    # Save to local storage
    saved_path = save_file(file_bytes, category, file.filename)
    
    # Compute hash
    file_hash = compute_bytes_hash(file_bytes)
    
    # Prepare callback data
    callback_data = UploadCallback(
        file_path=str(saved_path),
        filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        file_size=len(file_bytes),
        org_id=org_id,
        uploader_id=uploader_id,
        category=category
    )
    
    # Process the uploaded file using existing callback logic
    return await upload_callback(callback_data)

# ----------------- main endpoint -----------------
@router.post("/upload-callback")
async def upload_callback(data: UploadCallback):
    """
    Flow:
      - If ZIP: handle ZIP flow (extract files, save to local storage, insert collection and document records)
      - Else: treat as single file (PDF expected) and run parsing flow:
          1) read file from local storage path
          2) extract text & images (in thread)
          3) create local processed folder and save parsed.json and images.json
          4) insert MongoDB record with local paths and metadata
    """

    # --------- ZIP handling branch ----------
    is_zip = (data.mime_type in ["application/zip", "application/x-zip-compressed"]) or data.filename.lower().endswith(".zip")
    if is_zip:
        # 1) Read ZIP from local storage
        try:
            zip_path = Path(data.file_path)
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found: {data.file_path}")
            zip_bytes = zip_path.read_bytes()
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to read ZIP from local storage: {e}")

        # 2) Save temp zip file
        try:
            tmp_zip_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
            os.close(tmp_zip_fd)
            with open(tmp_zip_path, "wb") as f:
                f.write(zip_bytes)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to write temp ZIP: {e}")

        # 3) Insert collection record
        try:
            collection = await asyncio.to_thread(insert_collection, {
                "name": data.filename,
                "local_path": data.file_path,
                "org_id": data.org_id,
                "uploader_id": data.uploader_id,
                "category": data.category
            })
        except Exception as e:
            traceback.print_exc()
            collection = {"_id": None, "error": str(e)}

        collection_id = collection.get("_id")

        # 4) Extract ZIP into a temp dir
        extracted_docs = []
        temp_extract_dir = tempfile.mkdtemp(prefix="upload_extract_")
        try:
            with zipfile.ZipFile(tmp_zip_path, 'r') as z:
                z.extractall(temp_extract_dir)
        except Exception as e:
            traceback.print_exc()
            # cleanup and return error
            try:
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                os.remove(tmp_zip_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to extract ZIP: {e}")

        # 5) Walk extracted files and save to local storage & insert records
        try:
            for root, dirs, files in os.walk(temp_extract_dir):
                for file in files:
                    # skip nested zips by default (safe)
                    if file.lower().endswith(".zip"):
                        continue

                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, temp_extract_dir).replace("\\", "/")

                    try:
                        # read bytes
                        with open(local_path, "rb") as fh:
                            file_bytes = fh.read()

                        # guess mime type
                        mime_type, _ = mimetypes.guess_type(local_path)
                        if not mime_type:
                            mime_type = "application/octet-stream"

                        # save to local storage
                        saved_path = save_file(file_bytes, data.category, file)
                        file_hash = compute_bytes_hash(file_bytes)

                        # create document record
                        doc_record = {
                            "collection_id": str(collection_id) if collection_id else None,
                            "local_path": str(saved_path),
                            "filename": file,
                            "file_hash": file_hash,
                            "file_size": len(file_bytes),
                            "mime_type": mime_type,
                            "category": data.category,
                            "org_id": data.org_id,
                            "uploader_id": data.uploader_id,
                            "status": "extracted"
                        }
                        try:
                            inserted = await asyncio.to_thread(insert_document, doc_record)
                            extracted_docs.append(inserted)
                        except DuplicateKeyError:
                            # File already exists
                            from backend.services.mongodb_service import get_document_by_hash
                            existing = await asyncio.to_thread(get_document_by_hash, file_hash)
                            if existing:
                                existing["status_message"] = "File already exists"
                                extracted_docs.append(existing)
                            else:
                                extracted_docs.append({"file": file, "error": "Duplicate file"})
                        except Exception as e:
                            traceback.print_exc()
                            extracted_docs.append({
                                "file": file,
                                "local_path": local_path,
                                "error": str(e)
                            })
                    except Exception as e:
                        traceback.print_exc()
                        extracted_docs.append({
                            "file": file,
                            "local_path": local_path,
                            "error": str(e)
                        })
        finally:
            # cleanup
            try:
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                os.remove(tmp_zip_path)
            except Exception:
                pass

        return {
            "status": "ok",
            "type": "zip",
            "collection": collection,
            "documents": extracted_docs
        }

    # --------- NON-ZIP (assume single file, e.g., PDF) ----------
    # 1) Read file from local storage
    try:
        file_path = Path(data.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {data.file_path}")
        file_bytes = file_path.read_bytes()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to read from local storage: {e}")

    # 2) write file to temp path (use filename from callback)
    try:
        safe_filename = Path(data.filename).name
        tmp_dir = Path(tempfile.gettempdir())
        tmp_file_path = tmp_dir / safe_filename
        tmp_file_path.write_bytes(file_bytes)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to write temp file: {e}")

    # 3) parse text and extract images (run blocking I/O in threads)
    try:
        text_content = await asyncio.to_thread(extract_text, str(tmp_file_path))
        # optional cleanup of hyphenation and paragraph reflow
        text_content = dehyphenate_text(text_content)
        text_content = reflow_paragraphs(text_content)
        text_chunks = await asyncio.to_thread(chunk_text, text_content)
        extracted_images_meta = await asyncio.to_thread(extract_images_with_metadata, str(tmp_file_path))
    except Exception as e:
        traceback.print_exc()
        try:
            tmp_file_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Error parsing file: {e}")

    # 4) prepare local processed folder
    uid = uuid.uuid4().hex
    local_doc_dir = create_processed_folder(uid)

    # Build parsed JSON object
    parsed_json = {
        "file_path": data.file_path,
        "filename": data.filename,
        "mime_type": data.mime_type,
        "chunks": text_chunks,
        "full_text_preview": text_content[:10000]
    }

    # Save parsed.json locally
    parsed_json_path = local_doc_dir / "parsed.json"
    parsed_json_path.write_text(json.dumps(parsed_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save images metadata locally (images.json) and move image files into local folder
    images_meta = []
    for img in extracted_images_meta:
        src = Path(img["local_path"])
        if not src.exists():
            continue
        dst = local_doc_dir / src.name
        src.replace(dst)  # move from tempdir into local folder
        img_entry = {
            "page": img["page"],
            "image_index": img["image_index"],
            "filename": img["filename"],
            "local_path": str(dst),
            "bbox": img.get("bbox")
        }
        images_meta.append(img_entry)

    images_json_path = local_doc_dir / "images.json"
    images_json_path.write_text(json.dumps(images_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) Build DB record and insert
    file_hash = compute_bytes_hash(file_bytes)
    record = {
        "local_path": data.file_path,
        "filename": data.filename,
        "file_hash": file_hash,
        "file_size": data.file_size,
        "mime_type": data.mime_type,
        "category": data.category,
        "org_id": data.org_id,
        "uploader_id": data.uploader_id,
        "status": "parsed",
        "parsed_json_local_path": str(parsed_json_path),
        "images_local_path": str(local_doc_dir),
        "chunk_count": len(text_chunks)
    }

    try:
        inserted = await asyncio.to_thread(insert_document, record)
    except DuplicateKeyError as e:
        # File already exists - find and return the existing document
        print(f"Duplicate file detected: {file_hash}")
        from backend.services.mongodb_service import get_document_by_hash
        existing_doc = await asyncio.to_thread(get_document_by_hash, file_hash)
        if existing_doc:
            inserted = existing_doc
            inserted["status_message"] = "File already exists in database"
        else:
            # Shouldn't happen, but handle gracefully
            raise HTTPException(status_code=409, detail="Duplicate file detected but could not retrieve existing record")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Clean up temp file if desired
    try:
        tmp_file_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {"status": "ok", "document": inserted, "local_folder": str(local_doc_dir)}
