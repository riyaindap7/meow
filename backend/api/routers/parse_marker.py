# backend/api/routers/parse_marker.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
import asyncio
import subprocess
import tempfile
from pathlib import Path
import json
import uuid
from backend.services.local_storage_service import get_file_path, save_file, create_processed_folder
from backend.services.mongodb_service import insert_document, update_document, find_document
import fitz  # PyMuPDF

router = APIRouter()

async def run_marker_and_get_json(pdf_path: Path, output_format: str = "json", use_llm: bool = False):
    out_dir = pdf_path.parent / "marker_out"
    out_dir.mkdir(exist_ok=True)
    cmd = ["marker_single", str(pdf_path), "--output_format", output_format, "--output_dir", str(out_dir)]
    if use_llm:
        cmd += ["--use_llm"]
    # run in thread to avoid blocking
    def _run():
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r
    result = await asyncio.to_thread(_run)
    if result.returncode != 0:
        raise RuntimeError(f"Marker failed: {result.stderr[:1000]}")
    # Find first json file
    json_files = list(out_dir.glob("*.json"))
    if not json_files:
        # fall back: read all text outputs
        outputs = {}
        for p in out_dir.iterdir():
            if p.is_file():
                outputs[p.name] = p.read_text(encoding="utf-8", errors="ignore")
        return {"outputs": outputs}
    # parse json
    text = json_files[0].read_text(encoding="utf-8")
    return json.loads(text)

def extract_and_save_figures(pdf_path: Path, marker_json: dict, temp_dir: Path):
    """
    Use marker_json to find figure blocks and crop/save them.
    Returns list of dicts: {page, bbox, filename, remote_path}
    """
    saved = []
    doc = fitz.open(str(pdf_path))
    # Marker JSON shape: top-level 'blocks' (or similar). Try to find figure blocks robustly.
    blocks = marker_json.get("blocks") or marker_json.get("content") or marker_json.get("elements") or []
    for block in blocks:
        # Marker may use type or block_type; check both
        if block.get("type") == "figure" or block.get("block_type") == "figure":
            page_num = block.get("page", 1)
            bbox = block.get("bbox")  # expected [x0, y0, x1, y1]
            if not bbox:
                continue
            # fitz pages are 0-indexed
            page = doc[page_num - 1]
            rect = fitz.Rect(*bbox)
            pix = page.get_pixmap(clip=rect, dpi=150)
            fname = f"figure_p{page_num}_{uuid.uuid4().hex}.png"
            out_path = temp_dir / fname
            pix.save(str(out_path))
            saved.append({
                "page": page_num,
                "bbox": bbox,
                "filename": fname,
                "local_path": str(out_path)
            })
    doc.close()
    return saved

async def save_extracted_images(saved_list, doc_id: str):
    """Save extracted images to processed folder"""
    results = []
    processed_folder = create_processed_folder(doc_id)
    
    for item in saved_list:
        local_path = Path(item["local_path"])
        # Move to processed folder
        dest_path = processed_folder / local_path.name
        local_path.replace(dest_path)
        
        results.append({
            "local": str(dest_path),
            "filename": local_path.name,
            "page": item.get("page"),
            "bbox": item.get("bbox")
        })
    return results

@router.post("/parse-and-extract")
async def parse_and_extract(document_id: str, background: bool = False, use_llm: bool = False, bg: BackgroundTasks = None):
    """
    Args:
      document_id: MongoDB document ID
      background: if true, processing runs but returns 202 immediately
    """
    # 1. Get document from MongoDB
    doc = find_document({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    local_path = doc.get("local_path")
    if not local_path or not Path(local_path).exists():
        raise HTTPException(status_code=404, detail="Local file not found")
    
    # 2. Read PDF file
    try:
        pdf_path = Path(local_path)
        pdf_bytes = pdf_path.read_bytes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    # write to temp file for marker processing
    td = Path(tempfile.mkdtemp())
    temp_pdf_path = td / "input.pdf"
    temp_pdf_path.write_bytes(pdf_bytes)

    # Optionally run in background tasks
    async def _process():
        try:
            marker_json = await run_marker_and_get_json(temp_pdf_path, output_format="json", use_llm=use_llm)
            saved = extract_and_save_figures(temp_pdf_path, marker_json, td)
            image_results = await save_extracted_images(saved, document_id)
            
            # Update MongoDB record with parsed content
            try:
                processed_folder = create_processed_folder(document_id)
                parsed_json_path = processed_folder / "parsed.json"
                parsed_json_path.write_text(json.dumps(marker_json, indent=2))
                
                images_json_path = processed_folder / "images.json"
                images_json_path.write_text(json.dumps(image_results, indent=2))
                
                update_document(
                    {"_id": document_id},
                    {
                        "status": "parsed",
                        "parsed_json_local_path": str(parsed_json_path),
                        "images_local_path": str(processed_folder),
                        "chunk_count": len(marker_json.get("blocks", []))
                    }
                )
            except Exception as e:
                print(f"Error updating MongoDB: {e}")
        finally:
            # cleanup temp files
            for p in td.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                td.rmdir()
            except Exception:
                pass
        return {"status": "done"}

    if background:
        # schedule background processing
        bg.add_task(asyncio.create_task, _process())
        return {"status": "accepted", "document_id": document_id}
    else:
        result = await _process()
        return {"status": "ok", "document_id": document_id, "result": result}
