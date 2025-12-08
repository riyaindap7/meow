import os
import subprocess
from pathlib import Path

BASE_STORAGE_DIR = Path("backend/victorDB/single_pdf").resolve()
OUTPUT_BASE_DIR = BASE_STORAGE_DIR / "outputs"


def compute_output_dir(pdf_path: Path) -> Path:
    """Return output folder: backend/victorDB/single_pdf/outputs/<filename>"""
    name = pdf_path.stem
    return OUTPUT_BASE_DIR / name


def is_already_parsed(output_dir: Path) -> bool:
    """MinerU output markers"""
    markers = [
        "auto/content_list.json",
        "auto/layout.pdf",
        "auto/middle.json"
    ]
    return all((output_dir / m).exists() for m in markers)


def parse_single_pdf(pdf_path: str, skip_existing=True):
    pdf_path = Path(pdf_path).resolve()
    output_dir = compute_output_dir(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and is_already_parsed(output_dir):
        return {
            "status": "skipped",
            "message": f"Already parsed: {pdf_path.name}",
            "output_path": str(output_dir)
        }

    try:
        result = subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(output_dir)],
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "message": f"Parsed: {pdf_path.name}",
                "output_path": str(output_dir)
            }
        else:
            return {
                "status": "failed",
                "message": f"MinerU error code: {result.returncode}",
                "output_path": str(output_dir)
            }

    except FileNotFoundError:
        return {
            "status": "error",
            "message": "⚠️ mineru not found. Install with: pip install mineru",
            "output_path": str(output_dir)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "output_path": str(output_dir)
        }
