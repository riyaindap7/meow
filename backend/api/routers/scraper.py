# backend/api/routers/scraper.py

"""
MoE Scraper API Router

Endpoints for testing and running the Ministry of Education web scraper
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

from backend.services.moe_scraper_service import create_scraper_service, MoEScraperService

router = APIRouter(prefix="/scraper", tags=["scraper"])


class ScraperRequest(BaseModel):
    """Request to run scraper"""
    folders: Optional[List[str]] = None  # None means scrape all


@router.get("/folders")
async def list_folders():
    """
    List all configured scraper folders
    """
    folders = []
    for folder_name, config in MoEScraperService.SCRAPE_CONFIG.items():
        folders.append({
            "name": folder_name,
            "url": config["url"],
            "description": config["description"]
        })
    
    return {"folders": folders}


@router.post("/run")
async def run_scraper(request: ScraperRequest, background_tasks: BackgroundTasks):
    """
    Run the MoE scraper
    
    This scrapes MoE websites and uploads new PDFs to Google Drive.
    Runs in background to avoid timeout.
    
    Request body (optional):
    {
        "folders": ["scraped_moe_schemes"]  // Scrape specific folders, or null for all
    }
    """
    try:
        scraper = create_scraper_service()
        
        def run_scrape():
            if request.folders:
                return scraper.scrape_specific_folders(request.folders)
            else:
                return scraper.scrape_all_folders()
        
        # Run in background
        background_tasks.add_task(run_scrape)
        
        return {
            "status": "started",
            "message": "Scraper started in background",
            "folders": request.folders or "all"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start scraper: {str(e)}")


@router.post("/test")
async def test_scraper(folder_name: str):
    """
    Test scraper on a single folder (synchronous for testing)
    
    Example: POST /api/scraper/test?folder_name=scraped_moe_schemes
    """
    try:
        if folder_name not in MoEScraperService.SCRAPE_CONFIG:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown folder '{folder_name}'. Use /folders endpoint to see available folders."
            )
        
        scraper = create_scraper_service()
        config = MoEScraperService.SCRAPE_CONFIG[folder_name]
        
        # Run scrape for single folder
        result = scraper.scrape_folder(folder_name, config)
        
        return {
            "status": "completed",
            "result": result,
            "summary": scraper.stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraper test failed: {str(e)}")
