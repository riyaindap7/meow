# backend/services/moe_scraper_service.py

"""
Ministry of Education Web Scraper Service

This service incrementally scrapes MoE websites and uploads new content to Google Drive.
It checks for updates in existing folders and only scrapes/uploads new content.

Designed to work with existing Google Drive folder structure:
- moe_scraped_higher_edu_RUSA
- Scraped_moe_archived_advertisment
- scraped_moe_archived_circulars
- Scraped_moe_archived_press_releases
- Scraped_moe_archived_scholarships
- scraped_moe_archived_updates
- scraped_moe_documents&reports
- scraped_moe_higher_education_schemes
- scraped_moe_mothly_achivements
- scraped_moe_rti
- scraped_moe_schemes
- scraped_moe_statistics
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
import hashlib
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import json
import os
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
import io

# Load environment
load_dotenv()


class MoEScraperService:
    """
    Incremental MoE scraper that checks for updates and uploads only new content
    """
    
    # Mapping of Drive folders to their source URLs and scraping config
    SCRAPE_CONFIG = {
        "moe_scraped_higher_edu_RUSA": {
            "url": "https://www.education.gov.in/en/rusa",
            "description": "RUSA (Rashtriya Uchchatar Shiksha Abhiyan) documents"
        },
        "Scraped_moe_archived_advertisment": {
            "url": "https://www.education.gov.in/en/advertisements",
            "description": "Archived advertisements"
        },
        "scraped_moe_archived_circulars": {
            "url": "https://www.education.gov.in/en/archives-circulars-orders-notification",
            "description": "Archived circulars"
        },
        "Scraped_moe_archived_press_releases": {
            "url": "https://www.education.gov.in/en/press-releases",
            "description": "Archived press releases"
        },
        "Scraped_moe_archived_scholarships": {
            "url": "https://www.education.gov.in/en/scholarships",
            "description": "Archived scholarships"
        },
        "scraped_moe_archived_updates": {
            "url": "https://www.education.gov.in/en/updates",
            "description": "Archived updates"
        },
        "scraped_moe_documents&reports": {
            "url": "https://www.education.gov.in/en/documents_reports",
            "description": "Documents and reports"
        },
        "scraped_moe_higher_education_schemes": {
            "url": "https://www.education.gov.in/en/higher_education",
            "description": "Higher education schemes"
        },
        "scraped_moe_mothly_achivements": {
            "url": "https://www.education.gov.in/archives-monthly-achievements",
            "description": "Monthly achievements"
        },
        "scraped_moe_rti": {
            "url": "https://www.education.gov.in/en/rti_he",
            "description": "RTI (Right to Information) documents"
        },
        "scraped_moe_schemes": {
            "url": "https://www.education.gov.in/en/schemes",
            "description": "Government schemes"
        },
        "scraped_moe_statistics": {
            "url": "https://www.education.gov.in/en/statistics-new",
            "description": "Educational statistics"
        }
    }
    
    def __init__(self, drive_service, master_folder_id: str):
        """
        Initialize scraper service
        
        Args:
            drive_service: Authenticated Google Drive API service
            master_folder_id: Google Drive master folder ID containing all scrape folders
        """
        self.drive_service = drive_service
        self.master_folder_id = master_folder_id
        
        # Detect and cache the Shared Drive ID
        self.shared_drive_id = self._get_shared_drive_id()
        
        # Cache of existing files in Drive (folder_name -> set of filenames)
        self.existing_files_cache: Dict[str, Set[str]] = {}
        
        # Folder ID cache (folder_name -> drive_folder_id)
        self.folder_id_cache: Dict[str, str] = {}
        
        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Stats
        self.stats = {
            "folders_checked": 0,
            "pdfs_found": 0,
            "pdfs_new": 0,
            "pdfs_uploaded": 0,
            "pdfs_skipped": 0,
            "errors": []
        }
    
    def _get_shared_drive_id(self) -> Optional[str]:
        """
        Get the Shared Drive ID for the master folder
        
        Returns None if folder is in My Drive (won't work with service accounts)
        """
        try:
            file_meta = self.drive_service.files().get(
                fileId=self.master_folder_id,
                fields='id, name, driveId',
                supportsAllDrives=True
            ).execute()
            
            drive_id = file_meta.get('driveId')
            folder_name = file_meta.get('name', 'Unknown')
            
            if drive_id:
                print(f"✓ Detected Shared Drive: {folder_name}")
                print(f"  Drive ID: {drive_id}")
            else:
                print(f"⚠️  WARNING: Folder '{folder_name}' is in My Drive, not a Shared Drive!")
                print(f"   Service accounts cannot upload to My Drive.")
                print(f"   Please use a folder within a Shared Drive.")
            
            return drive_id
            
        except Exception as e:
            print(f"Error detecting Shared Drive: {e}")
            return None
    
    def get_folder_id(self, folder_name: str) -> Optional[str]:
        """
        Get Google Drive folder ID by name
        
        Caches results to avoid repeated API calls
        """
        if folder_name in self.folder_id_cache:
            return self.folder_id_cache[folder_name]
        
        try:
            query = f"name='{folder_name}' and '{self.master_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            
            results = self.drive_service.files().list(
                q=query,
                fields='files(id, name)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            files = results.get('files', [])
            
            if files:
                folder_id = files[0]['id']
                self.folder_id_cache[folder_name] = folder_id
                return folder_id
            else:
                print(f"Warning: Folder '{folder_name}' not found in Drive")
                return None
                
        except Exception as e:
            print(f"Error getting folder ID for '{folder_name}': {e}")
            return None
    
    def get_existing_files_in_folder(self, folder_id: str) -> Set[str]:
        """
        Get set of existing filenames in a Drive folder
        
        Used to avoid re-uploading duplicates
        """
        try:
            query = f"'{folder_id}' in parents and trashed=false"
            
            files = []
            page_token = None
            
            while True:
                results = self.drive_service.files().list(
                    q=query,
                    fields='nextPageToken, files(name)',
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True
                ).execute()
                
                files.extend(results.get('files', []))
                page_token = results.get('nextPageToken')
                
                if not page_token:
                    break
            
            filenames = {f['name'] for f in files}
            print(f"  Found {len(filenames)} existing files in folder")
            return filenames
            
        except Exception as e:
            print(f"Error listing files in folder: {e}")
            return set()
    
    def find_pdf_links(self, url: str) -> List[Dict[str, str]]:
        """
        Find all PDF download links on a webpage
        
        Returns:
            List of dicts with 'url', 'title', 'filename' for each PDF
        """
        try:
            print(f"  Scraping: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            pdf_links = []
            seen_urls = set()
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Check if it's a PDF link
                if href.lower().endswith('.pdf') or '/pdf/' in href.lower() or 'download' in href.lower():
                    # Make absolute URL
                    absolute_url = urljoin(url, href)
                    
                    # Skip duplicates
                    if absolute_url in seen_urls:
                        continue
                    seen_urls.add(absolute_url)
                    
                    # Get link text/title
                    link_text = link.get_text(strip=True)
                    title = link.get('title', link_text)
                    
                    # Generate filename from URL or title
                    parsed = urlparse(absolute_url)
                    filename = Path(parsed.path).name
                    
                    if not filename or filename == '':
                        # Generate from title or hash
                        filename = self._sanitize_filename(title or 'document') + '.pdf'
                    
                    if not filename.lower().endswith('.pdf'):
                        filename += '.pdf'
                    
                    pdf_links.append({
                        'url': absolute_url,
                        'title': title or 'Untitled',
                        'filename': filename
                    })
            
            print(f"  Found {len(pdf_links)} PDF links")
            return pdf_links
            
        except Exception as e:
            print(f"  Error scraping {url}: {e}")
            self.stats["errors"].append(f"Scraping error ({url}): {str(e)}")
            return []
    
    def _sanitize_filename(self, filename: str) -> str:
        """Clean filename for safe storage"""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename.strip()
    
    def download_pdf(self, url: str, filename: str) -> Optional[bytes]:
        """
        Download PDF file
        
        Returns:
            PDF bytes, or None if failed
        """
        try:
            print(f"    Downloading: {filename}")
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Read content
            pdf_bytes = response.content
            
            # Verify it's actually a PDF (check magic bytes)
            if not pdf_bytes.startswith(b'%PDF'):
                print(f"    Warning: File doesn't appear to be a PDF, skipping")
                return None
            
            print(f"    Downloaded {len(pdf_bytes)} bytes")
            return pdf_bytes
            
        except Exception as e:
            print(f"    Error downloading {url}: {e}")
            self.stats["errors"].append(f"Download error ({filename}): {str(e)}")
            return None
    
    def upload_to_drive(self, pdf_bytes: bytes, filename: str, folder_id: str) -> Optional[str]:
        """
        Upload PDF bytes to Google Drive folder
        
        Returns:
            Google Drive file ID, or None if failed
        """
        try:
            # Prepare file metadata
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            
            # Create media upload from bytes
            media = MediaIoBaseUpload(
                io.BytesIO(pdf_bytes),
                mimetype='application/pdf',
                resumable=True
            )
            
            # Upload
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, size, modifiedTime',
                supportsAllDrives=True
            ).execute()
            
            file_id = file.get('id')
            print(f"    ✓ Uploaded to Drive: {filename} (ID: {file_id})")
            
            return file_id
            
        except Exception as e:
            print(f"    Error uploading {filename} to Drive: {e}")
            self.stats["errors"].append(f"Upload error ({filename}): {str(e)}")
            return None
    
    def scrape_folder(self, folder_name: str, config: Dict) -> Dict:
        """
        Scrape a single MoE page and upload new PDFs to Drive folder
        
        Args:
            folder_name: Name of Google Drive folder
            config: Scraping configuration (url, description)
            
        Returns:
            Dict with scraping statistics
        """
        print(f"\n{'='*60}")
        print(f"Scraping: {folder_name}")
        print(f"URL: {config['url']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        # Get folder ID
        folder_id = self.get_folder_id(folder_name)
        
        if not folder_id:
            return {
                "folder_name": folder_name,
                "status": "error",
                "error": "Folder not found in Drive"
            }
        
        # Get existing files in folder
        existing_files = self.get_existing_files_in_folder(folder_id)
        
        # Find PDF links on page
        pdf_links = self.find_pdf_links(config['url'])
        self.stats["pdfs_found"] += len(pdf_links)
        
        # Track results for this folder
        folder_stats = {
            "folder_name": folder_name,
            "pdfs_found": len(pdf_links),
            "pdfs_new": 0,
            "pdfs_uploaded": 0,
            "pdfs_skipped": 0,
            "errors": []
        }
        
        # Process each PDF
        for i, pdf_info in enumerate(pdf_links, 1):
            pdf_url = pdf_info['url']
            filename = pdf_info['filename']
            title = pdf_info['title']
            
            print(f"\n  [{i}/{len(pdf_links)}] {filename}")
            
            # Check if already exists
            if filename in existing_files:
                print(f"    → Already exists, skipping")
                folder_stats["pdfs_skipped"] += 1
                self.stats["pdfs_skipped"] += 1
                continue
            
            folder_stats["pdfs_new"] += 1
            self.stats["pdfs_new"] += 1
            
            # Download PDF
            pdf_bytes = self.download_pdf(pdf_url, filename)
            
            if not pdf_bytes:
                folder_stats["errors"].append(f"Failed to download: {filename}")
                continue
            
            # Upload to Drive
            file_id = self.upload_to_drive(pdf_bytes, filename, folder_id)
            
            if file_id:
                folder_stats["pdfs_uploaded"] += 1
                self.stats["pdfs_uploaded"] += 1
                
                # Add to existing files cache
                existing_files.add(filename)
            else:
                folder_stats["errors"].append(f"Failed to upload: {filename}")
            
            # Be nice to the server
            time.sleep(2)
        
        # Summary for this folder
        print(f"\n  Summary for {folder_name}:")
        print(f"    PDFs found: {folder_stats['pdfs_found']}")
        print(f"    New PDFs: {folder_stats['pdfs_new']}")
        print(f"    Uploaded: {folder_stats['pdfs_uploaded']}")
        print(f"    Skipped: {folder_stats['pdfs_skipped']}")
        print(f"    Errors: {len(folder_stats['errors'])}")
        
        return folder_stats
    
    def scrape_all_folders(self) -> Dict:
        """
        Scrape all configured MoE pages and upload to respective Drive folders
        
        Returns:
            Combined statistics for all folders
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"MoE INCREMENTAL SCRAPER")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        folder_results = []
        
        for folder_name, config in self.SCRAPE_CONFIG.items():
            self.stats["folders_checked"] += 1
            
            result = self.scrape_folder(folder_name, config)
            folder_results.append(result)
            
            # Brief pause between folders
            time.sleep(3)
        
        duration = time.time() - start_time
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETED")
        print(f"{'='*60}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Folders checked: {self.stats['folders_checked']}")
        print(f"Total PDFs found: {self.stats['pdfs_found']}")
        print(f"New PDFs: {self.stats['pdfs_new']}")
        print(f"PDFs uploaded: {self.stats['pdfs_uploaded']}")
        print(f"PDFs skipped (already exist): {self.stats['pdfs_skipped']}")
        print(f"Total errors: {len(self.stats['errors'])}")
        print(f"{'='*60}\n")
        
        # Prepare final report
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "summary": self.stats,
            "folder_results": folder_results
        }
        
        return report
    
    def scrape_specific_folders(self, folder_names: List[str]) -> Dict:
        """
        Scrape only specific folders
        
        Args:
            folder_names: List of folder names to scrape
            
        Returns:
            Statistics for scraped folders
        """
        start_time = time.time()
        folder_results = []
        
        for folder_name in folder_names:
            if folder_name not in self.SCRAPE_CONFIG:
                print(f"Warning: Unknown folder '{folder_name}', skipping")
                continue
            
            config = self.SCRAPE_CONFIG[folder_name]
            self.stats["folders_checked"] += 1
            
            result = self.scrape_folder(folder_name, config)
            folder_results.append(result)
            
            time.sleep(3)
        
        duration = time.time() - start_time
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "summary": self.stats,
            "folder_results": folder_results
        }
        
        return report


def create_scraper_service() -> MoEScraperService:
    """
    Factory function to create authenticated MoE scraper service
    
    Reads credentials from environment variables
    """
    service_account_file = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE")
    master_folder_id = os.getenv("GOOGLE_DRIVE_MASTER_FOLDER_ID")
    
    if not service_account_file:
        raise ValueError("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE environment variable not set")
    
    if not master_folder_id:
        raise ValueError("GOOGLE_DRIVE_MASTER_FOLDER_ID environment variable not set")
    
    # Authenticate with Google Drive
    scopes = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes
    )
    
    drive_service = build('drive', 'v3', credentials=creds)
    
    return MoEScraperService(drive_service, master_folder_id)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE Incremental Scraper")
    parser.add_argument("--all", action="store_true", help="Scrape all folders")
    parser.add_argument("--folders", nargs="+", help="Scrape specific folders")
    parser.add_argument("--list", action="store_true", help="List all configured folders")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nConfigured folders:")
        for folder_name, config in MoEScraperService.SCRAPE_CONFIG.items():
            print(f"  - {folder_name}")
            print(f"    URL: {config['url']}")
            print(f"    Description: {config['description']}\n")
    
    elif args.all:
        scraper = create_scraper_service()
        report = scraper.scrape_all_folders()
        
        # Save report
        report_file = Path(f"scrape_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_file.write_text(json.dumps(report, indent=2))
        print(f"Report saved to: {report_file}")
    
    elif args.folders:
        scraper = create_scraper_service()
        report = scraper.scrape_specific_folders(args.folders)
        
        # Save report
        report_file = Path(f"scrape_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_file.write_text(json.dumps(report, indent=2))
        print(f"Report saved to: {report_file}")
    
    else:
        parser.print_help()
