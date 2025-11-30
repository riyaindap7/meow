# backend/services/moe_scraper_service.py

"""
Ministry of Education Web Scraper Service (Rclone-only version)

This service scrapes MoE websites and uploads new content to Google Drive using rclone.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from typing import List, Dict, Optional, Set
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import subprocess

# Load environment
load_dotenv()


class MoEScraperService:
    """
    MoE scraper that uses rclone for Google Drive operations
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
    
    def __init__(self):
        """Initialize scraper service with rclone"""
        # Rclone setup
        self.rclone_remote = os.getenv('RCLONE_REMOTE', 'gdrive:')
        self.master_folder_id = os.getenv('GOOGLE_DRIVE_MASTER_FOLDER_ID')
        
        # Cache of existing files (folder_name -> set of filenames)
        self.existing_files_cache: Dict[str, Set[str]] = {}
        
        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
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
    
    def _rclone_path(self, *parts) -> str:
        """Build rclone path from folder ID and subpaths"""
        if self.master_folder_id:
            # Use direct folder ID access
            base = f"{self.rclone_remote}{self.master_folder_id}"
            if parts:
                return f"{base}/{'/'.join(parts)}"
            return base
        else:
            # Fallback to named path
            return f"{self.rclone_remote}{'/'.join(parts)}"
    
    def get_existing_files_in_folder(self, folder_name: str) -> Set[str]:
        """Get set of existing filenames in a Drive folder via rclone"""
        try:
            path = self._rclone_path("pdfs", folder_name)
            result = subprocess.run([
                "rclone", "lsf",
                path,
                "--drive-shared-with-me"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                files = set(result.stdout.strip().split('\n'))
                filenames = {f for f in files if f}  # Remove empty strings
                print(f"  Found {len(filenames)} existing files in folder")
                return filenames
            return set()
        except Exception as e:
            print(f"Error listing files via rclone: {e}")
            return set()
    
    def download_pdf(self, url: str, filename: str) -> Optional[bytes]:
        """Download PDF file"""
        try:
            print(f"    Downloading: {filename}")
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            pdf_bytes = response.content
            
            # Verify it's actually a PDF
            if not pdf_bytes.startswith(b'%PDF'):
                print(f"    Warning: File doesn't appear to be a PDF, skipping")
                return None
            
            print(f"    Downloaded {len(pdf_bytes)} bytes")
            return pdf_bytes
            
        except Exception as e:
            print(f"    Error downloading {url}: {e}")
            self.stats["errors"].append(f"Download error ({filename}): {str(e)}")
            return None
    
    def upload_to_drive(self, pdf_bytes: bytes, filename: str, folder_name: str) -> Optional[str]:
        """Upload PDF bytes to Google Drive via rclone"""
        try:
            # Save temporarily
            temp_path = Path(f"data/moe/temp/{filename}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(pdf_bytes)
            
            # Upload via rclone
            remote_path = self._rclone_path("pdfs", folder_name, filename)
            result = subprocess.run([
                "rclone", "copyto", 
                str(temp_path), 
                remote_path,
                "--drive-shared-with-me"
            ], capture_output=True, text=True)
            
            # Clean up temp file
            temp_path.unlink()
            
            if result.returncode == 0:
                print(f"    ✓ Uploaded via rclone: {filename}")
                return filename
            else:
                print(f"    ⚠️ Rclone upload failed: {result.stderr}")
                self.stats["errors"].append(f"Rclone upload error ({filename}): {result.stderr}")
                return None
                
        except Exception as e:
            print(f"    Error uploading {filename} via rclone: {e}")
            self.stats["errors"].append(f"Upload error ({filename}): {str(e)}")
            return None
    
    def find_pdf_links(self, url: str) -> List[Dict[str, str]]:
        """Find all PDF download links on a webpage"""
        try:
            print(f"  Scraping: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            pdf_links = []
            seen_urls = set()
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Check if it's a PDF link
                if href.lower().endswith('.pdf') or '/pdf/' in href.lower() or 'download' in href.lower():
                    absolute_url = urljoin(url, href)
                    
                    if absolute_url in seen_urls:
                        continue
                    seen_urls.add(absolute_url)
                    
                    link_text = link.get_text(strip=True)
                    title = link.get('title', link_text)
                    
                    parsed = urlparse(absolute_url)
                    filename = Path(parsed.path).name
                    
                    if not filename or filename == '':
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
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename.strip()
    
    def scrape_folder(self, folder_name: str, config: Dict) -> Dict:
        """Scrape a single folder configuration"""
        print(f"\n{'='*60}")
        print(f"Scraping: {folder_name}")
        print(f"URL: {config['url']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        self.stats["folders_checked"] += 1
        
        # Get existing files in folder
        existing_files = self.get_existing_files_in_folder(folder_name)
        
        # Find PDF links on page
        pdf_links = self.find_pdf_links(config['url'])
        self.stats["pdfs_found"] += len(pdf_links)
        
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
            result = self.upload_to_drive(pdf_bytes, filename, folder_name)
            
            if result:
                folder_stats["pdfs_uploaded"] += 1
                self.stats["pdfs_uploaded"] += 1
                existing_files.add(filename)
            else:
                folder_stats["errors"].append(f"Failed to upload: {filename}")
            
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
        """Scrape all configured MoE pages"""
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"MoE INCREMENTAL SCRAPER")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        folder_results = []
        
        for folder_name, config in self.SCRAPE_CONFIG.items():
            result = self.scrape_folder(folder_name, config)
            folder_results.append(result)
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
        print(f"PDFs skipped: {self.stats['pdfs_skipped']}")
        print(f"Total errors: {len(self.stats['errors'])}")
        print(f"{'='*60}\n")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "summary": self.stats,
            "folder_results": folder_results
        }
        
        return report
    
    def scrape_specific_folders(self, folder_names: List[str]) -> Dict:
        """Scrape only specific folders"""
        start_time = time.time()
        folder_results = []
        
        for folder_name in folder_names:
            if folder_name not in self.SCRAPE_CONFIG:
                print(f"Warning: Unknown folder '{folder_name}', skipping")
                continue
            
            config = self.SCRAPE_CONFIG[folder_name]
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
    """Factory function to create MoE scraper service"""
    rclone_remote = os.getenv("RCLONE_REMOTE", "gdrive:")
    master_folder_id = os.getenv("GOOGLE_DRIVE_MASTER_FOLDER_ID")
    
    if not master_folder_id:
        raise ValueError("GOOGLE_DRIVE_MASTER_FOLDER_ID environment variable not set")
    
    print(f"Using rclone mode: {rclone_remote}")
    print(f"Master folder ID: {master_folder_id}")
    
    return MoEScraperService()


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
