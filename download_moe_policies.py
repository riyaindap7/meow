#!/usr/bin/env python3
"""
download_moe_policies.py

Crawls a set of official Indian education domains (Ministry of Education + related bodies),
finds policy / amendment documents (PDF/DOC/DOCX/RTF), downloads them and packages them into a ZIP.

Usage:
    python3 download_moe_policies.py

Outputs:
    ./downloads/  (downloaded files)
    ./moe_policies.zip

Notes:
 - Script respects robots.txt for each domain.
 - Default polite rate limit: 1 second between requests (adjust RATE_LIMIT variable).
 - If you want to add/remove domains or seed URLs, edit DOMAIN_WHITELIST and SEED_URLS below.
"""

import os
import time
import re
import sys
import shutil
import zipfile
import logging
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup
from collections import deque
from urllib.robotparser import RobotFileParser

# === CONFIG ===
DOMAIN_WHITELIST = {
    "education.gov.in",
    "mhrd.gov.in",
    "ncert.nic.in",
    "ugc.ac.in",
    "deb.ugc.ac.in",
    "ncte.gov.in",
    "aicte-india.org",
    # add more domains as needed
}

# Seed pages to start crawling (authoritative pages)
SEED_URLS = [
    "https://www.education.gov.in/en/policy_initiatives",
    "https://www.education.gov.in/en/national-education-policy",
    "https://www.education.gov.in/sites/upload_files/mhrd/files/NEP_Final_English_0.pdf",
    "https://ncert.nic.in/",
    "https://deb.ugc.ac.in/",
    "https://ugc.ac.in/",
    "https://ncte.gov.in/",
    "https://www.aicte-india.org/",
]

DOWNLOAD_DIR = "downloads"
ZIP_FILENAME = "moe_policies.zip"
USER_AGENT = "MOE-Policy-Scraper/1.0 (+https://example.org) PythonRequests"
RATE_LIMIT = 1.0  # seconds between requests
MAX_PAGES = 2000   # safety limit to avoid runaway crawl
ALLOWED_EXTENSIONS = (".pdf", ".doc", ".docx", ".rtf", ".ppt", ".pptx")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === helpers ===

def domain_of(url):
    return urlparse(url).netloc.lower()

def same_domain_allowed(url):
    d = domain_of(url).split(':')[0]
    return d in DOMAIN_WHITELIST

def sanitize_filename(url):
    parsed = urlparse(url)
    path = parsed.path
    filename = os.path.basename(path)
    if not filename:
        filename = "index"
    # add query hash to avoid collisions
    if parsed.query:
        filename = filename + "_" + str(abs(hash(parsed.query)))[:8]
    # basic sanitize
    filename = re.sub(r'[^0-9A-Za-z._-]', '_', filename)
    return filename

def is_document_link(href):
    if not href:
        return False
    href = href.split('#')[0]
    href_lower = href.lower()
    for ext in ALLOWED_EXTENSIONS:
        if href_lower.endswith(ext):
            return True
    return False

def obeys_robots(url, robots_cache):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base in robots_cache:
        rp = robots_cache[base]
    else:
        rp = RobotFileParser()
        robots_txt = urljoin(base, "/robots.txt")
        try:
            rp.set_url(robots_txt)
            rp.read()
        except Exception:
            # if robots fail to load, default to allowing
            rp = None
        robots_cache[base] = rp
    if rp is None:
        return True
    return rp.can_fetch(USER_AGENT, url)

# === crawler ===

def crawl_and_collect(seed_urls):
    visited = set()
    q = deque(seed_urls)
    found_docs = {}  # url -> source page
    robots_cache = {}

    pages_crawled = 0
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    while q and pages_crawled < MAX_PAGES:
        url = q.popleft()
        url, _ = urldefrag(url)  # remove fragment
        if url in visited:
            continue
        visited.add(url)

        if not same_domain_allowed(url):
            logging.debug(f"Skipping out-of-whitelist URL: {url}")
            continue

        if not obeys_robots(url, robots_cache):
            logging.info(f"Blocked by robots.txt: {url}")
            continue

        try:
            logging.info(f"Fetching: {url}")
            resp = session.get(url, timeout=20)
            time.sleep(RATE_LIMIT)
        except Exception as e:
            logging.warning(f"Request failed for {url}: {e}")
            continue

        pages_crawled += 1
        content_type = resp.headers.get("Content-Type", "").lower()

        # if direct document (pdf/doc), add to found_docs
        if any(ct in content_type for ct in ("application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument")) or any(url.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            # record document URL; source is itself
            found_docs[url] = url
            logging.info(f"Found document URL (direct): {url}")
            continue

        # parse html for links
        if "text/html" in content_type or "application/xhtml+xml" in content_type or resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            # find all links
            for a in soup.find_all("a", href=True):
                href = a['href'].strip()
                # make absolute
                href = urljoin(url, href)
                href, _ = urldefrag(href)
                # prioritize documents
                if is_document_link(href):
                    if same_domain_allowed(href):
                        if href not in found_docs:
                            found_docs[href] = url
                            logging.info(f"Queued document: {href} (from {url})")
                    continue
                # otherwise, if same domain and not visited, queue
                if same_domain_allowed(href) and href not in visited:
                    q.append(href)
            # also look for <iframe> or <embed> pointing to pdfs
            for tag in soup.find_all(["iframe", "embed"], src=True):
                src = urljoin(url, tag['src'])
                if is_document_link(src) and same_domain_allowed(src):
                    if src not in found_docs:
                        found_docs[src] = url
                        logging.info(f"Queued embedded document: {src} (from {url})")

    logging.info(f"Crawl finished. Pages crawled: {pages_crawled}, documents found: {len(found_docs)}")
    return found_docs

# === downloader ===

def download_documents(doc_map):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    downloaded = []
    for url, source in doc_map.items():
        try:
            logging.info(f"Downloading {url}")
            # head-check
            head = session.head(url, allow_redirects=True, timeout=20)
            ctype = head.headers.get("Content-Type", "").lower()
            # allow only if content-type looks like document OR url endswith extension
            if not any(x in ctype for x in ("pdf", "msword", "officedocument", "rtf")) and not url.lower().endswith(ALLOWED_EXTENSIONS):
                logging.warning(f"Skipping (content-type mismatch): {url} ({ctype})")
                continue
            # stream download
            r = session.get(url, stream=True, timeout=60)
            time.sleep(RATE_LIMIT)
            filename = sanitize_filename(url)
            # attempt to infer extension from headers or url
            if '.' not in filename or filename.endswith('_'):
                # try Content-Disposition
                cd = r.headers.get("Content-Disposition", "")
                m = re.search(r'filename=["\']?([^"\';]+)', cd)
                if m:
                    filename = m.group(1)
            # fallback ensure extension present
            if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                for ext in ALLOWED_EXTENSIONS:
                    if ext.strip('.') in ctype:
                        filename += ext
                        break
            outpath = os.path.join(DOWNLOAD_DIR, filename)
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            downloaded.append((outpath, url, source))
            logging.info(f"Saved: {outpath}")
        except Exception as e:
            logging.warning(f"Failed to download {url}: {e}")
    return downloaded

# === zip ===

def make_zip(downloaded_files, zipname=ZIP_FILENAME):
    if os.path.exists(zipname):
        os.remove(zipname)
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for path, url, src in downloaded_files:
            arcname = os.path.relpath(path, DOWNLOAD_DIR)
            z.write(path, arcname=os.path.join("policies", arcname))
    logging.info(f"Created ZIP: {zipname}")

# === main ===

def main():
    logging.info("Starting MOE policy scraper")
    doc_map = crawl_and_collect(SEED_URLS)
    if not doc_map:
        logging.error("No documents found. Exiting.")
        return
    downloaded = download_documents(doc_map)
    if not downloaded:
        logging.error("No documents downloaded. Exiting.")
        return
    make_zip(downloaded)
    logging.info("All done. Files in './downloads', ZIP: %s", ZIP_FILENAME)

if __name__ == "__main__":
    main()
