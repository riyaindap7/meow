#!/usr/bin/env python3
"""
download_moe_smart_policy_pdfs.py

Upgraded scraper that:
- Crawls MOE India + related departments
- Finds PDF links
- HEAD-checks PDF metadata
- Range-downloads only first 64 KB
- Extracts partial text
- Scores each PDF for policy/amendment relevance
- Downloads ONLY high-scoring PDFs
- Packages all results into a ZIP

Requirements:
    pip install requests beautifulsoup4 pypdf
"""

import os
import re
import time
import logging
import zipfile
import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from collections import deque
from pypdf import PdfReader
from io import BytesIO

# ======================================
# CONFIG
# ======================================

DOMAIN_WHITELIST = {
    "education.gov.in",
    "mhrd.gov.in",
    "ncert.nic.in",
    "ugc.ac.in",
    "deb.ugc.ac.in",
    "ncte.gov.in",
    "aicte-india.org",
}

SEED_URLS = [
    "https://www.education.gov.in/en/policy_initiatives",
    "https://www.education.gov.in/en/national-education-policy",
    "https://ncert.nic.in/",
    "https://ugc.ac.in/",
    "https://deb.ugc.ac.in/",
    "https://ncte.gov.in/",
    "https://www.aicte-india.org/",
]

DOWNLOAD_DIR = "downloads"
ZIP_FILENAME = "moe_policy_pdfs_smart.zip"
USER_AGENT = "MOE-Smart-Policy-Scraper/2.0"
RATE_LIMIT = 0.8
MAX_PAGES = 2000
SAMPLE_BYTES = 65536  # 64 KB
SCORE_THRESHOLD = 6   # <=== only download if score >= this

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ======================================
# KEYWORD RULES FOR SCORING
# ======================================

KEYWORD_SCORES = {
    5: ["amendment", "corrigendum", "revision", "updated", "update"],
    4: ["time series", "monthly", "quarterly", "annual"],
    3: ["table", "statistics", "dataset", "series", "annex"],
    2: ["2021", "2022", "2023", "2024", "2025"],  # date signals
}

NEGATIVE_PATTERNS = [
    "logo", "image", "icon", "media", "form", "template",
]

# ======================================
# HELPERS
# ======================================

def domain_allowed(url):
    return urlparse(url).netloc.lower().split(":")[0] in DOMAIN_WHITELIST

def sanitize_filename(url):
    name = os.path.basename(urlparse(url).path)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name if name.endswith(".pdf") else name + ".pdf"

def is_pdf_url(url):
    return url.lower().endswith(".pdf")

def obeys_robots(url, cache):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in cache:
        rp = RobotFileParser()
        rp.set_url(urljoin(base, "/robots.txt"))
        try:
            rp.read()
            cache[base] = rp
        except:
            cache[base] = None
    rp = cache.get(base)
    return True if rp is None else rp.can_fetch(USER_AGENT, url)

def score_pdf(sample_text, url):
    text = sample_text.lower()

    score = 0

    # Positive signals
    for val, keywords in KEYWORD_SCORES.items():
        for kw in keywords:
            if kw in text:
                score += val

    # Table-like detection
    numbers = len(re.findall(r"\d", text))
    if numbers > 30:
        score += 3

    # Date patterns
    if re.search(r"\b(20[12]\d)\b", text):
        score += 2

    # Negative signals
    for bad in NEGATIVE_PATTERNS:
        if bad in url.lower():
            score -= 5

    return score

# ======================================
# SMART PDF SCANNER
# ======================================

def evaluate_pdf(url, session):
    """HEAD → Range sample → extract → score"""
    try:
        head = session.head(url, allow_redirects=True, timeout=20)
        ctype = head.headers.get("Content-Type", "")
        size = int(head.headers.get("Content-Length", "0"))

        if "pdf" not in ctype.lower():
            return None, -999  # Not a real PDF

        if size < 10_000:
            return None, -10  # too small

        # Range-request sample
        r = session.get(url, headers={"Range": f"bytes=0-{SAMPLE_BYTES}"}, timeout=20)
        sample_bytes = r.content[:SAMPLE_BYTES]

        # Extract text
        text = ""
        try:
            reader = PdfReader(BytesIO(sample_bytes))
            first_page = reader.pages[0]
            text = first_page.extract_text() or ""
        except:
            # Fallback: raw text heuristics
            text = sample_bytes.decode("latin-1", "ignore")

        score = score_pdf(text, url)
        return sample_bytes, score

    except Exception as e:
        logging.warning(f"Error sampling {url}: {e}")
        return None, -999

# ======================================
# FULL DOWNLOAD
# ======================================

def download_pdf(url, session):
    try:
        logging.info(f"Downloading full PDF: {url}")
        r = session.get(url, stream=True, timeout=60)
        filepath = os.path.join(DOWNLOAD_DIR, sanitize_filename(url))

        with open(filepath, "wb") as f:
            for chunk in r.iter_content(16384):
                f.write(chunk)

        return filepath
    except Exception as e:
        logging.warning(f"Failed to download {url}: {e}")
        return None

# ======================================
# CRAWLER
# ======================================

def crawl_for_pdf_links(seed_urls):
    visited = set()
    queue = deque(seed_urls)
    robots_cache = {}
    pdf_urls = set()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    count = 0

    while queue and count < MAX_PAGES:
        url = queue.popleft()
        url, _ = urldefrag(url)

        if url in visited or not domain_allowed(url) or not obeys_robots(url, robots_cache):
            continue

        visited.add(url)
        count += 1

        try:
            logging.info(f"Fetching: {url}")
            r = session.get(url, timeout=20)
            time.sleep(RATE_LIMIT)
        except:
            continue

        ctype = r.headers.get("Content-Type", "")

        if "html" not in ctype:
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # Extract links
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            link, _ = urldefrag(link)

            if is_pdf_url(link) and domain_allowed(link):
                pdf_urls.add(link)
            elif domain_allowed(link) and link not in visited:
                queue.append(link)

    return pdf_urls

# ======================================
# MAIN LOGIC
# ======================================

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    logging.info("Starting smart policy scraper...")

    pdf_links = crawl_for_pdf_links(SEED_URLS)
    logging.info(f"PDF candidates found: {len(pdf_links)}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    downloaded = []

    for url in pdf_links:
        sample, score = evaluate_pdf(url, session)
        logging.info(f"Score {score:>3} → {url}")

        if score >= SCORE_THRESHOLD:
            f = download_pdf(url, session)
            if f:
                downloaded.append(f)

    # Build ZIP
    with zipfile.ZipFile(ZIP_FILENAME, "w", zipfile.ZIP_DEFLATED) as z:
        for f in downloaded:
            z.write(f, os.path.join("policy_pdfs", os.path.basename(f)))

    logging.info(f"Finished! Relevant PDFs downloaded: {len(downloaded)}")
    logging.info(f"ZIP created: {ZIP_FILENAME}")

if __name__ == "__main__":
    main()
