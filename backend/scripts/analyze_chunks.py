"""
Script to analyze all_text_chunks.json for unique ministries and source references

Usage:
    python backend/scripts/analyze_chunks.py
"""  

import sys
from pathlib import Path
import json
import re
from collections import Counter
from typing import Set, List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def extract_ministries_from_text(text: str) -> Set[str]:
    """Extract ministry/organization names from text using regex patterns"""
    ministries = set()
    
    # Patterns for common government organizations
    patterns = [
        # Ministry patterns
        r"Ministry\s+of\s+[A-Z][A-Za-z\s,&()-]+(?:and\s+[A-Z][A-Za-z\s,&()-]+)?",
        r"(?:Ministry|Department|Board|Commission|Council|Authority|Institute|Bureau|Directorate)\s+(?:of|for)\s+[A-Z][A-Za-z\s,&()-]+",
        
        # Department patterns
        r"Department\s+of\s+[A-Z][A-Za-z\s,&()-]+(?:and\s+[A-Z][A-Za-z\s,&()-]+)?",
        
        # Commission patterns
        r"[A-Z][A-Za-z\s]+(?:Commission|Board|Council|Authority|Corporation)",
        
        # Special cases
        r"UGC",
        r"AICTE",
        r"NAAC",
        r"NBA",
        r"NITI\s+Aayog",
        r"Planning\s+Commission",
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            ministry_name = match.group(0).strip()
            # Clean up
            ministry_name = re.sub(r'\s+', ' ', ministry_name)
            # Only add if it's substantial
            if len(ministry_name) > 5:
                ministries.add(ministry_name)
    
    return ministries


def analyze_chunks(chunks_path: Path) -> Dict:
    """
    Analyze all_text_chunks.json for unique ministries and sources
    
    Args:
        chunks_path: Path to all_text_chunks.json
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {chunks_path.name}")
    print(f"{'='*80}\n")
    
    # Initialize counters
    all_ministries = set()
    all_sources = set()
    ministry_counter = Counter()
    source_counter = Counter()
    
    total_chunks = 0
    chunks_with_ministries = 0
    
    # Read and process file in streaming mode (for large files)
    print("⏳ Loading chunks (this may take a moment for large files)...\n")
    
    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle different JSON structures
            chunks = []
            if isinstance(data, list):
                chunks = data
            elif isinstance(data, dict):
                # Try common keys
                for key in ['chunks', 'text_chunks', 'data']:
                    if key in data:
                        chunks = data[key]
                        break
                else:
                    # If it's a dict of chunks, get all values
                    chunks = list(data.values())
            
            total_chunks = len(chunks)
            print(f"📄 Total chunks loaded: {total_chunks:,}\n")
            
            # Process each chunk
            print("🔍 Analyzing chunks...")
            for i, chunk in enumerate(chunks):
                if (i + 1) % 10000 == 0:
                    print(f"   Processed {i+1:,}/{total_chunks:,} chunks...")
                
                # Extract text content
                text = ""
                if isinstance(chunk, dict):
                    # Try different possible text fields
                    text = chunk.get('text') or chunk.get('content') or chunk.get('chunk_text') or ""
                    
                    # Extract source/document name
                    source = (
                        chunk.get('document_name') or 
                        chunk.get('source') or 
                        chunk.get('file_name') or 
                        chunk.get('document_id') or 
                        "Unknown"
                    )
                    all_sources.add(source)
                    source_counter[source] += 1
                    
                elif isinstance(chunk, str):
                    text = chunk
                
                # Extract ministries from text
                if text:
                    ministries = extract_ministries_from_text(text)
                    if ministries:
                        chunks_with_ministries += 1
                        all_ministries.update(ministries)
                        for ministry in ministries:
                            ministry_counter[ministry] += 1
            
            print(f"   ✅ Processed all {total_chunks:,} chunks!\n")
    
    except json.JSONDecodeError as e:
        print(f"❌ Error reading JSON: {e}")
        return {}
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 ANALYSIS RESULTS")
    print(f"{'='*80}\n")
    
    print(f"📄 Total Chunks: {total_chunks:,}")
    print(f"📚 Unique Sources/Documents: {len(all_sources):,}")
    print(f"🏛️  Unique Ministries/Organizations: {len(all_ministries):,}")
    print(f"📊 Chunks containing ministry mentions: {chunks_with_ministries:,} ({chunks_with_ministries/total_chunks*100:.1f}%)")
    
    # Top sources
    print(f"\n{'─'*80}")
    print(f"📚 TOP 20 SOURCES (by chunk count):")
    print(f"{'─'*80}\n")
    for source, count in source_counter.most_common(20):
        print(f"   {count:>6,} chunks | {source}")
    
    # Top ministries
    print(f"\n{'─'*80}")
    print(f"🏛️  TOP 30 MINISTRIES/ORGANIZATIONS (by mentions):")
    print(f"{'─'*80}\n")
    for ministry, count in ministry_counter.most_common(30):
        print(f"   {count:>6,} mentions | {ministry}")
    
    # All unique ministries (sorted)
    print(f"\n{'─'*80}")
    print(f"🏛️  ALL UNIQUE MINISTRIES/ORGANIZATIONS ({len(all_ministries)}):")
    print(f"{'─'*80}\n")
    for i, ministry in enumerate(sorted(all_ministries), 1):
        mentions = ministry_counter[ministry]
        print(f"   {i:>3}. {ministry:<60} ({mentions:,} mentions)")
    
    # All unique sources (sorted)
    print(f"\n{'─'*80}")
    print(f"📚 ALL UNIQUE SOURCES/DOCUMENTS ({len(all_sources)}):")
    print(f"{'─'*80}\n")
    for i, source in enumerate(sorted(all_sources), 1):
        chunks = source_counter[source]
        print(f"   {i:>3}. {source:<60} ({chunks:,} chunks)")
    
    print(f"\n{'='*80}\n")
    
    # Return structured results
    return {
        "total_chunks": total_chunks,
        "unique_sources": len(all_sources),
        "unique_ministries": len(all_ministries),
        "chunks_with_ministries": chunks_with_ministries,
        "all_sources": sorted(all_sources),
        "all_ministries": sorted(all_ministries),
        "top_sources": source_counter.most_common(20),
        "top_ministries": ministry_counter.most_common(30),
    }


def save_results_to_file(results: Dict, output_path: Path):
    """Save analysis results to a JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to: {output_path}")


if __name__ == "__main__":
    # Define paths
    base_path = Path(__file__).parent.parent / "vectorDB"
    
    # Option 1: chunked_outputs_v2
    chunks_path_v2 = base_path / "chunked_outputs_v2" / "all_text_chunks.json"
    
    # Option 2: embeddings_consolidated
    chunks_path_consolidated = base_path / "embeddings_consolidated" / "all_text_chunks.json"
    
    # Check which file exists
    if chunks_path_v2.exists():
        print(f"✅ Found: {chunks_path_v2}")
        results = analyze_chunks(chunks_path_v2)
        
        # Save results
        output_path = base_path / "analysis_results.json"
        save_results_to_file(results, output_path)
        
    elif chunks_path_consolidated.exists():
        print(f"✅ Found: {chunks_path_consolidated}")
        results = analyze_chunks(chunks_path_consolidated)
        
        # Save results
        output_path = base_path / "analysis_results.json"
        save_results_to_file(results, output_path)
        
    else:
        print("❌ Error: Could not find all_text_chunks.json in either location:")
        print(f"   - {chunks_path_v2}")
        print(f"   - {chunks_path_consolidated}")
        print("\nPlease check the file path.")