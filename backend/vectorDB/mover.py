"""
Unique categories found (5):
  - Scraped_moe_archived_press_releases
  - Scraped_moe_archived_scholarships
  - moe_scraped_higher_edu_RUSA
  - scraped_moe_archived_circulars
  - scraped_moe_documents&reports
"""
import json
import re
import os

def extract_category(source_file):
    """
    Extract category from source_file path.
    Expected format: outputs//<<category>>/<<filename>> or outputs/<<category>>/<<filename>>
    """
    if not source_file:
        return "unknown"
    
    # Normalize path separators
    normalized_path = source_file.replace("\\", "/")
    
    # Pattern to match outputs//category/filename or outputs/category/filename
    patterns = [
        r'outputs//([^/]+)/',  # outputs//category/
        r'outputs/([^/]+)/',   # outputs/category/
    ]
    
    for pattern in patterns:
        match = re.search(pattern, normalized_path)
        if match:
            return match.group(1)
    
    return "unknown"

def process_json_file(filepath):
    """
    Process a JSON file and add Category field to each entry.
    Returns: (updated_entries_count, unique_categories)
    """
    print(f"\nProcessing: {filepath}")
    
    # Read the JSON file
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"Error: Expected a list in {filepath}")
        return 0, set()
    
    categories = set()
    updated_count = 0
    
    for entry in data:
        if isinstance(entry, dict):
            source_file = entry.get('source_file', '')
            category = extract_category(source_file)
            entry['Category'] = category
            categories.add(category)
            updated_count += 1
    
    # Write back to the file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return updated_count, categories

def main():
    # Define the file paths
    base_path = r"c:\Users\tanma\Desktop\projects\meow\backend\vectorDB\embeddings_consolidated"
    
    files_to_process = [
        os.path.join(base_path, "all_text_chunks.json"),
        os.path.join(base_path, "all_table_chunks.json")
    ]
    
    all_categories = set()
    total_entries_updated = 0
    
    print("=" * 60)
    print("Adding Category field to JSON files")
    print("=" * 60)
    
    for filepath in files_to_process:
        if os.path.exists(filepath):
            updated_count, categories = process_json_file(filepath)
            all_categories.update(categories)
            total_entries_updated += updated_count
            
            print(f"  - Entries updated: {updated_count}")
            print(f"  - Categories found: {sorted(categories)}")
        else:
            print(f"\nFile not found: {filepath}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal entries updated: {total_entries_updated}")
    print(f"\nUnique categories found ({len(all_categories)}):")
    for cat in sorted(all_categories):
        print(f"  - {cat}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
