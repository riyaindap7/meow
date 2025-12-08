# from pathlib import Path
# import numpy as np
# import json
# from pymilvus import (
#     connections, FieldSchema, CollectionSchema,
#     DataType, Collection, utility, MilvusException
# )
# import time
# from tqdm import tqdm

# MILVUS_HOST = "localhost"
# MILVUS_PORT = "19530"
# EMBEDDINGS_DIR = Path("embeddings_local")
# CHUNKED_OUTPUTS_DIR = Path("chunked_outputs")

# def connect_milvus(retries=15, delay=6):
#     """Connect to Milvus with retry logic"""
#     for i in range(1, retries + 1):
#         try:
#             connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)
#             print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
#             return
#         except MilvusException as e:
#             print(f"⚠️ Retry {i}/{retries}: {e}")
#             time.sleep(delay)
#     raise SystemExit("❌ Failed: Milvus not reachable.")

# def drop_if_exists(name: str):
#     """Drop collection if it exists"""
#     if utility.has_collection(name):
#         utility.drop_collection(name)
#         print(f"🗑️ Dropped existing collection: {name}")

# def create_collection(name: str, dim: int, extra_fields: list):
#     """Create a new Milvus collection"""
#     fields = [
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
#     ] + extra_fields
#     schema = CollectionSchema(fields, description=f"{name} collection")
#     col = Collection(name, schema)
#     print(f"✅ Created collection: {name}")
#     return col

# def build_hnsw(col: Collection):
#     """Build HNSW index with Inner Product metric"""
#     index_params = {
#         "metric_type": "IP",
#         "index_type": "HNSW",
#         "params": {"M": 16, "efConstruction": 200}
#     }
#     col.create_index(field_name="embedding", index_params=index_params)
#     print(f"🔍 HNSW index created (IP) for {col.name}")
#     col.load()
#     print(f"📂 Loaded collection: {col.name}")

# def load_chunks_from_json(document_name: str):
#     """Load complete text chunks from chunked_outputs JSON files"""
#     # Try to find the text JSON file in chunked_outputs
#     possible_paths = [
#         CHUNKED_OUTPUTS_DIR / document_name / document_name / f"{document_name}_text.json",
#         CHUNKED_OUTPUTS_DIR / document_name / f"{document_name}_text.json",
#     ]
    
#     for json_path in possible_paths:
#         if json_path.exists():
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 chunks = json.load(f)
#             print(f"   📄 Loaded {len(chunks)} chunks from {json_path.name}")
#             return chunks
    
#     print(f"   ⚠️ No text chunks JSON found for {document_name}")
#     return []

# def align_chunks_with_embeddings(chunks, embeddings, metadata_json):
#     """
#     Align chunks with embeddings based on chunk_index order.
#     Ensures that chunk[i] corresponds to embedding[i].
#     """
#     # Get the order from text_metadata.json
#     metadata_chunks = metadata_json.get('chunks', [])
    
#     # Create a mapping: chunk_index -> chunk data
#     chunks_dict = {chunk.get('chunk_index', idx): chunk for idx, chunk in enumerate(chunks)}
    
#     # Sort chunks by chunk_index to match embedding order
#     sorted_chunks = []
#     for idx in range(len(embeddings)):
#         if idx in chunks_dict:
#             sorted_chunks.append(chunks_dict[idx])
#         else:
#             # If chunk not found, create a placeholder
#             print(f"   ⚠️ Warning: Missing chunk at index {idx}")
#             sorted_chunks.append({
#                 'chunk_index': idx,
#                 'text': '[MISSING CHUNK]',
#                 'chunk_id': f'missing_{idx}',
#                 'global_chunk_id': f'missing_{idx}',
#                 'page_idx': 0,
#                 'section_hierarchy': '',
#                 'heading_context': '',
#                 'char_count': 0,
#                 'word_count': 0
#             })
    
#     return sorted_chunks

# def insert_victor_text():
#     """Insert text embeddings with complete text from chunked_outputs"""
#     print("\n" + "="*60)
#     print("📝 Inserting VictorText Collection")
#     print("="*60)
    
#     if not EMBEDDINGS_DIR.exists():
#         print(f"❌ Directory not found: {EMBEDDINGS_DIR}")
#         return
    
#     # Collect all embeddings and metadata
#     all_embeddings = []
#     all_metadata = []
    
#     # Scan all document folders
#     doc_folders = [d for d in EMBEDDINGS_DIR.iterdir() if d.is_dir()]
#     print(f"📁 Found {len(doc_folders)} document folders")
    
#     for doc_folder in tqdm(doc_folders, desc="Loading documents"):
#         emb_file = doc_folder / "text_embeddings.npy"
#         meta_file = doc_folder / "text_metadata.json"
        
#         if not emb_file.exists() or not meta_file.exists():
#             print(f"\n⚠️ Skipping {doc_folder.name} (missing files)")
#             continue
        
#         # Load embeddings
#         embeddings = np.load(emb_file)
        
#         # Load metadata (for document_id and count validation)
#         with open(meta_file, 'r', encoding='utf-8') as f:
#             metadata_json = json.load(f)
        
#         document_id = metadata_json['document_id']
        
#         # Load complete text chunks from JSON files
#         chunks = load_chunks_from_json(doc_folder.name)
        
#         if not chunks:
#             print(f"\n❌ No chunks found for {doc_folder.name}. Skipping.")
#             continue
        
#         # Validate chunk count matches embeddings
#         if len(chunks) != embeddings.shape[0]:
#             print(f"\n⚠️ Mismatch in {doc_folder.name}: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks")
#             print(f"   Attempting to align chunks...")
        
#         # ✅ CRITICAL FIX: Align chunks with embeddings by chunk_index
#         aligned_chunks = align_chunks_with_embeddings(chunks, embeddings, metadata_json)
        
#         # Final validation
#         if len(aligned_chunks) != embeddings.shape[0]:
#             print(f"\n❌ Cannot align chunks for {doc_folder.name}. Skipping.")
#             continue
        
#         # Store embeddings
#         all_embeddings.append(embeddings)
        
#         # Store metadata with COMPLETE text in correct order
#         for idx, chunk in enumerate(aligned_chunks):
#             # Verify chunk_index matches position
#             expected_index = chunk.get('chunk_index', idx)
#             if expected_index != idx:
#                 print(f"\n⚠️ Warning: chunk_index mismatch at position {idx} (expected {expected_index})")
            
#             all_metadata.append({
#                 'document_name': doc_folder.name,
#                 'document_id': document_id,
#                 'chunk_id': chunk.get('chunk_id', f'{document_id}_text_{idx}'),
#                 'global_chunk_id': chunk.get('global_chunk_id', f'text_{document_id}_{idx}'),
#                 'page_idx': chunk.get('page_idx', 0),
#                 'chunk_index': idx,
#                 'section_hierarchy': chunk.get('section_hierarchy', ''),
#                 'heading_context': chunk.get('heading_context', ''),
#                 'text': chunk.get('text', ''),
#                 'char_count': chunk.get('char_count', 0),
#                 'word_count': chunk.get('word_count', 0)
#             })
    
#     if not all_embeddings:
#         print("❌ No valid embeddings found")
#         return
    
#     # Concatenate all embeddings
#     embeddings_array = np.vstack(all_embeddings)
#     print(f"\n📊 Total embeddings: {embeddings_array.shape[0]}")
#     print(f"📏 Embedding dimension: {embeddings_array.shape[1]}")
#     print(f"📋 Total metadata entries: {len(all_metadata)}")
    
#     # Final sanity check
#     assert embeddings_array.shape[0] == len(all_metadata), "Embeddings and metadata count mismatch!"
    
#     # Drop existing collection
#     drop_if_exists("VictorText")
    
#     # Create collection with increased VARCHAR limits
#     col = create_collection(
#         "VictorText",
#         dim=embeddings_array.shape[1],
#         extra_fields=[
#             FieldSchema("document_name", DataType.VARCHAR, max_length=256),
#             FieldSchema("document_id", DataType.VARCHAR, max_length=256),
#             FieldSchema("chunk_id", DataType.VARCHAR, max_length=256),
#             FieldSchema("global_chunk_id", DataType.VARCHAR, max_length=256),
#             FieldSchema("page_idx", DataType.INT64),
#             FieldSchema("chunk_index", DataType.INT64),
#             FieldSchema("section_hierarchy", DataType.VARCHAR, max_length=5000),  # ✅ Increased to 5000
#             FieldSchema("heading_context", DataType.VARCHAR, max_length=5000),    # ✅ Increased to 5000
#             FieldSchema("text", DataType.VARCHAR, max_length=15000),              # ✅ Increased to 15000
#             FieldSchema("char_count", DataType.INT64),
#             FieldSchema("word_count", DataType.INT64),
#         ]
#     )
    
#     # Prepare data for insertion with proper truncation
#     data = [
#         embeddings_array.tolist(),
#         [m['document_name'][:256] for m in all_metadata],
#         [m['document_id'][:256] for m in all_metadata],
#         [m['chunk_id'][:256] for m in all_metadata],
#         [m['global_chunk_id'][:256] for m in all_metadata],
#         [m['page_idx'] for m in all_metadata],
#         [m['chunk_index'] for m in all_metadata],
#         [str(m['section_hierarchy'])[:5000] for m in all_metadata],  # ✅ Truncate to 5000
#         [str(m['heading_context'])[:5000] for m in all_metadata],    # ✅ Truncate to 5000
#         [str(m['text'])[:15000] for m in all_metadata],              # ✅ Truncate to 15000
#         [m['char_count'] for m in all_metadata],
#         [m['word_count'] for m in all_metadata],
#     ]
    
#     # Insert data in batches to avoid memory issues
#     batch_size = 1000
#     total_inserted = 0
    
#     print("\n💾 Inserting data into Milvus...")
#     for i in tqdm(range(0, len(data[0]), batch_size), desc="Inserting batches"):
#         batch_data = [d[i:i+batch_size] for d in data]
#         col.insert(batch_data)
#         total_inserted += len(batch_data[0])
    
#     col.flush()
#     print(f"✅ Inserted {col.num_entities} text chunk vectors with complete text")
    
#     # Build HNSW index
#     build_hnsw(col)

# def main():
#     """Main execution function"""
#     print("\n" + "="*60)
#     print("🚀 Milvus VictorText Collection Creator")
#     print("="*60)
    
#     connect_milvus()
#     insert_victor_text()
    
#     print("\n" + "="*60)
#     print("✅ Done ingesting VictorText collection")
#     print("="*60)

# if __name__ == "__main__":
#     main()

from pathlib import Path
import numpy as np
import json
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility, MilvusException
)
import time
from tqdm import tqdm

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
EMBEDDINGS_DIR = Path("embeddings_consolidated")  # ✅ Changed to consolidated embeddings
CHUNKED_OUTPUTS_DIR = Path("chunked_outputs_v2")  # ✅ Changed to consolidated chunks

def connect_milvus(retries=15, delay=6):
    """Connect to Milvus with retry logic"""
    for i in range(1, retries + 1):
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)
            print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
            return
        except MilvusException as e:
            print(f"⚠️ Retry {i}/{retries}: {e}")
            time.sleep(delay)
    raise SystemExit("❌ Failed: Milvus not reachable.")

def drop_if_exists(name: str):
    """Drop collection if it exists"""
    if utility.has_collection(name):
        utility.drop_collection(name)
        print(f"🗑️ Dropped existing collection: {name}")

def create_collection(name: str, dim: int, extra_fields: list):
    """Create a new Milvus collection"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ] + extra_fields
    schema = CollectionSchema(fields, description=f"{name} collection")
    col = Collection(name, schema)
    print(f"✅ Created collection: {name}")
    return col

def build_hnsw(col: Collection):
    """Build HNSW index with Inner Product metric"""
    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index(field_name="embedding", index_params=index_params)
    print(f"🔍 HNSW index created (IP) for {col.name}")
    col.load()
    print(f"📂 Loaded collection: {col.name}")

def insert_vtext():
    """Insert text embeddings from consolidated files into Vtext collection"""
    print("\n" + "="*60)
    print("📝 Inserting Vtext Collection")
    print("="*60)
    
    # Load embeddings
    emb_file = EMBEDDINGS_DIR / "all_text_embeddings.npy"
    chunks_file = CHUNKED_OUTPUTS_DIR / "all_text_chunks.json"
    metadata_file = EMBEDDINGS_DIR / "all_text_chunks.json"
    
    if not emb_file.exists():
        print(f"❌ Embeddings file not found: {emb_file}")
        return
    
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    print(f"📂 Loading embeddings from: {emb_file}")
    embeddings = np.load(emb_file)
    
    print(f"📂 Loading chunks from: {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Load metadata for validation
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"📊 Metadata loaded: {metadata.get('total_chunks', 0)} chunks")
    
    print(f"📊 Total embeddings: {embeddings.shape[0]}")
    print(f"📏 Embedding dimension: {embeddings.shape[1]}")
    print(f"📋 Total chunks: {len(chunks)}")
    
    # Validate counts
    if embeddings.shape[0] != len(chunks):
        print(f"⚠️ Warning: Mismatch between embeddings ({embeddings.shape[0]}) and chunks ({len(chunks)})")
        min_count = min(embeddings.shape[0], len(chunks))
        print(f"   Using first {min_count} entries")
        embeddings = embeddings[:min_count]
        chunks = chunks[:min_count]
    
    # Drop existing collection
    drop_if_exists("Vtext")
    
    # Create collection
    col = create_collection(
        "Vtext",
        dim=embeddings.shape[1],
        extra_fields=[
            FieldSchema("global_chunk_id", DataType.VARCHAR, max_length=256),
            FieldSchema("document_id", DataType.VARCHAR, max_length=256),
            FieldSchema("source_file", DataType.VARCHAR, max_length=500),
            FieldSchema("page_idx", DataType.INT64),
            FieldSchema("chunk_index", DataType.INT64),
            FieldSchema("section_hierarchy", DataType.VARCHAR, max_length=20000),
            FieldSchema("text", DataType.VARCHAR, max_length=22000),
            FieldSchema("char_count", DataType.INT64),
            FieldSchema("word_count", DataType.INT64),
        ]
    )
    
    # ✅ Truncation tracking
    truncation_stats = {
        "global_chunk_id": {"count": 0, "max_original": 0, "samples": []},
        "document_id": {"count": 0, "max_original": 0, "samples": []},
        "source_file": {"count": 0, "max_original": 0, "samples": []},
        "section_hierarchy": {"count": 0, "max_original": 0, "samples": []},
        "text": {"count": 0, "max_original": 0, "samples": []},
    }
    
    def safe_truncate(text: str, max_length: int, field_name: str, chunk_index: int = -1) -> str:
        """Safely truncate text to max length and track statistics"""
        if text is None:
            return ""
        text_str = str(text)
        original_length = len(text_str)
        
        if original_length > max_length:
            # Track truncation
            truncation_stats[field_name]["count"] += 1
            truncation_stats[field_name]["max_original"] = max(
                truncation_stats[field_name]["max_original"], 
                original_length
            )
            
            # Store sample (first 5 occurrences)
            if len(truncation_stats[field_name]["samples"]) < 5:
                truncation_stats[field_name]["samples"].append({
                    "chunk_index": chunk_index,
                    "original_length": original_length,
                    "truncated_length": max_length,
                    "preview": text_str[:100] + "..." if len(text_str) > 100 else text_str
                })
            
            # Print immediate warning
            print(f"⚠️  Truncating {field_name} at index {chunk_index}: {original_length} → {max_length} chars")
            
            return text_str[:max_length-3] + "..."
        return text_str
    
    # Prepare data for insertion with safe truncation
    print("\n💾 Preparing data for insertion...")
    data = [
        embeddings.tolist(),
        [safe_truncate(chunk.get('global_chunk_id', ''), 256, 'global_chunk_id', i) for i, chunk in enumerate(chunks)],
        [safe_truncate(chunk.get('document_id', ''), 256, 'document_id', i) for i, chunk in enumerate(chunks)],
        [safe_truncate(chunk.get('source_file', ''), 500, 'source_file', i) for i, chunk in enumerate(chunks)],
        [int(chunk.get('page_idx', 0)) for chunk in chunks],
        [int(chunk.get('chunk_index', 0)) for chunk in chunks],
        [safe_truncate(chunk.get('section_hierarchy', ''), 20000, 'section_hierarchy', i) for i, chunk in enumerate(chunks)],
        [safe_truncate(chunk.get('text', ''), 22000, 'text', i) for i, chunk in enumerate(chunks)],
        [int(chunk.get('char_count', 0)) for chunk in chunks],
        [int(chunk.get('word_count', 0)) for chunk in chunks],
    ]
    
    # Insert data in batches
    batch_size = 1000
    total_inserted = 0
    
    print("\n💾 Inserting data into Milvus...")
    for i in tqdm(range(0, len(data[0]), batch_size), desc="Inserting batches"):
        batch_data = [d[i:i+batch_size] for d in data]
        try:
            col.insert(batch_data)
            total_inserted += len(batch_data[0])
        except Exception as e:
            print(f"\n❌ Error inserting batch {i//batch_size}: {e}")
            raise
    
    col.flush()
    print(f"✅ Inserted {col.num_entities} text chunk vectors")
    
    # Print truncation summary
    print("\n" + "="*60)
    print("📊 TRUNCATION SUMMARY")
    print("="*60)
    
    total_truncations = sum(stats["count"] for stats in truncation_stats.values())
    
    if total_truncations == 0:
        print("✅ No fields were truncated")
    else:
        print(f"⚠️  Total truncations: {total_truncations}\n")
        
        for field_name, stats in truncation_stats.items():
            if stats["count"] > 0:
                print(f"📝 {field_name}:")
                print(f"   Truncated: {stats['count']} times")
                print(f"   Max original length: {stats['max_original']} chars")
                
                if stats["samples"]:
                    print(f"   Samples:")
                    for sample in stats["samples"]:
                        print(f"      - Index {sample['chunk_index']}: {sample['original_length']} → {sample['truncated_length']} chars")
                        print(f"        Preview: {sample['preview']}")
                print()
    
    print("="*60)
    
    # Build HNSW index
    build_hnsw(col)


def insert_vtable():
    """Insert table embeddings from consolidated files into VTable collection"""
    print("\n" + "="*60)
    print("📊 Inserting VTable Collection")
    print("="*60)
    
    # Load embeddings
    emb_file = EMBEDDINGS_DIR / "all_table_embeddings.npy"
    chunks_file = CHUNKED_OUTPUTS_DIR / "all_table_chunks.json"
    metadata_file = EMBEDDINGS_DIR / "all_table_chunks.json"
    
    if not emb_file.exists():
        print(f"❌ Embeddings file not found: {emb_file}")
        return
    
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    print(f"📂 Loading embeddings from: {emb_file}")
    embeddings = np.load(emb_file)
    
    print(f"📂 Loading chunks from: {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Load metadata for validation
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"📊 Metadata loaded: {metadata.get('total_chunks', 0)} chunks")
    
    print(f"📊 Total embeddings: {embeddings.shape[0]}")
    print(f"📏 Embedding dimension: {embeddings.shape[1]}")
    print(f"📋 Total chunks: {len(chunks)}")
    
    # Validate counts
    if embeddings.shape[0] != len(chunks):
        print(f"⚠️ Warning: Mismatch between embeddings ({embeddings.shape[0]}) and chunks ({len(chunks)})")
        min_count = min(embeddings.shape[0], len(chunks))
        print(f"   Using first {min_count} entries")
        embeddings = embeddings[:min_count]
        chunks = chunks[:min_count]
    
    # Drop existing collection
    drop_if_exists("VTable")
    
    # Create collection
    col = create_collection(
        "VTable",
        dim=embeddings.shape[1],
        extra_fields=[
            FieldSchema("global_chunk_id", DataType.VARCHAR, max_length=256),
            FieldSchema("document_id", DataType.VARCHAR, max_length=256),
            FieldSchema("source_file", DataType.VARCHAR, max_length=500),
            FieldSchema("page_idx", DataType.INT64),
            FieldSchema("chunk_index", DataType.INT64),
            FieldSchema("table_text", DataType.VARCHAR, max_length=50000),
            FieldSchema("caption", DataType.VARCHAR, max_length=40000),
            FieldSchema("num_rows", DataType.INT64),
            FieldSchema("num_cols", DataType.INT64),
        ]
    )
    
    # ✅ Truncation tracking
    truncation_stats = {
        "global_chunk_id": {"count": 0, "max_original": 0, "samples": []},
        "document_id": {"count": 0, "max_original": 0, "samples": []},
        "source_file": {"count": 0, "max_original": 0, "samples": []},
        "table_text": {"count": 0, "max_original": 0, "samples": []},
        "caption": {"count": 0, "max_original": 0, "samples": []},
    }
    
    def safe_truncate(text: str, max_length: int, field_name: str, chunk_index: int = -1) -> str:
        """Safely truncate text to max length and track statistics"""
        if text is None:
            return ""
        text_str = str(text)
        original_length = len(text_str)
        
        if original_length > max_length:
            # Track truncation
            truncation_stats[field_name]["count"] += 1
            truncation_stats[field_name]["max_original"] = max(
                truncation_stats[field_name]["max_original"], 
                original_length
            )
            
            # Store sample (first 5 occurrences)
            if len(truncation_stats[field_name]["samples"]) < 5:
                truncation_stats[field_name]["samples"].append({
                    "chunk_index": chunk_index,
                    "original_length": original_length,
                    "truncated_length": max_length,
                    "preview": text_str[:100] + "..." if len(text_str) > 100 else text_str
                })
            
            # Print immediate warning
            print(f"⚠️  Truncating {field_name} at index {chunk_index}: {original_length} → {max_length} chars")
            
            return text_str[:max_length-3] + "..."
        return text_str
    
    # Prepare data for insertion with safe truncation
    print("\n💾 Preparing data for insertion...")
    data = [
        embeddings.tolist(),
        [safe_truncate(chunk.get('global_chunk_id', ''), 256, 'global_chunk_id', i) for i, chunk in enumerate(chunks)],
        [safe_truncate(chunk.get('document_id', ''), 256, 'document_id', i) for i, chunk in enumerate(chunks)],
        [safe_truncate(chunk.get('source_file', ''), 500, 'source_file', i) for i, chunk in enumerate(chunks)],
        [int(chunk.get('page_idx', 0)) for chunk in chunks],
        [int(chunk.get('chunk_index', 0)) for chunk in chunks],
        [safe_truncate(chunk.get('table_text', ''), 50000, 'table_text', i) for i, chunk in enumerate(chunks)],
        [safe_truncate(chunk.get('caption', ''), 40000, 'caption', i) for i, chunk in enumerate(chunks)],
        [int(chunk.get('table_data', {}).get('num_rows', 0)) for chunk in chunks],
        [int(chunk.get('table_data', {}).get('num_columns', 0)) for chunk in chunks],
    ]
    
    # Insert data in batches
    batch_size = 500  # Smaller batch size for tables
    total_inserted = 0
    
    print("\n💾 Inserting data into Milvus...")
    for i in tqdm(range(0, len(data[0]), batch_size), desc="Inserting batches"):
        batch_data = [d[i:i+batch_size] for d in data]
        try:
            col.insert(batch_data)
            total_inserted += len(batch_data[0])
        except Exception as e:
            print(f"\n❌ Error inserting batch {i//batch_size}: {e}")
            raise
    
    col.flush()
    print(f"✅ Inserted {col.num_entities} table chunk vectors")
    
    # Print truncation summary
    print("\n" + "="*60)
    print("📊 TRUNCATION SUMMARY")
    print("="*60)
    
    total_truncations = sum(stats["count"] for stats in truncation_stats.values())
    
    if total_truncations == 0:
        print("✅ No fields were truncated")
    else:
        print(f"⚠️  Total truncations: {total_truncations}\n")
        
        for field_name, stats in truncation_stats.items():
            if stats["count"] > 0:
                print(f"📝 {field_name}:")
                print(f"   Truncated: {stats['count']} times")
                print(f"   Max original length: {stats['max_original']} chars")
                
                if stats["samples"]:
                    print(f"   Samples:")
                    for sample in stats["samples"]:
                        print(f"      - Index {sample['chunk_index']}: {sample['original_length']} → {sample['truncated_length']} chars")
                        print(f"        Preview: {sample['preview']}")
                print()
    
    print("="*60)
    
    # Build HNSW index
    build_hnsw(col)

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🚀 Milvus Vtext & VTable Collection Creator")
    print("="*60)
    
    connect_milvus()
    
    # Insert text collection
    insert_vtext()
    
    # Insert table collection
    insert_vtable()
    
    print("\n" + "="*60)
    print("✅ Done ingesting Vtext and VTable collections")
    print("="*60)

if __name__ == "__main__":
    main()