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
#             print(f"⚠ Retry {i}/{retries}: {e}")
#             time.sleep(delay)
#     raise SystemExit("❌ Failed: Milvus not reachable.")

# def drop_if_exists(name: str):
#     """Drop collection if it exists"""
#     if utility.has_collection(name):
#         utility.drop_collection(name)
#         print(f"🗑 Dropped existing collection: {name}")

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
    
#     print(f"   ⚠ No text chunks JSON found for {document_name}")
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
#             print(f"   ⚠ Warning: Missing chunk at index {idx}")
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
#             print(f"\n⚠ Skipping {doc_folder.name} (missing files)")
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
#             print(f"\n⚠ Mismatch in {doc_folder.name}: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks")
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
#                 print(f"\n⚠ Warning: chunk_index mismatch at position {idx} (expected {expected_index})")
            
#             all_metadata.append({
#                 'document_name': doc_folder.name,
#                 'document_id': document_id,
#                 'chunk_id': chunk.get('chunk_id', f'{document_id}text{idx}'),
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

# if _name_ == "_main_":
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
from scipy import sparse

MILVUS_HOST = "192.168.65.160"
MILVUS_PORT = "19530"

# Updated paths - flat file structure
EMBEDDINGS_DIR = Path("embeddings_consolidated")
CHUNKED_OUTPUTS_DIR = Path("chunked_outputs_v2")
SPARSE_EMBEDDINGS_DIR = Path("sparse_embeddings")


def connect_milvus(retries=15, delay=6):
    """Connect to Milvus with retry logic"""
    for i in range(1, retries + 1):
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)
            print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
            return
        except MilvusException as e:
            print(f"⚠ Retry {i}/{retries}: {e}")
            time.sleep(delay)
    raise SystemExit("❌ Failed: Milvus not reachable.")


def drop_if_exists(name: str):
    """Drop collection if it exists"""
    if utility.has_collection(name):
        utility.drop_collection(name)
        print(f"🗑 Dropped existing collection: {name}")


def create_text_hybrid_collection(name: str, dense_dim: int):
    """Create text collection with both dense and sparse vector fields"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # Dense embedding (HNSW)
        FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        # Sparse embedding (for hybrid search)
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        # Metadata fields
        FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="global_chunk_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="page_idx", dtype=DataType.INT32),
        FieldSchema(name="chunk_index", dtype=DataType.INT32),
        FieldSchema(name="section_hierarchy", dtype=DataType.VARCHAR, max_length=65335),
        FieldSchema(name="heading_context", dtype=DataType.VARCHAR, max_length=65335),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="char_count", dtype=DataType.INT32),
        FieldSchema(name="word_count", dtype=DataType.INT32),
        # New fields
        FieldSchema(name="Category", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="ministry", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="published_date", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="source_reference", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="semantic_labels", dtype=DataType.VARCHAR, max_length=65535),  # JSON string
    ]
    schema = CollectionSchema(fields, description=f"{name} hybrid collection with dense + sparse vectors")
    col = Collection(name, schema)
    print(f"✅ Created hybrid text collection: {name}")
    return col


def create_table_hybrid_collection(name: str, dense_dim: int):
    """Create table collection with both dense and sparse vector fields"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        # Metadata fields
        FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="global_chunk_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="page_idx", dtype=DataType.INT32),
        FieldSchema(name="table_index", dtype=DataType.INT32),
        FieldSchema(name="section_hierarchy", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="heading_context", dtype=DataType.VARCHAR, max_length=65535),
        # Table-specific fields
        FieldSchema(name="table_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="table_markdown", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="table_html", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="row_count", dtype=DataType.INT32),
        FieldSchema(name="col_count", dtype=DataType.INT32),
        # New fields
        FieldSchema(name="Category", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="ministry", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="published_date", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="source_reference", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="semantic_labels", dtype=DataType.VARCHAR, max_length=65535),  # JSON string
    ]
    schema = CollectionSchema(fields, description=f"{name} hybrid collection for tables")
    col = Collection(name, schema)
    print(f"✅ Created hybrid table collection: {name}")
    return col


def build_hybrid_indexes(col: Collection):
    """Build HNSW index for dense and SPARSE_INVERTED_INDEX for sparse"""
    # HNSW index for dense vectors (IP metric)
    dense_index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index(field_name="dense_embedding", index_params=dense_index_params)
    print(f"🔍 HNSW index created (IP) for dense_embedding")
    
    # Sparse inverted index for sparse vectors (IP metric for BGE-M3)
    sparse_index_params = {
        "metric_type": "IP",
        "index_type": "SPARSE_INVERTED_INDEX",
        "params": {"drop_ratio_build": 0.2}
    }
    col.create_index(field_name="sparse_embedding", index_params=sparse_index_params)
    print(f"🔍 SPARSE_INVERTED_INDEX created (IP) for sparse_embedding")
    
    col.load()
    print(f"📂 Loaded collection: {col.name}")


def load_sparse_npz(npz_path: Path):
    """Load sparse embeddings from NPZ file and return CSR matrix + metadata"""
    print(f"📂 Loading sparse embeddings from {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    indices = data['indices']
    values = data['values']
    indptr = data['indptr']
    shape = data['shape']
    metadata = data['metadata']
    
    # Reconstruct CSR matrix
    csr_matrix = sparse.csr_matrix(
        (values, indices, indptr),
        shape=(shape[0], shape[1])
    )
    
    print(f"   ✅ Loaded {csr_matrix.shape[0]} sparse embeddings")
    print(f"   📏 Vocabulary size: {csr_matrix.shape[1]}")
    print(f"   📊 Total non-zero: {csr_matrix.nnz}")
    
    return csr_matrix, metadata


def csr_row_to_sparse_dict(csr_matrix, row_idx: int) -> dict:
    """Convert a single CSR row to Milvus sparse dict format {index: value}"""
    start = csr_matrix.indptr[row_idx]
    end = csr_matrix.indptr[row_idx + 1]
    
    indices = csr_matrix.indices[start:end]
    values = csr_matrix.data[start:end]
    
    # Milvus expects {int: float} dict for sparse vectors
    return {int(idx): float(val) for idx, val in zip(indices, values)}


def build_global_chunk_id_mapping(sparse_metadata):
    """Build mapping from global_chunk_id to sparse embedding index"""
    mapping = {}
    for idx, meta in enumerate(sparse_metadata):
        if isinstance(meta, dict):
            global_id = meta.get('global_chunk_id')
        elif hasattr(meta, 'item'):
            global_id = meta.item().get('global_chunk_id')
        else:
            global_id = None
        
        if global_id:
            mapping[global_id] = idx
    
    print(f"   📋 Built mapping for {len(mapping)} global_chunk_ids")
    return mapping


def insert_victor_text_hybrid():
    """Insert text embeddings with both dense and sparse vectors - FLAT FILE VERSION"""
    print("\n" + "="*60)
    print("📝 Inserting VictorText2 Collection (Hybrid: Dense + Sparse)")
    print("="*60)
    
    # Load dense embeddings from consolidated file
    dense_emb_path = EMBEDDINGS_DIR / "all_text_embeddings.npy"
    chunks_path = EMBEDDINGS_DIR / "all_text_chunks_enriched.json"
    sparse_npz_path = SPARSE_EMBEDDINGS_DIR / "text_sparse_embeddings.npz"
    
    # Validate files exist
    if not dense_emb_path.exists():
        print(f"❌ Dense embeddings not found: {dense_emb_path}")
        return
    if not chunks_path.exists():
        print(f"❌ Chunks JSON not found: {chunks_path}")
        return
    if not sparse_npz_path.exists():
        print(f"❌ Sparse embeddings not found: {sparse_npz_path}")
        return
    
    # Load dense embeddings
    print(f"📂 Loading dense embeddings from {dense_emb_path}")
    dense_embeddings = np.load(dense_emb_path)
    print(f"   ✅ Loaded {dense_embeddings.shape[0]} dense embeddings, dim={dense_embeddings.shape[1]}")
    
    # Load chunks metadata
    print(f"📂 Loading chunks from {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"   ✅ Loaded {len(chunks)} chunks")
    
    # Load sparse embeddings
    sparse_csr, sparse_metadata = load_sparse_npz(sparse_npz_path)
    sparse_id_mapping = build_global_chunk_id_mapping(sparse_metadata)
    
    # Validate counts match
    if len(chunks) != dense_embeddings.shape[0]:
        print(f"⚠ Warning: {dense_embeddings.shape[0]} embeddings vs {len(chunks)} chunks")
        min_count = min(len(chunks), dense_embeddings.shape[0])
        print(f"   Using first {min_count} entries")
        chunks = chunks[:min_count]
        dense_embeddings = dense_embeddings[:min_count]
    
    # Prepare data
    all_sparse_embeddings = []
    all_metadata = []
    skipped_sparse = 0
    
    print("🔄 Processing chunks...")
    for idx, chunk in enumerate(tqdm(chunks, desc="Preparing data")):
        global_chunk_id = chunk.get('global_chunk_id', f'text_unknown_{idx}')
        
        # Get sparse embedding by global_chunk_id
        sparse_idx = sparse_id_mapping.get(global_chunk_id)
        
        if sparse_idx is not None:
            sparse_dict = csr_row_to_sparse_dict(sparse_csr, sparse_idx)
        else:
            sparse_dict = {0: 0.0}  # Milvus requires at least one element
            skipped_sparse += 1
        
        all_sparse_embeddings.append(sparse_dict)
        
        # Convert semantic_labels to JSON string
        semantic_labels = chunk.get('semantic_labels', {})
        semantic_labels_str = json.dumps(semantic_labels) if isinstance(semantic_labels, dict) else str(semantic_labels)
        
        all_metadata.append({
            'document_name': str(chunk.get('document_name', chunk.get('source', 'unknown')))[:512],
            'document_id': str(chunk.get('document_id', 'unknown'))[:256],
            'chunk_id': str(chunk.get('chunk_id', f'text_{idx}'))[:256],
            'global_chunk_id': str(global_chunk_id)[:256],
            'page_idx': int(chunk.get('page_idx', chunk.get('page', 0))),
            'chunk_index': int(chunk.get('chunk_index', idx)),
            'section_hierarchy': str(chunk.get('section_hierarchy', ''))[:30000],
            'heading_context': str(chunk.get('heading_context', ''))[:30000],
            'text': str(chunk.get('text', ''))[:65535],
            'char_count': int(chunk.get('char_count', len(chunk.get('text', '')))),
            'word_count': int(chunk.get('word_count', len(chunk.get('text', '').split()))),
            # New fields
            'Category': str(chunk.get('Category', ''))[:1024],
            'document_type': str(chunk.get('document_type', ''))[:1024],
            'ministry': str(chunk.get('ministry', ''))[:1024],
            'published_date': str(chunk.get('published_date', '') or '')[:256],
            'source_reference': str(chunk.get('source_reference', ''))[:1024],
            'version': str(chunk.get('version', '') or '')[:128],
            'language': str(chunk.get('language', 'english'))[:128],
            'semantic_labels': semantic_labels_str[:65535],
        })
    
    print(f"\n📊 Total text embeddings: {len(all_metadata)}")
    print(f"⚠ Chunks without sparse match: {skipped_sparse}")
    
    dim = dense_embeddings.shape[1]
    print(f"📏 Dense embedding dimension: {dim}")
    
    # Create collection
    drop_if_exists("VictorText2")
    col = create_text_hybrid_collection("VictorText2", dim)
    
    # Insert in batches
    batch_size = 1000
    total = len(all_metadata)
    
    print("\n💾 Inserting data into Milvus...")
    for i in tqdm(range(0, total, batch_size), desc="Inserting batches"):
        end = min(i + batch_size, total)
        
        batch_data = [
            dense_embeddings[i:end].tolist(),
            all_sparse_embeddings[i:end],
            [m['document_name'] for m in all_metadata[i:end]],
            [m['document_id'] for m in all_metadata[i:end]],
            [m['chunk_id'] for m in all_metadata[i:end]],
            [m['global_chunk_id'] for m in all_metadata[i:end]],
            [m['page_idx'] for m in all_metadata[i:end]],
            [m['chunk_index'] for m in all_metadata[i:end]],
            [m['section_hierarchy'] for m in all_metadata[i:end]],
            [m['heading_context'] for m in all_metadata[i:end]],
            [m['text'] for m in all_metadata[i:end]],
            [m['char_count'] for m in all_metadata[i:end]],
            [m['word_count'] for m in all_metadata[i:end]],
            # New fields
            [m['Category'] for m in all_metadata[i:end]],
            [m['document_type'] for m in all_metadata[i:end]],
            [m['ministry'] for m in all_metadata[i:end]],
            [m['published_date'] for m in all_metadata[i:end]],
            [m['source_reference'] for m in all_metadata[i:end]],
            [m['version'] for m in all_metadata[i:end]],
            [m['language'] for m in all_metadata[i:end]],
            [m['semantic_labels'] for m in all_metadata[i:end]],
        ]
        
        col.insert(batch_data)
    
    col.flush()
    print(f"✅ Inserted {col.num_entities} text entities")
    
    build_hybrid_indexes(col)
    print(f"🎉 VictorText2 hybrid collection ready!")


def insert_victor_table_hybrid():
    """Insert table embeddings with both dense and sparse vectors - FLAT FILE VERSION"""
    print("\n" + "="*60)
    print("📊 Inserting VictorTable2 Collection (Hybrid: Dense + Sparse)")
    print("="*60)
    
    # Load from consolidated files
    dense_emb_path = EMBEDDINGS_DIR / "all_table_embeddings.npy"
    chunks_path = EMBEDDINGS_DIR / "all_table_chunks_enriched.json"
    sparse_npz_path = SPARSE_EMBEDDINGS_DIR / "table_sparse_embeddings.npz"
    
    # Validate files exist
    if not dense_emb_path.exists():
        print(f"❌ Dense embeddings not found: {dense_emb_path}")
        return
    if not chunks_path.exists():
        print(f"❌ Chunks JSON not found: {chunks_path}")
        return
    if not sparse_npz_path.exists():
        print(f"❌ Sparse embeddings not found: {sparse_npz_path}")
        return
    
    # Load dense embeddings
    print(f"📂 Loading dense embeddings from {dense_emb_path}")
    dense_embeddings = np.load(dense_emb_path)
    print(f"   ✅ Loaded {dense_embeddings.shape[0]} dense embeddings, dim={dense_embeddings.shape[1]}")
    
    # Load chunks metadata
    print(f"📂 Loading chunks from {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"   ✅ Loaded {len(chunks)} chunks")
    
    # Load sparse embeddings
    sparse_csr, sparse_metadata = load_sparse_npz(sparse_npz_path)
    sparse_id_mapping = build_global_chunk_id_mapping(sparse_metadata)
    
    # Validate counts match
    if len(chunks) != dense_embeddings.shape[0]:
        print(f"⚠ Warning: {dense_embeddings.shape[0]} embeddings vs {len(chunks)} chunks")
        min_count = min(len(chunks), dense_embeddings.shape[0])
        print(f"   Using first {min_count} entries")
        chunks = chunks[:min_count]
        dense_embeddings = dense_embeddings[:min_count]
    
    # Prepare data
    all_sparse_embeddings = []
    all_metadata = []
    skipped_sparse = 0
    
    print("🔄 Processing table chunks...")
    for idx, chunk in enumerate(tqdm(chunks, desc="Preparing data")):
        global_chunk_id = chunk.get('global_chunk_id', f'table_unknown_{idx}')
        
        sparse_idx = sparse_id_mapping.get(global_chunk_id)
        
        if sparse_idx is not None:
            sparse_dict = csr_row_to_sparse_dict(sparse_csr, sparse_idx)
        else:
            sparse_dict = {0: 0.0}
            skipped_sparse += 1
        
        all_sparse_embeddings.append(sparse_dict)
        all_metadata.append({
            'document_name': str(chunk.get('document_name', chunk.get('source', 'unknown')))[:512],
            'document_id': str(chunk.get('document_id', 'unknown'))[:256],
            'chunk_id': str(chunk.get('chunk_id', f'table_{idx}'))[:256],
            'global_chunk_id': str(global_chunk_id)[:256],
            'page_idx': int(chunk.get('page_idx', chunk.get('page', 0))),
            'table_index': int(chunk.get('table_index', idx)),
            'section_hierarchy': str(chunk.get('section_hierarchy', ''))[:50000],
            'heading_context': str(chunk.get('heading_context', ''))[:50000],
            'table_text': str(chunk.get('table_text', chunk.get('text', '')))[:65535],
            'table_markdown': str(chunk.get('table_markdown', ''))[:65535],
            'table_html': str(chunk.get('table_html', ''))[:65535],
            'caption': str(chunk.get('caption', ''))[:65535],
            'row_count': int(chunk.get('row_count', 0)),
            'col_count': int(chunk.get('col_count', 0)),
        })
    
    print(f"\n📊 Total table embeddings: {len(all_metadata)}")
    print(f"⚠ Tables without sparse match: {skipped_sparse}")
    
    dim = dense_embeddings.shape[1]
    print(f"📏 Dense embedding dimension: {dim}")
    
    drop_if_exists("VictorTable2")
    col = create_table_hybrid_collection("VictorTable2", dim)
    
    batch_size = 1000
    total = len(all_metadata)
    
    print("\n💾 Inserting data into Milvus...")
    for i in tqdm(range(0, total, batch_size), desc="Inserting batches"):
        end = min(i + batch_size, total)
        
        batch_data = [
            dense_embeddings[i:end].tolist(),
            all_sparse_embeddings[i:end],
            [m['document_name'] for m in all_metadata[i:end]],
            [m['document_id'] for m in all_metadata[i:end]],
            [m['chunk_id'] for m in all_metadata[i:end]],
            [m['global_chunk_id'] for m in all_metadata[i:end]],
            [m['page_idx'] for m in all_metadata[i:end]],
            [m['table_index'] for m in all_metadata[i:end]],
            [m['section_hierarchy'] for m in all_metadata[i:end]],
            [m['heading_context'] for m in all_metadata[i:end]],
            [m['table_text'] for m in all_metadata[i:end]],
            [m['table_markdown'] for m in all_metadata[i:end]],
            [m['table_html'] for m in all_metadata[i:end]],
            [m['caption'] for m in all_metadata[i:end]],
            [m['row_count'] for m in all_metadata[i:end]],
            [m['col_count'] for m in all_metadata[i:end]],
        ]
        
        col.insert(batch_data)
    
    col.flush()
    print(f"✅ Inserted {col.num_entities} table entities")
    
    build_hybrid_indexes(col)
    print(f"🎉 VictorTable2 hybrid collection ready!")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🚀 MILVUS HYBRID COLLECTION CREATOR")
    print("   Dense (HNSW) + Sparse (Inverted Index) for Hybrid Search")
    print("="*70)
    print(f"\n📁 Paths:")
    print(f"   Dense embeddings: {EMBEDDINGS_DIR}")
    print(f"   Sparse embeddings: {SPARSE_EMBEDDINGS_DIR}")
    print(f"   Chunks: {EMBEDDINGS_DIR}")
    
    connect_milvus()
    
    insert_victor_text_hybrid()
    insert_victor_table_hybrid()
    
    print("\n" + "="*70)
    print("✅ COMPLETE! Created collections:")
    print("   • VictorText2  - Text chunks with hybrid search")
    print("   • VictorTable2 - Table chunks with hybrid search")
    print("="*70)


if __name__ == "__main__":
    main()