"""Quick test to check if Milvus is working with Vtext schema"""
from pymilvus import connections, Collection, utility
import sys

try:
    print("🔄 Connecting to Milvus at localhost:19530...")
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("✅ Connected to Milvus successfully!")
    
    # List all collections
    print(f"\n📋 Available collections:")
    collections = utility.list_collections()
    if not collections:
        print("   ❌ No collections found!")
        print("\n💡 You need to create the collection first using:")
        print("   python backend/scripts/create_milvus_collection.py")
        sys.exit(1)
    
    for coll_name in collections:
        print(f"   ✅ {coll_name}")
    
    # Use the Vtext collection
    collection_name = "Vtext"
    if collection_name not in collections:
        print(f"\n❌ Collection '{collection_name}' not found!")
        print(f"💡 Available: {', '.join(collections)}")
        sys.exit(1)
    
    print(f"\n🔍 Testing Vtext collection: {collection_name}")
    
    collection = Collection(collection_name)
    collection.load()
    
    print(f"✅ Collection loaded successfully!")
    print(f"📊 Number of entities: {collection.num_entities}")
    
    # Get schema to check Vtext fields
    print(f"\n📋 Collection Schema (Vtext fields):")
    schema = collection.schema
    for field in schema.fields:
        print(f"   - {field.name} ({field.dtype})")
    
    # Try a simple query with ALL Vtext fields
    print(f"\n🔎 Testing search query with Vtext schema...")
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    # Create a dummy query vector (768 dimensions for OpenAI embeddings)
    import numpy as np
    query_vector = np.random.rand(768).tolist()
    
    # Request ALL Vtext fields
    vtext_fields = [
        "text", "source_file", "page_idx", 
        "global_chunk_id", "document_id", "chunk_index",
        "section_hierarchy", "char_count", "word_count"
    ]
    
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=vtext_fields
    )
    
    print(f"✅ Search completed successfully!")
    print(f"📋 Found {len(results[0])} results\n")
    
    for i, hit in enumerate(results[0], 1):
        print(f"{'='*80}")
        print(f"Result {i} - Score: {hit.score:.4f}")
        print(f"{'='*80}")
        
        # Display standard fields
        print(f"📄 Source File: {hit.entity.get('source_file', 'N/A')}")
        print(f"📖 Page: {hit.entity.get('page_idx', 'N/A')}")
        print(f"📝 Text: {hit.entity.get('text', 'N/A')[:150]}...")
        
        # Display Vtext-specific fields
        print(f"\n🔍 Vtext Metadata:")
        print(f"   Global Chunk ID: {hit.entity.get('global_chunk_id', 'N/A')}")
        print(f"   Document ID: {hit.entity.get('document_id', 'N/A')}")
        print(f"   Chunk Index: {hit.entity.get('chunk_index', 'N/A')}")
        print(f"   Section Hierarchy: {hit.entity.get('section_hierarchy', 'N/A')}")
        print(f"   Char Count: {hit.entity.get('char_count', 'N/A')}")
        print(f"   Word Count: {hit.entity.get('word_count', 'N/A')}")
        print()
    
    print("="*80)
    print("✅ MILVUS WITH VTEXT SCHEMA IS WORKING CORRECTLY!")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print(f"❌ MILVUS CONNECTION FAILED!")
    sys.exit(1)
