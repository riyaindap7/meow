from pymilvus import connections, utility, Collection, DataType

def connect_milvus():
    """Connect to Milvus"""
    try:
        connections.connect("default", host="localhost", port="19530", timeout=10)
        print("✅ Connected to Milvus")
        return True
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

def list_collections():
    """List all collections"""
    if not connect_milvus():
        return
    
    collections = utility.list_collections()
    
    if not collections:
        print("❌ No collections found")
        return
    
    print(f"\n📚 Found {len(collections)} collection(s):")
    print("="*80)
    
    for name in collections:
        try:
            collection = Collection(name)
            print(f"\n📊 Collection: {name}")
            print(f"   Entities: {collection.num_entities}")
            print(f"   Description: {collection.description}")
        except Exception as e:
            print(f"   ⚠ Error loading {name}: {e}")

def show_schema(collection_name):
    """Show collection schema"""
    if not connect_milvus():
        return
    
    if not utility.has_collection(collection_name):
        print(f"❌ Collection '{collection_name}' not found")
        return
    
    collection = Collection(collection_name)
    schema = collection.schema
    
    print(f"\n🔍 Schema for collection: {collection_name}")
    print("="*80)
    print(f"Description: {schema.description}")
    print(f"\nFields:")
    
    # Map DataType enums to readable names
    dtype_map = {
        DataType.BOOL: "BOOL",
        DataType.INT8: "INT8",
        DataType.INT16: "INT16",
        DataType.INT32: "INT32",
        DataType.INT64: "INT64",
        DataType.FLOAT: "FLOAT",
        DataType.DOUBLE: "DOUBLE",
        DataType.STRING: "STRING",
        DataType.VARCHAR: "VARCHAR",
        DataType.BINARY_VECTOR: "BINARY_VECTOR",
        DataType.FLOAT_VECTOR: "FLOAT_VECTOR",
    }
    
    for field in schema.fields:
        print(f"\n  📌 {field.name}")
        print(f"      Type: {dtype_map.get(field.dtype, str(field.dtype))}")
        
        if field.dtype == DataType.FLOAT_VECTOR:
            print(f"      Dimension: {field.params.get('dim', 'N/A')}")
        
        if field.is_primary:
            print(f"      ⭐ Primary Key: Yes")
            print(f"      Auto ID: {field.auto_id}")
        
        if field.dtype == DataType.VARCHAR:
            print(f"      Max Length: {field.params.get('max_length', 'N/A')}")
    
    # Show indexes
    print(f"\n📑 Indexes:")
    try:
        indexes = collection.indexes
        if indexes:
            for idx in indexes:
                print(f"\n  Index on: {idx.field_name}")
                print(f"    Type: {idx.params.get('index_type', 'N/A')}")
                print(f"    Metric: {idx.params.get('metric_type', 'N/A')}")
                print(f"    Params: {idx.params.get('params', {})}")
        else:
            print("  No indexes found")
    except Exception as e:
        print(f"  ⚠ Error reading indexes: {e}")

def delete_collection(collection_name):
    """Delete a collection"""
    if not connect_milvus():
        return
    
    if not utility.has_collection(collection_name):
        print(f"❌ Collection '{collection_name}' not found")
        return
    
    confirm = input(f"⚠  Are you sure you want to delete '{collection_name}'? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Deletion cancelled")
        return
    
    utility.drop_collection(collection_name)
    print(f"✅ Collection '{collection_name}' deleted successfully")

def delete_all_collections():
    """Delete ALL collections"""
    if not connect_milvus():
        return
    
    collections = utility.list_collections()
    
    if not collections:
        print("❌ No collections to delete")
        return
    
    print(f"⚠  Found {len(collections)} collection(s): {collections}")
    confirm = input("⚠  Delete ALL collections? This cannot be undone! (TYPE 'DELETE ALL' to confirm): ")
    
    if confirm != 'DELETE ALL':
        print("❌ Deletion cancelled")
        return
    
    for name in collections:
        utility.drop_collection(name)
        print(f"✅ Deleted: {name}")
    
    print(f"✅ All {len(collections)} collections deleted")

def collection_stats(collection_name):
    """Show detailed collection statistics"""
    if not connect_milvus():
        return
    
    if not utility.has_collection(collection_name):
        print(f"❌ Collection '{collection_name}' not found")
        return
    
    collection = Collection(collection_name)
    
    print(f"\n📊 Statistics for: {collection_name}")
    print("="*80)
    print(f"Total entities: {collection.num_entities}")
    
    # Get collection info
    try:
        collection.load()
        print(f"Status: ✅ Loaded in memory")
    except Exception as e:
        print(f"Status: ⚠  Not loaded ({e})")
    
    # Get index info
    print(f"\n📑 Index Information:")
    try:
        indexes = collection.indexes
        if indexes:
            for idx in indexes:
                print(f"  Field: {idx.field_name}")
                print(f"  Index Type: {idx.params.get('index_type')}")
                print(f"  Metric Type: {idx.params.get('metric_type')}")
                print(f"  Parameters: {idx.params.get('params', {})}")
        else:
            print("  No indexes built")
    except Exception as e:
        print(f"  ⚠ No index info available: {e}")

def sample_data(collection_name, limit=3):
    """Show sample data from collection"""
    if not connect_milvus():
        return
    
    if not utility.has_collection(collection_name):
        print(f"❌ Collection '{collection_name}' not found")
        return
    
    collection = Collection(collection_name)
    
    try:
        collection.load()
        
        # Get schema to dynamically determine available fields
        schema = collection.schema
        output_fields = [
            field.name for field in schema.fields 
            if field.name != "id" and field.dtype != DataType.FLOAT_VECTOR
        ]
        
        # Query with actual fields
        results = collection.query(
            expr="",  # Empty expression gets all records
            limit=limit,
            output_fields=output_fields
        )
        
        print(f"\n📄 Sample data from {collection_name} (showing {len(results)} items):")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Sample {i} ---")
            for key, value in result.items():
                # Truncate long strings for readability
                if isinstance(value, str) and len(value) > 150:
                    print(f"{key}: {value[:150]}...")
                else:
                    print(f"{key}: {value}")
    
    except Exception as e:
        print(f"❌ Error querying data: {e}")

# Interactive Menu
if __name__ == "__main__":
    while True:
        print("\n" + "="*80)
        print("🗄  MILVUS MANAGEMENT MENU")
        print("="*80)
        print("1. List all collections")
        print("2. Show collection schema")
        print("3. Show collection statistics")
        print("4. Show sample data")
        print("5. Delete specific collection")
        print("6. Delete ALL collections")
        print("7. Exit")
        print("="*80)
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            list_collections()
        
        elif choice == "2":
            collection_name = input("Enter collection name: ").strip()
            show_schema(collection_name)
        
        elif choice == "3":
            collection_name = input("Enter collection name: ").strip()
            collection_stats(collection_name)
        
        elif choice == "4":
            collection_name = input("Enter collection name: ").strip()
            sample_data(collection_name)
        
        elif choice == "5":
            collection_name = input("Enter collection name: ").strip()
            delete_collection(collection_name)
        
        elif choice == "6":
            delete_all_collections()
        
        elif choice == "7":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")