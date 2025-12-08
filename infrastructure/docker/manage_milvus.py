from pymilvus import connections, utility, Collection, DataType

def connect_milvus():
    """Connect to Milvus"""
    connections.connect("default", host="192.168.65.80", port="19530")
    print("✅ Connected to Milvus")

def list_collections():
    """List all collections"""
    connect_milvus()
    collections = utility.list_collections()
    
    if not collections:
        print("❌ No collections found")
        return
    
    print(f"\n📚 Found {len(collections)} collection(s):")
    print("="*80)
    
    for name in collections:
        collection = Collection(name)
        print(f"\n📊 Collection: {name}")
        print(f"   Entities: {collection.num_entities}")
        print(f"   Description: {collection.description}")

def show_schema(collection_name):
    """Show collection schema"""
    connect_milvus()
    
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
        print(f"      Type: {dtype_map.get(field.dtype, field.dtype)}")
        
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
        print(f"  Error reading indexes: {e}")

def delete_collection(collection_name):
    """Delete a collection"""
    connect_milvus()
    
    if not utility.has_collection(collection_name):
        print(f"❌ Collection '{collection_name}' not found")
        return
    
    confirm = input(f"⚠️  Are you sure you want to delete '{collection_name}'? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Deletion cancelled")
        return
    
    utility.drop_collection(collection_name)
    print(f"✅ Collection '{collection_name}' deleted successfully")

def delete_all_collections():
    """Delete ALL collections"""
    connect_milvus()
    collections = utility.list_collections()
    
    if not collections:
        print("❌ No collections to delete")
        return
    
    print(f"⚠️  Found {len(collections)} collection(s): {collections}")
    confirm = input("⚠️  Delete ALL collections? This cannot be undone! (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("❌ Deletion cancelled")
        return
    
    for name in collections:
        utility.drop_collection(name)
        print(f"✅ Deleted: {name}")
    
    print(f"✅ All collections deleted")

def collection_stats(collection_name):
    """Show detailed collection statistics"""
    connect_milvus()
    
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
    except:
        print(f"Status: ⚠️  Not loaded")
    
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
        print(f"  No index info available: {e}")

# Interactive Menu
if __name__ == "__main__":
    while True:
        print("\n" + "="*80)
        print("🗄️  MILVUS MANAGEMENT MENU")
        print("="*80)
        print("1. List all collections")
        print("2. Show collection schema")
        print("3. Show collection statistics")
        print("4. Delete specific collection")
        print("5. Delete ALL collections")
        print("6. Exit")
        print("="*80)
        
        choice = input("\nEnter choice (1-6): ").strip()
        
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
            delete_collection(collection_name)
        
        elif choice == "5":
            delete_all_collections()
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")