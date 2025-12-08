import os
from pymilvus import connections, utility, Collection
from dotenv import load_dotenv

load_dotenv()

def test_milvus_connection():
    """Test Milvus connection and check collection"""
    
    print("="*70)
    print("🔍 Milvus Connection Test")
    print("="*70)
    
    # Get configuration from .env
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("COLLECTION_NAME", "Vtext")
    
    print(f"\n📋 Configuration:")
    print(f"   Milvus Host: {milvus_host}")
    print(f"   Milvus Port: {milvus_port}")
    print(f"   Collection Name: {collection_name}")
    
    try:
        print(f"\n🔄 Connecting to Milvus...")
        
        # Connect to Milvus
        connections.connect(
            "default",
            host=milvus_host,
            port=milvus_port,
            timeout=10
        )
        
        print(f"✅ Milvus Connection SUCCESSFUL!")
        
        # Get server info
        print(f"\n📊 Milvus Server Info:")
        server_version = utility.get_server_version()
        print(f"   Version: {server_version}")
        
        # List all collections
        collections_list = utility.list_collections()
        print(f"\n📚 Available Collections ({len(collections_list)}):")
        for col_name in collections_list:
            print(f"   - {col_name}")
        
        # Check if target collection exists
        print(f"\n🔍 Checking Collection: {collection_name}")
        
        if utility.has_collection(collection_name):
            print(f"✅ Collection '{collection_name}' EXISTS")
            
            # Load collection
            collection = Collection(collection_name)
            collection.load()
            
            # Get collection info
            print(f"\n📋 Collection Details:")
            print(f"   Total Entities: {collection.num_entities}")
            
            # Get schema
            schema = collection.schema
            print(f"\n🏗️  Schema Fields:")
            for field in schema.fields:
                print(f"   - {field.name}")
                print(f"      Type: {field.dtype}")
                if hasattr(field, 'params') and field.params:
                    print(f"      Params: {field.params}")
            
            # Get indexes
            print(f"\n📑 Indexes:")
            indexes = collection.indexes
            if indexes:
                for idx in indexes:
                    print(f"   - {idx.field_name}")
                    print(f"      Index Type: {idx.index_type}")
                    print(f"      Metric Type: {idx.metric_type}")
            else:
                print(f"   No indexes found")
            
            # Disconnect
            connections.disconnect("default")
            print(f"\n✅ Disconnected successfully")
            
        else:
            print(f"❌ Collection '{collection_name}' DOES NOT EXIST")
            print(f"\n⚠️  Available collections are:")
            for col in collections_list:
                print(f"   - {col}")
            
            connections.disconnect("default")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        print(f"\n📋 Full Traceback:")
        print(traceback.format_exc())
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_milvus_connection()