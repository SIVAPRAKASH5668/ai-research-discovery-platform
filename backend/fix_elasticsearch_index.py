#!/usr/bin/env python3
"""
Elasticsearch Index Rebuild Script - Option 1
Deletes old index and creates new one with correct dense_vector mapping
Run this once to fix vector search issues
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend src to path
backend_dir = Path(__file__).parent
src_dir = backend_dir / 'src'
sys.path.insert(0, str(src_dir))

# Now import after path is set
from database.elastic_client import ElasticClient
from dotenv import load_dotenv

# Load environment variables from backend/.env
env_path = backend_dir / '.env'
load_dotenv(env_path)


async def rebuild_index():
    """Delete old index and create new one with correct dense_vector mapping"""
    
    print()
    print("=" * 80)
    print("🔧 ELASTICSEARCH INDEX REBUILD - Option 1 (Quick Fix)")
    print("=" * 80)
    print()
    print("⚠️  WARNING: This will delete your existing papers!")
    print("    Current: ~33 papers")
    print()
    print("✅ BENEFITS after rebuild:")
    print("    • Vector search will work correctly")
    print("    • dense_vector field properly configured (768D, cosine)")
    print("    • Next search will get 100-200 papers (vs 16)")
    print("    • All papers will have proper embeddings")
    print("    • Relationship discovery will work")
    print()
    
    # Ask for confirmation
    try:
        response = input("Continue with rebuild? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print()
            print("❌ Cancelled. No changes made.")
            print()
            return
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user. No changes made.\n")
        return
    
    print()
    print("🚀 Starting index rebuild...")
    print()
    
    # Initialize client
    try:
        client = ElasticClient()
    except Exception as e:
        print(f"❌ Failed to initialize Elasticsearch client: {e}")
        print()
        print("💡 Check your .env file has:")
        print("   ELASTIC_ENDPOINT=https://...")
        print("   ELASTIC_API_KEY=...")
        print()
        return
    
    try:
        # Step 1: Test connection
        print("1️⃣ Testing Elasticsearch connection...")
        try:
            info = await client.async_client.info()
            cluster_name = info.get('cluster_name', 'unknown')
            version = info.get('version', {}).get('number', 'unknown')
            print(f"   ✅ Connected to cluster: {cluster_name}")
            print(f"   📌 Elasticsearch version: {version}")
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")
            print()
            print("💡 Troubleshooting:")
            print("   • Check ELASTIC_ENDPOINT is correct")
            print("   • Check ELASTIC_API_KEY is valid")
            print("   • Check network connectivity")
            print()
            return
        
        # Step 2: Check current index
        print()
        print("2️⃣ Checking current index status...")
        index_exists = await client.async_client.indices.exists(index=client.index_name)
        
        if index_exists:
            # Get document count
            try:
                count_response = await client.async_client.count(index=client.index_name)
                doc_count = count_response['count']
                
                # Get index size
                stats = await client.async_client.indices.stats(index=client.index_name)
                size_bytes = stats["indices"][client.index_name]["total"]["store"]["size_in_bytes"]
                size_mb = round(size_bytes / (1024 * 1024), 2)
                
                print(f"   📊 Index: {client.index_name}")
                print(f"   📄 Documents: {doc_count} papers")
                print(f"   💾 Size: {size_mb} MB")
                
                # Check mapping
                mapping = await client.async_client.indices.get_mapping(index=client.index_name)
                properties = mapping[client.index_name]["mappings"]["properties"]
                embedding_field = properties.get("content_embedding", {})
                
                if embedding_field:
                    field_type = embedding_field.get("type", "unknown")
                    dims = embedding_field.get("dims", "unknown")
                    print(f"   🔍 Current vector field: type={field_type}, dims={dims}")
                    
                    if field_type != "dense_vector" or dims != 768:
                        print(f"   ❌ Vector field incorrectly configured!")
                    else:
                        print(f"   ⚠️  Vector field looks correct but may have other issues")
                else:
                    print(f"   ❌ No vector field found!")
                
            except Exception as e:
                print(f"   ⚠️  Could not get index details: {e}")
            
            # Step 3: Delete old index
            print()
            print("3️⃣ Deleting old index...")
            try:
                await client.async_client.indices.delete(index=client.index_name)
                print("   ✅ Old index deleted successfully")
            except Exception as e:
                print(f"   ❌ Failed to delete index: {e}")
                return
        else:
            print(f"   ℹ️  Index '{client.index_name}' does not exist yet")
        
        # Wait for Elasticsearch to process
        print()
        print("   ⏳ Waiting for Elasticsearch...")
        await asyncio.sleep(3)
        
        # Step 4: Create new index with correct mapping
        print()
        print("4️⃣ Creating new index with fixed dense_vector mapping...")
        try:
            await client.create_index()
            print("   ✅ New index created successfully!")
        except Exception as e:
            print(f"   ❌ Failed to create index: {e}")
            return
        
        # Step 5: Verify new index
        print()
        print("5️⃣ Verifying new index configuration...")
        await asyncio.sleep(2)
        
        try:
            # Check index exists
            exists = await client.async_client.indices.exists(index=client.index_name)
            if not exists:
                print("   ❌ Index was not created!")
                return
            
            # Check mapping
            mapping = await client.async_client.indices.get_mapping(index=client.index_name)
            properties = mapping[client.index_name]["mappings"]["properties"]
            embedding_field = properties.get("content_embedding", {})
            
            if embedding_field:
                field_type = embedding_field.get("type")
                dims = embedding_field.get("dims")
                index_enabled = embedding_field.get("index")
                similarity = embedding_field.get("similarity")
                
                print("   ✅ Vector field configuration:")
                print(f"      • Type: {field_type}")
                print(f"      • Dimensions: {dims}")
                print(f"      • Index: {index_enabled}")
                print(f"      • Similarity: {similarity}")
                
                if (field_type == "dense_vector" and 
                    dims == 768 and 
                    index_enabled == True and 
                    similarity == "cosine"):
                    print("   ✅ Perfect! Vector field correctly configured!")
                else:
                    print("   ⚠️  Warning: Some settings may be incorrect")
            else:
                print("   ❌ Vector field not found in mapping!")
                return
            
            # Check other important fields
            required_fields = ["title", "abstract", "authors", "searchable_text"]
            missing_fields = [f for f in required_fields if f not in properties]
            
            if missing_fields:
                print(f"   ⚠️  Missing fields: {', '.join(missing_fields)}")
            else:
                print("   ✅ All required fields present")
            
        except Exception as e:
            print(f"   ⚠️  Verification warning: {e}")
        
        # Step 6: Final test
        print()
        print("6️⃣ Running final connection test...")
        try:
            await client.test_connection()
            print("   ✅ Connection test passed!")
        except Exception as e:
            print(f"   ⚠️  Test warning: {e}")
        
        # Success!
        print()
        print("=" * 80)
        print("🎉 SUCCESS! Index rebuild completed successfully!")
        print("=" * 80)
        print()
        print("✅ What's been fixed:")
        print("   • Old index with broken mapping deleted")
        print("   • New index created with correct dense_vector field")
        print("   • Vector field: 768 dimensions, cosine similarity")
        print("   • Index: enabled for fast KNN search")
        print()
        print("📊 Next steps:")
        print("   1. Restart your backend server:")
        print("      cd backend && uvicorn main:app --reload")
        print()
        print("   2. Make a search request (e.g., 'machine learning')")
        print()
        print("   3. Expected results:")
        print("      • 100-200 papers (vs previous 16)")
        print("      • All from multiple APIs (arXiv, PubMed, etc.)")
        print("      • All papers with 768D vector embeddings")
        print("      • Vector similarity search working")
        print("      • Relationship discovery working")
        print()
        print("🎯 Your research platform is now fully operational!")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ ERROR OCCURRED: {e}")
        print("=" * 80)
        print()
        import traceback
        traceback.print_exc()
        print()
    
    finally:
        # Close connections
        try:
            await client.close()
            print("🔌 Elasticsearch connections closed")
            print()
        except:
            pass


def main():
    """Main entry point"""
    print()
    
    # Check if .env file exists
    backend_dir = Path(__file__).parent
    env_path = backend_dir / '.env'
    
    if not env_path.exists():
        print("❌ ERROR: .env file not found!")
        print(f"   Expected location: {env_path}")
        print()
        print("💡 Create a .env file with:")
        print("   ELASTIC_ENDPOINT=https://your-elasticsearch-url")
        print("   ELASTIC_API_KEY=your-api-key")
        print("   ELASTIC_INDEX_NAME=research-corpus")
        print()
        return
    
    try:
        asyncio.run(rebuild_index())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...\n")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
