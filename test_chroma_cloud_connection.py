#!/usr/bin/env python3
"""
test_chroma_cloud_connection.py

Simple script to test your Chroma Cloud connection.
Run this to verify your credentials before running the full migration.
"""

import os
from dotenv import load_dotenv
import chromadb

def test_connection():
    """Test connection to Chroma Cloud."""
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    api_key = os.getenv('CHROMA_API_KEY')
    tenant = os.getenv('CHROMA_TENANT')
    database = os.getenv('CHROMA_DATABASE')
    
    if not api_key:
        print("❌ CHROMA_API_KEY not found in .env file")
        return False
    
    if not tenant:
        print("❌ CHROMA_TENANT not found in .env file")
        return False
        
    if not database:
        print("❌ CHROMA_DATABASE not found in .env file")
        return False
    
    print(f"🔑 API Key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    print(f"🏢 Tenant: {tenant}")
    print(f"🗄️  Database: {database}")
    print()
    
    try:
        print("🌐 Connecting to Chroma Cloud...")
        
        # Connect using CloudClient
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        
        # Test basic operations
        print(f"   📡 Connected to Chroma Cloud")
        
        # List collections
        collections = client.list_collections()
        print(f"   📚 Found {len(collections)} collections:")
        for collection in collections:
            print(f"      - {collection.name}")
        
        # Test creating a test collection (will be deleted)
        test_collection_name = "test_connection_collection"
        try:
            test_collection = client.create_collection(name=test_collection_name)
            print(f"   ✅ Successfully created test collection")
            
            # Clean up test collection
            client.delete_collection(name=test_collection_name)
            print(f"   🗑️  Cleaned up test collection")
            
        except Exception as create_error:
            # Try to get existing collection instead
            try:
                client.get_collection(name=test_collection_name)
                print(f"   ✅ Can access collections (test collection already exists)")
            except:
                print(f"   ⚠️  Cannot create collections: {create_error}")
        
        print(f"✅ SUCCESS: Connected to Chroma Cloud")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Chroma Cloud: {str(e)}")
        return False
def main():
    """Main function."""
    print("🧪 Chroma Cloud Connection Test")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found!")
        print("Please create a .env file with your Trychroma credentials:")
        print("  cp .env.example .env")
        print("  # Edit .env with your actual credentials")
        return
    
    success = test_connection()
    
    if success:
        print("\n🎉 Connection test passed!")
        print("You can now run your application with Chroma Cloud!")
    else:
        print("\n💡 Troubleshooting:")
        print("1. Verify your CHROMA_API_KEY is correct")
        print("2. Check that your CHROMA_TENANT and CHROMA_DATABASE are correct")
        print("3. Ensure your Chroma Cloud account is active")
        print("4. Contact Chroma support if the issue persists")

if __name__ == "__main__":
    main()
