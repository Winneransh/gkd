#!/usr/bin/env python3
"""
migrate_to_trychroma.py

Migration script to help users migrate from local ChromaDB to Trychroma cloud service.
This script will:
1. Check current local ChromaDB collections
2. Export data from local ChromaDB
3. Connect to Trychroma cloud service
4. Import data to Trychroma
5. Verify the migration
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBMigrator:
    """Migrates data from local ChromaDB to Trychroma cloud service."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Local ChromaDB settings
        self.local_persist_dir = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        
        # Trychroma cloud settings
        self.cloud_host = os.getenv('CHROMA_SERVER_HOST')
        self.cloud_port = int(os.getenv('CHROMA_SERVER_PORT', '443'))
        self.cloud_api_token = os.getenv('CHROMA_API_TOKEN')
        
        # Initialize clients
        self.local_client = None
        self.cloud_client = None
        
    def check_configuration(self) -> bool:
        """Check if Trychroma configuration is valid."""
        if not self.cloud_host:
            logger.error("CHROMA_SERVER_HOST not set in environment variables")
            return False
            
        if not self.cloud_api_token:
            logger.error("CHROMA_API_TOKEN not set in environment variables")
            return False
            
        logger.info(f"✓ Trychroma configuration found:")
        logger.info(f"  Host: {self.cloud_host}")
        logger.info(f"  Port: {self.cloud_port}")
        logger.info(f"  API Token: {'*' * (len(self.cloud_api_token) - 4)}{self.cloud_api_token[-4:]}")
        
        return True
    
    def connect_local_client(self) -> bool:
        """Connect to local ChromaDB."""
        try:
            if not Path(self.local_persist_dir).exists():
                logger.error(f"Local ChromaDB directory not found: {self.local_persist_dir}")
                return False
                
            self.local_client = chromadb.PersistentClient(path=self.local_persist_dir)
            logger.info(f"✓ Connected to local ChromaDB: {self.local_persist_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to local ChromaDB: {str(e)}")
            return False
    
    def connect_cloud_client(self) -> bool:
        """Connect to Trychroma cloud service."""
        try:
            self.cloud_client = chromadb.HttpClient(
                host=self.cloud_host,
                port=self.cloud_port,
                ssl=True,
                headers={"Authorization": f"Bearer {self.cloud_api_token}"}
            )
            
            # Test connection by listing collections
            collections = self.cloud_client.list_collections()
            logger.info(f"✓ Connected to Trychroma cloud service")
            logger.info(f"  Found {len(collections)} existing collections in cloud")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Trychroma: {str(e)}")
            return False
    
    def list_local_collections(self) -> List[str]:
        """List all collections in local ChromaDB."""
        if not self.local_client:
            return []
            
        try:
            collections = self.local_client.list_collections()
            collection_names = [collection.name for collection in collections]
            logger.info(f"Found {len(collection_names)} local collections: {collection_names}")
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to list local collections: {str(e)}")
            return []
    
    def export_collection_data(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Export all data from a local collection."""
        try:
            collection = self.local_client.get_collection(name=collection_name)
            
            # Get all data from the collection
            result = collection.get(include=["documents", "metadatas", "embeddings"])
            
            collection_data = {
                "name": collection_name,
                "metadata": collection.metadata,
                "count": collection.count(),
                "ids": result["ids"],
                "documents": result["documents"],
                "metadatas": result["metadatas"],
                "embeddings": result["embeddings"]
            }
            
            logger.info(f"✓ Exported {collection_data['count']} documents from '{collection_name}'")
            return collection_data
            
        except Exception as e:
            logger.error(f"Failed to export collection '{collection_name}': {str(e)}")
            return None
    
    def import_collection_data(self, collection_data: Dict[str, Any]) -> bool:
        """Import data to Trychroma cloud collection."""
        try:
            collection_name = collection_data["name"]
            
            # Create or get collection in cloud
            try:
                collection = self.cloud_client.create_collection(
                    name=collection_name,
                    metadata=collection_data["metadata"] or {}
                )
                logger.info(f"✓ Created new collection '{collection_name}' in Trychroma")
            except Exception:
                # Collection might already exist
                collection = self.cloud_client.get_collection(name=collection_name)
                logger.info(f"✓ Using existing collection '{collection_name}' in Trychroma")
            
            # Add documents to cloud collection
            if collection_data["ids"] and len(collection_data["ids"]) > 0:
                collection.add(
                    ids=collection_data["ids"],
                    documents=collection_data["documents"],
                    metadatas=collection_data["metadatas"],
                    embeddings=collection_data["embeddings"]
                )
                
                logger.info(f"✓ Imported {len(collection_data['ids'])} documents to Trychroma collection '{collection_name}'")
                return True
            else:
                logger.warning(f"No documents to import for collection '{collection_name}'")
                return True
                
        except Exception as e:
            logger.error(f"Failed to import collection '{collection_data['name']}': {str(e)}")
            return False
    
    def verify_migration(self, collection_name: str, local_count: int) -> bool:
        """Verify that migration was successful."""
        try:
            cloud_collection = self.cloud_client.get_collection(name=collection_name)
            cloud_count = cloud_collection.count()
            
            if cloud_count == local_count:
                logger.info(f"✓ Migration verified for '{collection_name}': {cloud_count} documents")
                return True
            else:
                logger.error(f"Migration verification failed for '{collection_name}': local={local_count}, cloud={cloud_count}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify migration for '{collection_name}': {str(e)}")
            return False
    
    def migrate_all_collections(self) -> bool:
        """Migrate all collections from local to cloud."""
        logger.info("🚀 Starting ChromaDB to Trychroma migration...")
        
        # Check configuration
        if not self.check_configuration():
            return False
        
        # Connect to both local and cloud
        if not self.connect_local_client():
            return False
            
        if not self.connect_cloud_client():
            return False
        
        # Get local collections
        local_collections = self.list_local_collections()
        if not local_collections:
            logger.info("No local collections found to migrate")
            return True
        
        success_count = 0
        total_collections = len(local_collections)
        
        # Migrate each collection
        for collection_name in local_collections:
            logger.info(f"\n📦 Migrating collection: {collection_name}")
            
            # Export from local
            collection_data = self.export_collection_data(collection_name)
            if not collection_data:
                continue
            
            # Import to cloud
            if self.import_collection_data(collection_data):
                # Verify migration
                if self.verify_migration(collection_name, collection_data["count"]):
                    success_count += 1
        
        # Summary
        logger.info(f"\n🎉 Migration completed: {success_count}/{total_collections} collections migrated successfully")
        
        if success_count == total_collections:
            logger.info("✓ All collections migrated successfully!")
            logger.info("You can now update your environment to use CHROMA_USE_LOCAL=false")
            return True
        else:
            logger.error(f"⚠️  Some collections failed to migrate. Please check the logs above.")
            return False

def main():
    """Main function to run the migration."""
    print("=" * 70)
    print("🔄 ChromaDB to Trychroma Migration Tool")
    print("=" * 70)
    
    migrator = ChromaDBMigrator()
    
    # Check if .env file exists
    if not Path('.env').exists():
        print("\n⚠️  .env file not found!")
        print("Please create a .env file with your Trychroma configuration.")
        print("You can copy .env.example and update the values:")
        print("  cp .env.example .env")
        print("  # Edit .env with your Trychroma credentials")
        return False
    
    # Run migration
    success = migrator.migrate_all_collections()
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Update your .env file: CHROMA_USE_LOCAL=false")
        print("2. Restart your application")
        print("3. Test that everything works with Trychroma")
        print("4. (Optional) Remove local ChromaDB directory after testing")
    else:
        print("\n❌ Migration failed. Please check the logs above.")
        print("Make sure your Trychroma credentials are correct in .env")
    
    return success

if __name__ == "__main__":
    main()
