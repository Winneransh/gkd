"""
Chroma Client Factory for Chroma Cloud service only
"""
import logging
from typing import Optional
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
import chromadb
from chromadb.api import ClientAPI
from chroma_config import ChromaConfig

logger = logging.getLogger(__name__)

class ChromaClientFactory:
    """Factory for creating Chroma Cloud clients and vectorstores."""
    
    def __init__(self, config: ChromaConfig):
        self.config = config
        self._client: Optional[ClientAPI] = None
    
    def get_client(self) -> ClientAPI:
        """Get Chroma Cloud client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self) -> ClientAPI:
        """Create Chroma Cloud client."""
        # Validate configuration first
        self.config.validate_configuration()
        
        logger.info(f"Creating Chroma Cloud client (tenant: {self.config.chroma_tenant}, database: {self.config.chroma_database})")
        return chromadb.CloudClient(
            api_key=self.config.chroma_api_key,
            tenant=self.config.chroma_tenant,
            database=self.config.chroma_database
        )
    
    def get_vectorstore(self, 
                       collection_name: str,
                       embedding_function: Embeddings) -> Chroma:
        """
        Get Chroma vectorstore instance.
        
        Args:
            collection_name: Name of the collection
            embedding_function: Embeddings to use
            
        Returns:
            Chroma vectorstore instance
        """
        client = self.get_client()
        
        # Always use Chroma Cloud - no local fallback
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
    
    def create_collection_if_not_exists(self, 
                                      collection_name: str,
                                      embedding_function: Embeddings,
                                      metadata: Optional[dict] = None) -> None:
        """
        Create collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            embedding_function: Embeddings to use
            metadata: Optional metadata for the collection
        """
        client = self.get_client()
        
        try:
            # Try to get existing collection
            collection = client.get_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
        except Exception:
            # Collection doesn't exist, create it
            logger.info(f"Creating new collection: '{collection_name}'")
            collection = client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            logger.info(f"Successfully created collection: '{collection_name}'")
    
    def list_collections(self) -> list:
        """List all collections."""
        client = self.get_client()
        collections = client.list_collections()
        return [collection.name for collection in collections]
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            client = self.get_client()
            client.delete_collection(name=collection_name)
            logger.info(f"Successfully deleted collection: '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {str(e)}")
            return False
    
    def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a collection."""
        try:
            client = self.get_client()
            collection = client.get_collection(name=collection_name)
            
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {str(e)}")
            return {}
