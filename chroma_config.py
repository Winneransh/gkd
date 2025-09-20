"""
Chroma Cloud Configuration - Uses chromadb.CloudClient
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ChromaConfig:
    """Configuration for Chroma Cloud service."""
    
    # Chroma Cloud configuration
    chroma_api_key: Optional[str] = None
    chroma_tenant: Optional[str] = None
    chroma_database: Optional[str] = None
    
    @classmethod
    def from_environment(cls) -> 'ChromaConfig':
        """Create configuration from environment variables."""
        return cls(
            chroma_api_key=os.getenv('CHROMA_API_KEY'),
            chroma_tenant=os.getenv('CHROMA_TENANT'),
            chroma_database=os.getenv('CHROMA_DATABASE')
        )
    
    def is_cloud_configured(self) -> bool:
        """Check if Chroma Cloud configuration is available."""
        return (
            self.chroma_api_key is not None and
            self.chroma_tenant is not None and
            self.chroma_database is not None
        )
    
    def validate_configuration(self) -> None:
        """Validate that Chroma Cloud configuration is complete."""
        if not self.is_cloud_configured():
            missing = []
            if not self.chroma_api_key:
                missing.append("CHROMA_API_KEY")
            if not self.chroma_tenant:
                missing.append("CHROMA_TENANT")
            if not self.chroma_database:
                missing.append("CHROMA_DATABASE")
            
            raise ValueError(
                f"Chroma Cloud configuration incomplete. Missing: {', '.join(missing)}. "
                f"Please set these environment variables in your .env file."
            )
