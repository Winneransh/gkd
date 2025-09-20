import os
from urllib.parse import urlparse
from typing import Optional
from dotenv import load_dotenv
import chromadb
from langchain_community.vectorstores import Chroma

load_dotenv()

def _is_cloud_configured() -> bool:
    """Return True if Chroma Cloud env vars are present."""
    return bool(
        os.getenv("CHROMA_API_KEY") and os.getenv("CHROMA_TENANT") and os.getenv("CHROMA_DATABASE")
    )


def _build_cloud_http_client_from_env() -> chromadb.HttpClient:
    """Build a chromadb.HttpClient for Chroma Cloud using env vars.

    Required env vars:
    - CHROMA_API_KEY
    - CHROMA_TENANT
    - CHROMA_DATABASE

    Optional overrides:
    - CHROMA_URL or CHROMA_HOST/CHROMA_PORT/CHROMA_SSL to override default cloud host
    """
    api_key = os.getenv("CHROMA_API_KEY", "").strip()
    tenant = os.getenv("CHROMA_TENANT", "").strip()
    database = os.getenv("CHROMA_DATABASE", "").strip()

    # Default to Chroma Cloud endpoint unless overridden
    url = os.getenv("CHROMA_URL")
    if url:
        parsed = urlparse(url)
        host = parsed.hostname or "api.trychroma.com"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        ssl = parsed.scheme == "https"
    else:
        host = os.getenv("CHROMA_HOST", "api.trychroma.com")
        port = int(os.getenv("CHROMA_PORT", "443"))
        ssl_env = os.getenv("CHROMA_SSL", "true").lower()
        ssl = ssl_env in ("1", "true", "yes")

    headers = {
        # Common auth forms supported by Chroma Cloud
        "Authorization": f"Bearer {api_key}",
        "X-Chroma-API-Key": api_key,
        "X-Chroma-Api-Key": api_key,
        "X-Api-Key": api_key,
        # Tenant / Database routing
        "X-Chroma-Tenant": tenant,
        "X-Chroma-Database": database,
    }

    return chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)


def _build_http_client_from_env() -> chromadb.HttpClient:
    """Build a chromadb.HttpClient for Chroma Cloud.

    Requires CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE.
    """
    if not _is_cloud_configured():
        raise RuntimeError("Chroma Cloud env vars missing: set CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE")
    return _build_cloud_http_client_from_env()


def get_chroma_vectorstore(collection_name: str, embedding_function) -> Chroma:
    """Create a LangChain Chroma vectorstore backed by Chroma Cloud only."""
    client = _build_http_client_from_env()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        client=client,
    )


def chroma_collection_has_data(collection_name: str = "legal_documents") -> bool:
    """Check whether a Chroma Cloud collection exists and has vectors.

    Returns False on any error or if env vars are missing.
    """
    try:
        client = _build_http_client_from_env()
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            return False
        try:
            count = collection.count()
            return bool(count and count > 0)
        except Exception:
            try:
                items = collection.peek()
                return bool(items and len(items.get("ids", [])) > 0)
            except Exception:
                return False
    except Exception:
        return False


