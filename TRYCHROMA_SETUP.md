# Trychroma Cloud Setup Guide

This guide will help you migrate from local ChromaDB to Trychroma cloud service.

## Option 1: Trychroma Cloud Service

### Step 1: Sign Up
1. Visit [https://trychroma.com](https://trychroma.com)
2. Create an account or sign in
3. Create a new workspace/project

### Step 2: Get Connection Details
From your Trychroma dashboard, you'll typically get:

- **API Token**: Your authentication key
- **Tenant**: Your tenant name (e.g., `default_tenant` or your custom tenant)
- **Database**: Your database name (e.g., `default_database` or your custom database)

**Note**: The server host is usually standardized for Trychroma cloud service.

### Step 3: Update Environment Variables
Copy `.env.example` to `.env` and update with your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
CHROMA_USE_LOCAL=false
CHROMA_SERVER_HOST=api.trychroma.com
CHROMA_SERVER_PORT=443
CHROMA_API_TOKEN=your_actual_api_token_here
CHROMA_TENANT=your_tenant_name
CHROMA_DATABASE=your_database_name
GOOGLE_API_KEY=your_google_api_key_here
```

**Common Trychroma host URLs to try:**
- `api.trychroma.com` (most common)
- `cloud.trychroma.com`
- `app.trychroma.com`

## Option 2: Self-Hosted Chroma Server

If you prefer to host your own Chroma server:

### Using Docker
```bash
# Run Chroma server
docker run -p 8000:8000 ghcr.io/chroma-core/chroma:latest

# Your .env configuration:
CHROMA_USE_LOCAL=false
CHROMA_SERVER_HOST=localhost
CHROMA_SERVER_PORT=8000
CHROMA_API_TOKEN=  # Leave empty for local server
```

### Using Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  chroma:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/.chroma
volumes:
  chroma_data:
```

Run: `docker-compose up -d`

## Option 3: Chroma Cloud (Official)

If available in your region:

1. Visit [https://chroma.com](https://chroma.com)
2. Sign up for Chroma Cloud
3. Follow their setup instructions
4. Update your `.env` with their provided credentials

## Migration Process

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
Update your `.env` file with the appropriate credentials from above.

### Step 3: Run Migration Script
```bash
python migrate_to_trychroma.py
```

This script will:
- Check your configuration
- Connect to both local and cloud ChromaDB
- Export all collections from local
- Import them to the cloud service
- Verify the migration

### Step 4: Test Your Application
```bash
python app.py
```

### Step 5: Clean Up (Optional)
After confirming everything works:
```bash
# Remove local ChromaDB directory
rm -rf ./chroma_db
```

## Troubleshooting

### Connection Issues
- Verify your API token is correct
- Check if the server host includes the protocol (it shouldn't - just the hostname)
- Ensure the port is correct (443 for HTTPS, 8000 for local)

### Authentication Issues  
- Double-check your API token
- Make sure the token has the correct permissions
- Try regenerating the token

### Migration Issues
- Check that your local ChromaDB exists and has data
- Verify network connectivity to the cloud service
- Check the migration script logs for detailed error messages

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `CHROMA_USE_LOCAL` | Use local ChromaDB (true) or cloud (false) | `false` |
| `CHROMA_SERVER_HOST` | Cloud server hostname | `workspace-123.trychroma.com` |
| `CHROMA_SERVER_PORT` | Server port | `443` |
| `CHROMA_API_TOKEN` | Authentication token | `your_token_here` |
| `CHROMA_TENANT` | Tenant name | `default_tenant` |
| `CHROMA_DATABASE` | Database name | `default_database` |
| `GOOGLE_API_KEY` | Google Gemini API key | `your_google_key` |

## Testing Connection

You can test your connection with this simple script:

```python
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

try:
    client = chromadb.HttpClient(
        host=os.getenv('CHROMA_SERVER_HOST'),
        port=int(os.getenv('CHROMA_SERVER_PORT', '443')),
        ssl=True,
        headers={"Authorization": f"Bearer {os.getenv('CHROMA_API_TOKEN')}"}
    )
    
    collections = client.list_collections()
    print(f"✅ Connected successfully! Found {len(collections)} collections.")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
```
