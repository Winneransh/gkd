"""
app.py - Production FastAPI with persistent ChromaDB storage
"""

import os
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Your existing imports (unchanged)
from document_classifier import LegalDocumentClassifier
import legal_document_chatbot

app = FastAPI(title="Legal Document Assistant", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = Path('uploads')
GOOGLE_API_KEY = "AIzaSyBTyuRM-0x_T3Fs3ornVKvnvyM417GTOcc"

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Global state
current_document_type = "Unknown"
chatbot_instance = None

# Models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    success: bool

class UploadResponse(BaseModel):
    success: bool
    document_type: str
    confidence: float
    message: str

# Utilities
def check_api_key():
    return GOOGLE_API_KEY and 'your_google_api_key' not in GOOGLE_API_KEY

def chromadb_exists():
    return Path("./chroma_db").exists() and any(Path("./chroma_db").iterdir())

# Routes
@app.get("/")
async def root():
    return {"message": "Legal Document Assistant API", "status": "running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    global current_document_type, chatbot_instance
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    if not check_api_key():
        raise HTTPException(500, "API key not configured")
    
    # Save temp file
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_path = UPLOAD_FOLDER / filename
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Process with your classifier (auto-creates ChromaDB)
        classifier = LegalDocumentClassifier(GOOGLE_API_KEY)
        result = classifier.process_pdf_and_classify(str(file_path))
        
        # Clean up temp file
        file_path.unlink()
        
        if result.get('success'):
            current_document_type = result['classification']['document_type']
            
            # Initialize chatbot with persistent ChromaDB
            chatbot_instance = legal_document_chatbot.LegalDocumentChatbot(
                GOOGLE_API_KEY, 
                document_type=current_document_type
            )
            
            return UploadResponse(
                success=True,
                document_type=current_document_type,
                confidence=result['classification']['confidence'],
                message=f"Document processed. ChromaDB stored with {result['total_chunks']} chunks."
            )
        else:
            return UploadResponse(
                success=False,
                document_type="Unknown",
                confidence=0,
                message="Processing failed"
            )
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    if not chromadb_exists():
        raise HTTPException(400, "No documents uploaded. ChromaDB not found.")
    
    if not chatbot_instance:
        raise HTTPException(400, "Chatbot not initialized. Upload document first.")
    
    try:
        answer = chatbot_instance.chat(request.question)
        return ChatResponse(answer=answer, success=True)
    except Exception as e:
        return ChatResponse(answer=f"Error: {str(e)}", success=False)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)