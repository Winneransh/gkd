"""
app.py

FastAPI application with document upload and chatbot routes.
"""

import os
import json
from datetime import datetime
from typing import Optional
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import pipeline components
from document_classifier import LegalDocumentClassifier
import legal_document_chatbot

app = FastAPI(title="Legal Document Assistant API", version="1.0.0")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# API Keys Configuration
API_KEYS = {
    'document_classifier': 'your_api_key_0_here',
    'query_analyzer': 'your_api_key_1_here',
    'search_generator': 'your_api_key_2_here',
    'rag_system': 'your_api_key_3_here',
    'consensus_evaluator': 'your_api_key_4_here',
    'contradiction_resolver': 'your_api_key_5_here',
    'final_synthesizer': 'your_api_key_6_here'
}

# Global variables
current_document_type = "Unknown"
chatbot_instance = None

# Pydantic models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    success: bool
    error: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    document_type: str
    confidence: float
    message: str
    error: Optional[str] = None

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_api_keys() -> bool:
    """Check if all API keys are configured."""
    missing_keys = [key for key, value in API_KEYS.items() if 'your_api_key' in value]
    return len(missing_keys) == 0

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Legal Document Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload - POST - Upload PDF document for analysis",
            "chat": "/chat - POST - Ask questions about uploaded document"
        }
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and classify a PDF document.
    
    Args:
        file: PDF file to upload and analyze
        
    Returns:
        Document classification results
    """
    global current_document_type, chatbot_instance
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        if not check_api_keys():
            raise HTTPException(status_code=500, detail="API keys not configured")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Write file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            buffer.write(content)
        
        # Initialize document classifier
        classifier = LegalDocumentClassifier(API_KEYS['document_classifier'])
        
        # Process and classify document
        result = classifier.process_pdf_and_classify(file_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        if result.get('success', False):
            classification = result['classification']
            current_document_type = classification['document_type']
            
            # Initialize chatbot with new document type
            chatbot_api_keys = {k: v for k, v in API_KEYS.items() if k != 'document_classifier'}
            chatbot_instance = legal_document_chatbot.LegalDocumentChatbot(
                chatbot_api_keys, 
                document_type=current_document_type
            )
            
            return UploadResponse(
                success=True,
                document_type=classification['document_type'],
                confidence=classification['confidence'],
                message=f"Document successfully classified as: {classification['document_type']}"
            )
        else:
            error_msg = result.get('error', 'Unknown classification error')
            return UploadResponse(
                success=False,
                document_type="Unknown",
                confidence=0.0,
                message="Document classification failed",
                error=error_msg
            )
            
    except HTTPException:
        raise
    except Exception as e:
        return UploadResponse(
            success=False,
            document_type="Unknown", 
            confidence=0.0,
            message="Upload processing failed",
            error=str(e)
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """
    Chat with the uploaded document.
    
    Args:
        request: Chat request with question
        
    Returns:
        Chat response with answer
    """
    global chatbot_instance
    
    try:
        if not check_api_keys():
            raise HTTPException(status_code=500, detail="API keys not configured")
        
        if not os.path.exists("./chroma_db"):
            raise HTTPException(status_code=500, detail="No document processed. Please upload a document first.")
        
        if chatbot_instance is None:
            raise HTTPException(status_code=400, detail="Please upload a document first.")
        
        # Process question
        answer = chatbot_instance.chat(request.question)
        
        return ChatResponse(
            answer=answer,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return ChatResponse(
            answer="Sorry, I encountered an error processing your question.",
            success=False,
            error=str(e)
        )

def secure_filename(filename: str) -> str:
    """Secure filename by removing unsafe characters."""
    filename = filename.replace(" ", "_")
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return "".join(c for c in filename if c in allowed_chars)

if __name__ == "__main__":
    print("🚀 Starting Legal Document Assistant API...")
    print("📋 API Documentation available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )