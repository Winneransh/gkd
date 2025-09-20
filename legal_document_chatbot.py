"""
legal_document_chatbot.py

Simple interactive chatbot using the streamlined pipeline with basic context.
FIXED: API key management and import compatibility.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
import streamlined_end_to_end
from chroma_helper import chroma_collection_has_data

class LegalDocumentChatbot:
    """
    Simple interactive chatbot for legal document queries.
    """
    
    def __init__(self, google_api_key: str, document_type: str = "Offer Letter"):  # FIXED: Single API key
        """
        Initialize chatbot with pipeline.
        
        Args:
            google_api_key: Single Google API key for all components
            document_type: Type of document to analyze
        """
        self.pipeline = streamlined_end_to_end.StreamlinedPipeline(google_api_key, document_type)  # FIXED: Pass single key
        self.document_type = document_type
        self.session_context = []  # Simple list to store Q&A pairs
        
    def chat(self, user_query: str) -> str:
        """
        Process user query and return formatted response.
        
        Args:
            user_query: User's question
            
        Returns:
            Formatted response string
        """
        # Process query through pipeline
        response = self.pipeline.process_query(user_query)
        
        # Add to simple context
        self.session_context.append({
            'user': user_query,
            'bot': response.get('final_answer', 'Error occurred'),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        # Format response
        if response.get('success', False):
            return response.get('final_answer', 'No answer generated')
        else:
            return f"Error: {response.get('error', 'Unknown error')}"
    
    def get_context(self) -> str:
        """Get recent conversation context."""
        if not self.session_context:
            return "No previous questions in this session."
        
        context_text = "Recent Questions:\n"
        for i, exchange in enumerate(self.session_context[-3:], 1):  # Last 3 exchanges
            context_text += f"{i}. [{exchange['timestamp']}] {exchange['user']}\n"
        
        return context_text
    
    def start_chat(self):
        """Start interactive chat session."""
        print(f"\nLegal Document Assistant ({self.document_type})")
        print("=" * 50)
        print("Ask questions about your document or type 'quit' to exit")
        print("Type 'context' to see recent questions")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"\nSession ended. Asked {len(self.session_context)} questions.")
                    break
                
                if user_input.lower() == 'context':
                    print(f"\n{self.get_context()}")
                    continue
                
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                # Get response
                print("\nProcessing...")
                response = self.chat(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print(f"\nSession ended. Asked {len(self.session_context)} questions.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

def main():
    """Main function to start chatbot."""
    
    # FIXED: Single API key configuration
    GOOGLE_API_KEY = 'your_google_api_key_here'  # Replace with your actual Google API key
    
    # Check API key
    if not GOOGLE_API_KEY or 'your_google_api_key' in GOOGLE_API_KEY:
        print("Please set your Google API key in GOOGLE_API_KEY variable")
        return
    
    # Check Chroma Cloud collection
    if not chroma_collection_has_data("legal_documents"):
        print("Chroma Cloud collection not found or empty. Please upload/process documents first.")
        return
    
    # Initialize and start chatbot
    chatbot = LegalDocumentChatbot(GOOGLE_API_KEY, document_type="Offer Letter")  # FIXED: Pass single key
    chatbot.start_chat()

if __name__ == "__main__":
    main()