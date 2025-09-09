"""
legal_document_chatbot.py

Simple interactive chatbot using the streamlined pipeline with basic context.
"""

import json
from datetime import datetime
from typing import List, Dict, Any
import streamlined_end_to_end

class LegalDocumentChatbot:
    """
    Simple interactive chatbot for legal document queries.
    """
    
    def __init__(self, api_keys: Dict[str, str], document_type: str = "Offer Letter"):
        """
        Initialize chatbot with pipeline.
        
        Args:
            api_keys: Dictionary with API keys for each component
            document_type: Type of document to analyze
        """
        self.pipeline = streamlined_end_to_end.StreamlinedPipeline(api_keys, document_type)
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
            return f"❌ {response.get('error', 'Unknown error')}"
    
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
        print(f"\n🏛️  Legal Document Assistant ({self.document_type})")
        print("=" * 50)
        print("Ask questions about your document or type 'quit' to exit")
        print("Type 'context' to see recent questions")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"\n👋 Session ended. Asked {len(self.session_context)} questions.")
                    break
                
                if user_input.lower() == 'context':
                    print(f"\n📝 {self.get_context()}")
                    continue
                
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                # Get response
                print("\n🤖 Processing...")
                response = self.chat(user_input)
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print(f"\n👋 Session ended. Asked {len(self.session_context)} questions.")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

def main():
    """Main function to start chatbot."""
    
    # API Keys Configuration
    API_KEYS = {
        'query_analyzer': 'your_api_key_1_here',
        'search_generator': 'your_api_key_2_here', 
        'rag_system': 'your_api_key_3_here',
        'consensus_evaluator': 'your_api_key_4_here',
        'contradiction_resolver': 'your_api_key_5_here',
        'final_synthesizer': 'your_api_key_6_here'
    }
    
    # Check API keys
    missing_keys = [key for key, value in API_KEYS.items() if 'your_api_key' in value]
    if missing_keys:
        print(f"❌ Please set API keys for: {missing_keys}")
        return
    
    # Check ChromaDB
    if not os.path.exists("./chroma_db"):
        print("❌ ChromaDB folder not found. Please process documents first.")
        return
    
    # Initialize and start chatbot
    chatbot = LegalDocumentChatbot(API_KEYS, document_type="Offer Letter")
    chatbot.start_chat()

if __name__ == "__main__":
    import os
    main()