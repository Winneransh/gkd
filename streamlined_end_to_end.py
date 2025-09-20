"""
streamlined_end_to_end.py

Clean end-to-end pipeline from query analyzer to final synthesis.
No document classification, no intermediate logging, only final results.
FIXED: Import names, method names, and API key management.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Suppress all logging except critical errors
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('langchain').setLevel(logging.CRITICAL)
logging.getLogger('chromadb').setLevel(logging.CRITICAL)

# Import pipeline components - FIXED IMPORTS
from query_analyzer import LegalDocumentQueryAnalyzer
from search_query_generator import SearchQueryGenerator
from llm_enhanced_rag import LLMEnhancedRAG
from consensus_evaluator import ConsensusEvaluator
from contradiction_resolver import EnhancedContradictionResolver
from final_synthesizer import UnifiedFinalAnswerSynthesizer  # FIXED: Was FinalAnswerSynthesizer
from chroma_helper import chroma_collection_has_data

class StreamlinedPipeline:
    """
    Streamlined end-to-end pipeline with minimal logging and clean output.
    """
    
    def __init__(self, google_api_key: str, document_type: str = "Offer Letter"):  # FIXED: Single API key
        """
        Initialize pipeline with single API key and document type.
        
        Args:
            google_api_key: Single Google API key for all components
            document_type: Type of document (default: Offer Letter)
        """
        self.google_api_key = google_api_key
        self.document_type = document_type
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Initialize all components with same API key - FIXED: Simplified API key management
        self.query_analyzer = LegalDocumentQueryAnalyzer(self.google_api_key)
        self.search_generator = SearchQueryGenerator(self.google_api_key)
        self.rag_system = LLMEnhancedRAG(self.google_api_key, embedding_model=self.embedding_model)
        self.consensus_evaluator = ConsensusEvaluator([self.google_api_key], embedding_model=self.embedding_model)
        self.contradiction_resolver = EnhancedContradictionResolver(self.google_api_key, embedding_model=self.embedding_model)
        self.final_synthesizer = UnifiedFinalAnswerSynthesizer(self.google_api_key, embedding_model=self.embedding_model)  # FIXED: Class name
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query through complete pipeline and return final results only.
        
        Args:
            user_query: The user's question
            
        Returns:
            Final pipeline results with minimal metadata
        """
        try:
            # Create mock classification result (no document classifier)
            classification_result = {
                'success': True,
                'document_name': 'existing_document.pdf',
                'classification': {
                    'document_type': self.document_type,
                    'confidence': 98,
                    'key_indicators': ['legal document', 'contract', 'agreement'],
                    'reasoning': 'Using existing ChromaDB with preprocessed documents'
                },
                'document_text': 'Document content available in ChromaDB',
                'processed_at': datetime.now().isoformat()
            }
            
            # Step 1: Query Analysis
            query_analysis = self.query_analyzer.analyze_query(user_query, classification_result)
            if 'error' in query_analysis:
                return self._error_response(f"Query analysis failed: {query_analysis['error']}")
            
            # Step 2: Search Angle Generation
            search_angles = self.search_generator.generate_search_angles(query_analysis)
            if 'error' in search_angles:
                return self._error_response(f"Search angle generation failed: {search_angles['error']}")
            
            # Step 3: RAG Processing
            rag_results = self.rag_system.process_all_search_angles(search_angles)
            if 'error' in rag_results:
                return self._error_response(f"RAG processing failed: {rag_results['error']}")
            
            # Step 4: Consensus Evaluation
            consensus_results = self.consensus_evaluator.evaluate_consensus(rag_results)
            if 'error' in consensus_results:
                return self._error_response(f"Consensus evaluation failed: {consensus_results['error']}")
            
            # Step 5: Contradiction Resolution (if needed)
            resolution_results = None
            contradictory_queries = consensus_results.get('queries_needing_refinement', [])
            
            if contradictory_queries:
                resolution_results = self.contradiction_resolver.resolve_contradictions(consensus_results)
                if 'error' in resolution_results:
                    return self._error_response(f"Contradiction resolution failed: {resolution_results['error']}")
            
            # Step 6: Final Synthesis - FIXED: Method name
            final_answer = self.final_synthesizer.synthesize_unified_answer(  # FIXED: Was synthesize_final_answer
                user_question=user_query,
                rag_output=rag_results,
                consensus_output=consensus_results,
                resolution_output=resolution_results
            )
            
            if 'error' in final_answer:
                return self._error_response(f"Final synthesis failed: {final_answer['error']}")
            
            # Return clean final results - FIXED: Access correct field name
            return {
                'user_query': user_query,
                'document_type': self.document_type,
                'final_answer': final_answer['unified_final_answer'],  # FIXED: Was 'final_answer'
                'quality_metrics': final_answer.get('quality_metrics', {}),
                'processing_summary': {
                    'queries_analyzed': len(query_analysis.get('single_queries', [])) + len(query_analysis.get('hybrid_queries', [])),
                    'search_angles_generated': search_angles.get('total_angles_generated', 0),
                    'successful_answers': rag_results.get('successful_answers', 0),
                    'consensus_success_rate': consensus_results.get('overall_summary', {}).get('success_rate', 0),
                    'contradictions_resolved': len(contradictory_queries) if contradictory_queries else 0,
                    'enhanced_features_used': resolution_results.get('web_searches_performed', 0) + resolution_results.get('duration_calculations_performed', 0) if resolution_results else 0
                },
                'success': True,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._error_response(f"Pipeline error: {str(e)}")
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            'user_query': '',
            'document_type': self.document_type,
            'final_answer': 'An error occurred during processing. Please try again.',
            'error': error_message,
            'success': False,
            'processed_at': datetime.now().isoformat()
        }

def main():
    """Main execution function."""
    
    # FIXED: Single API key configuration
    GOOGLE_API_KEY = 'your_google_api_key_here'  # Replace with your actual Google API key
    
    # Check API key
    if not GOOGLE_API_KEY or 'your_google_api_key' in GOOGLE_API_KEY:
        return {
            'error': 'Please set your Google API key in GOOGLE_API_KEY variable',
            'success': False
        }
    
    # Check Chroma Cloud collection
    if not chroma_collection_has_data("legal_documents"):
        return {
            'error': 'Chroma Cloud collection not found or empty. Please process documents first.',
            'success': False
        }
    
    # Initialize pipeline
    pipeline = StreamlinedPipeline(GOOGLE_API_KEY, document_type="Offer Letter")
    
    # Process query
    user_query = "What is the stipend and total duration of internship?"
    result = pipeline.process_query(user_query)
    
    return result

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))