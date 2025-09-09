"""
streamlined_end_to_end.py

Clean end-to-end pipeline from query analyzer to final synthesis.
No document classification, no intermediate logging, only final results.
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

# Import pipeline components
from query_analyzer import LegalDocumentQueryAnalyzer
from search_query_generator import SearchQueryGenerator
from llm_enhanced_rag import LLMEnhancedRAG
from consensus_evaluator import ConsensusEvaluator
from contradiction_resolver import EnhancedContradictionResolver
from final_synthesizer import FinalAnswerSynthesizer

class StreamlinedPipeline:
    """
    Streamlined end-to-end pipeline with minimal logging and clean output.
    """
    
    def __init__(self, api_keys: Dict[str, str], document_type: str = "Offer Letter"):
        """
        Initialize pipeline with API keys and document type.
        
        Args:
            api_keys: Dictionary with API keys for each component
            document_type: Type of document (default: Offer Letter)
        """
        self.api_keys = api_keys
        self.document_type = document_type
        
        # Initialize all components silently
        self.query_analyzer = LegalDocumentQueryAnalyzer(api_keys['query_analyzer'])
        self.search_generator = SearchQueryGenerator(api_keys['search_generator'])
        self.rag_system = LLMEnhancedRAG(api_keys['rag_system'])
        self.consensus_evaluator = ConsensusEvaluator([api_keys['consensus_evaluator']])
        self.contradiction_resolver = EnhancedContradictionResolver(api_keys['contradiction_resolver'])
        self.final_synthesizer = FinalAnswerSynthesizer(api_keys['final_synthesizer'])
    
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
            
            # Step 6: Final Synthesis
            final_answer = self.final_synthesizer.synthesize_final_answer(
                user_question=user_query,
                rag_output=rag_results,
                consensus_output=consensus_results,
                resolution_output=resolution_results
            )
            
            if 'error' in final_answer:
                return self._error_response(f"Final synthesis failed: {final_answer['error']}")
            
            # Return clean final results
            return {
                'user_query': user_query,
                'document_type': self.document_type,
                'final_answer': final_answer['final_answer'],
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
        return {
            'error': f'Missing API keys: {missing_keys}',
            'success': False
        }
    
    # Check ChromaDB
    if not os.path.exists("./chroma_db"):
        return {
            'error': 'ChromaDB folder not found. Please process documents first.',
            'success': False
        }
    
    # Initialize pipeline
    pipeline = StreamlinedPipeline(API_KEYS, document_type="Offer Letter")
    
    # Process query
    user_query = "What is the stipend and total duration of internship?"
    result = pipeline.process_query(user_query)
    
    return result

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))