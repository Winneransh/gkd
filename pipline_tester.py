"""
pipeline_tester.py

Simple pipeline testing script for a single hardcoded query and document.
Saves complete, non-truncated results from all 6 steps to a detailed JSON file.
Query: "What is the stipend and total duration of internship?"
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePipelineTester:
    """
    Simple pipeline tester for single query and document with complete JSON output.
    """
    
    def __init__(self, google_api_key: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the pipeline tester.
        
        Args:
            google_api_key: Google API key for Gemini
            embedding_model: HuggingFace embedding model name
        """
        self.google_api_key = google_api_key
        self.embedding_model = embedding_model
        
        # Hardcoded configuration
        self.pdf_path = "document.pdf"  # Change this to your document path
        self.user_query = "What is the stipend and total duration of internship?"
        
        # Initialize all pipeline components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Import all required modules
            from document_classifier import LegalDocumentClassifier
            from query_analyzer import LegalDocumentQueryAnalyzer  
            from search_query_generator import SearchQueryGenerator
            from llm_enhanced_rag import LLMEnhancedRAG
            from consensus_evaluator import ConsensusEvaluator
            from contradiction_resolver import EnhancedContradictionResolver
            from final_synthesizer import UnifiedFinalAnswerSynthesizer
            
            # Initialize components
            self.classifier = LegalDocumentClassifier(
                self.google_api_key, 
                embedding_model=self.embedding_model
            )
            
            self.analyzer = LegalDocumentQueryAnalyzer(self.google_api_key)
            self.generator = SearchQueryGenerator(self.google_api_key)
            
            self.rag_system = LLMEnhancedRAG(
                self.google_api_key, 
                embedding_model=self.embedding_model
            )
            
            self.evaluator = ConsensusEvaluator(
                [self.google_api_key], 
                embedding_model=self.embedding_model
            )
            
            self.resolver = EnhancedContradictionResolver(
                self.google_api_key, 
                embedding_model=self.embedding_model
            )
            
            self.synthesizer = UnifiedFinalAnswerSynthesizer(
                self.google_api_key, 
                embedding_model=self.embedding_model
            )
            
            print("✅ All pipeline components initialized successfully")
            
        except ImportError as e:
            logger.error(f"❌ Import Error: {e}")
            raise ImportError(
                "Please ensure all required modules are available:\n"
                "  - document_classifier.py\n"
                "  - query_analyzer.py\n"
                "  - search_query_generator.py\n"
                "  - llm_enhanced_rag.py\n"
                "  - consensus_evaluator.py\n"
                "  - contradiction_resolver.py\n"
                "  - final_synthesizer.py"
            )
    
    def run_single_pipeline_test(self) -> Dict[str, Any]:
        """
        Run the complete 6-step pipeline for the hardcoded query and document.
        
        Returns:
            Complete pipeline results with all step details
        """
        print("🚀 RUNNING PIPELINE FOR STIPEND AND DURATION ANALYSIS")
        print("=" * 70)
        print(f"📄 Document: {self.pdf_path}")
        print(f"❓ Query: {self.user_query}")
        print("=" * 70)
        
        # Initialize results container
        complete_results = {
            'pipeline_metadata': {
                'document_path': self.pdf_path,
                'user_query': self.user_query,
                'embedding_model': self.embedding_model,
                'processing_start_time': datetime.now().isoformat(),
                'pipeline_version': '2.0_complete_json_export'
            },
            'step_results': {},
            'pipeline_success': False,
            'final_answer': '',
            'processing_summary': {}
        }
        
        try:
            # Validate PDF exists
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
            # STEP 1: Document Classification
            print("\n📋 STEP 1: Document Classification...")
            step1_result = self.classifier.process_pdf_and_classify(self.pdf_path)
            complete_results['step_results']['step_1_classification'] = step1_result
            
            if not step1_result.get("success"):
                raise Exception(f"Classification failed: {step1_result.get('error', 'Unknown error')}")
            
            doc_type = step1_result.get('classification', {}).get('document_type', 'Unknown')
            confidence = step1_result.get('classification', {}).get('confidence', 0)
            chunks = step1_result.get('total_chunks', 0)
            
            print(f"   ✅ Document Type: {doc_type} (Confidence: {confidence:.1f}%)")
            print(f"   📄 Total Chunks: {chunks}")
            
            # STEP 2: Query Analysis
            print("\n🔍 STEP 2: Query Analysis...")
            step2_result = self.analyzer.analyze_query(self.user_query, step1_result)
            complete_results['step_results']['step_2_query_analysis'] = step2_result
            
            if "error" in step2_result:
                raise Exception(f"Query analysis failed: {step2_result['error']}")
            
            single_queries = len(step2_result.get('single_queries', []))
            hybrid_queries = len(step2_result.get('hybrid_queries', []))
            
            print(f"   ✅ Single Queries Generated: {single_queries}")
            print(f"   🔄 Hybrid Queries Generated: {hybrid_queries}")
            
            # STEP 3: Search Angle Generation
            print("\n🎯 STEP 3: Search Angle Generation...")
            step3_result = self.generator.generate_search_angles(step2_result)
            complete_results['step_results']['step_3_search_angles'] = step3_result
            
            if "error" in step3_result:
                raise Exception(f"Search angles generation failed: {step3_result['error']}")
            
            total_angles = step3_result.get('total_angles_generated', 0)
            print(f"   ✅ Total Search Angles Generated: {total_angles}")
            
            # STEP 4A: Individual RAG Processing
            print("\n🤖 STEP 4A: Individual RAG Processing...")
            step4a_result = self.rag_system.process_all_search_angles(step3_result)
            complete_results['step_results']['step_4a_rag_processing'] = step4a_result
            
            if "error" in step4a_result:
                raise Exception(f"RAG processing failed: {step4a_result['error']}")
            
            processed_angles = step4a_result.get('total_search_angles_processed', 0)
            successful_answers = step4a_result.get('successful_answers', 0)
            success_rate = (successful_answers / processed_angles * 100) if processed_angles > 0 else 0
            
            print(f"   ✅ Search Angles Processed: {processed_angles}")
            print(f"   📊 Successful Answers: {successful_answers} ({success_rate:.1f}% success rate)")
            
            # STEP 4B: Consensus Evaluation
            print("\n⚖️ STEP 4B: Consensus Evaluation...")
            step4b_result = self.evaluator.evaluate_consensus(step4a_result)
            complete_results['step_results']['step_4b_consensus_evaluation'] = step4b_result
            
            if "error" in step4b_result:
                raise Exception(f"Consensus evaluation failed: {step4b_result['error']}")
            
            evaluated_queries = len(step4b_result.get('consensus_evaluations', {}))
            verdicts_summary = step4b_result.get('overall_summary', {}).get('verdicts_summary', {})
            
            print(f"   ✅ Queries Evaluated: {evaluated_queries}")
            print(f"   📈 Verdict Summary: {dict(verdicts_summary)}")
            
            # STEP 5: Enhanced Resolution
            print("\n🔧 STEP 5: Enhanced Resolution...")
            step5_result = self.resolver.resolve_contradictions(step4b_result)
            complete_results['step_results']['step_5_enhanced_resolution'] = step5_result
            
            contradictory_queries = step5_result.get('contradictory_queries_found', 0)
            resolved_queries = step5_result.get('successfully_resolved', 0)
            
            if "error" not in step5_result:
                print(f"   ✅ Contradictory Queries Found: {contradictory_queries}")
                print(f"   🎯 Successfully Resolved: {resolved_queries}")
            else:
                print(f"   ⚠️ Resolution completed with warnings")
            
            # STEP 6: Unified Final Synthesis
            print("\n📝 STEP 6: Unified Final Synthesis...")
            step6_result = self.synthesizer.synthesize_unified_answer(
                self.user_query, step4a_result, step4b_result, step5_result
            )
            complete_results['step_results']['step_6_unified_synthesis'] = step6_result
            
            if not step6_result.get("synthesis_success"):
                raise Exception(f"Synthesis failed: {step6_result.get('error', 'Unknown error')}")
            
            quality_score = step6_result.get('quality_metrics', {}).get('processing_metrics', {}).get('synthesis_quality_score', 0)
            final_answer = step6_result.get('unified_final_answer', 'No answer generated')
            
            print(f"   ✅ Synthesis Quality Score: {quality_score:.1f}%")
            
            # Update final results
            complete_results['pipeline_success'] = True
            complete_results['final_answer'] = final_answer
            complete_results['processing_summary'] = {
                'document_type': doc_type,
                'document_confidence': confidence,
                'total_chunks_processed': chunks,
                'search_angles_generated': total_angles,
                'search_angles_processed': processed_angles,
                'successful_rag_answers': successful_answers,
                'rag_success_rate': round(success_rate, 2),
                'queries_evaluated_by_consensus': evaluated_queries,
                'contradictory_queries_found': contradictory_queries,
                'contradictions_resolved': resolved_queries,
                'final_synthesis_quality': round(quality_score, 2),
                'pipeline_completion_time': datetime.now().isoformat()
            }
            
            # Display final results
            print("\n" + "=" * 70)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            
            print(f"\n📋 Document Type: {doc_type}")
            print(f"❓ Question: {self.user_query}")
            print(f"\n💡 FINAL ANSWER:")
            print(f"   {final_answer}")
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   • Document Chunks: {chunks}")
            print(f"   • Search Angles Generated: {total_angles}")
            print(f"   • RAG Success Rate: {success_rate:.1f}%")
            print(f"   • Synthesis Quality: {quality_score:.1f}%")
            print(f"   • Contradictions Resolved: {resolved_queries}")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed at some step: {str(e)}")
            complete_results['pipeline_success'] = False
            complete_results['pipeline_error'] = str(e)
            complete_results['error_timestamp'] = datetime.now().isoformat()
            return complete_results
    
    def save_complete_json_results(self, results: Dict[str, Any]) -> str:
        """
        Save complete pipeline results to a detailed JSON file.
        
        Args:
            results: Complete pipeline results dictionary
            
        Returns:
            Filename of the saved JSON file
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stipend_duration_analysis_complete_{timestamp}.json"
            
            # Create enhanced results with additional metadata
            enhanced_results = {
                'analysis_metadata': {
                    'analysis_type': 'Stipend and Duration Analysis',
                    'document_analyzed': self.pdf_path,
                    'query_processed': self.user_query,
                    'embedding_model_used': self.embedding_model,
                    'export_timestamp': datetime.now().isoformat(),
                    'json_version': '2.0_complete_non_truncated'
                },
                'user_question_breakdown': {
                    'original_question': self.user_query,
                    'question_components': [
                        'stipend amount',
                        'internship duration',
                        'total time period'
                    ],
                    'expected_answer_type': 'Financial amount and time duration'
                },
                'complete_pipeline_results': results,
                'step_by_step_details': {
                    'step_1': 'Document Classification - Identifies document type and extracts text chunks',
                    'step_2': 'Query Analysis - Breaks down user question into searchable components', 
                    'step_3': 'Search Angle Generation - Creates multiple search perspectives',
                    'step_4a': 'RAG Processing - Retrieves and answers from document chunks',
                    'step_4b': 'Consensus Evaluation - Validates answer quality and consistency',
                    'step_5': 'Enhanced Resolution - Resolves contradictions and enhances answers',
                    'step_6': 'Unified Synthesis - Creates final single paragraph answer'
                }
            }
            
            # Save with full content, no truncation
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(enhanced_results, f, indent=2, ensure_ascii=False, sort_keys=False)
            
            file_size = os.path.getsize(filename)
            print(f"\n💾 Complete results saved to: {filename}")
            print(f"   📏 File size: {file_size:,} bytes")
            print(f"   📊 Contains complete, non-truncated output from all 6 steps")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving JSON results: {str(e)}")
            return f"Error saving file: {str(e)}"
    
    def run_and_save(self):
        """Run the complete pipeline and save results to JSON."""
        print("🎯 STIPEND AND DURATION ANALYSIS PIPELINE")
        print("📄 Single Document | 🔍 Single Query | 💾 Complete JSON Export")
        
        # Run pipeline
        results = self.run_single_pipeline_test()
        
        # Save complete results
        json_filename = self.save_complete_json_results(results)
        
        # Final status
        if results.get('pipeline_success', False):
            print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"📄 Answer: {results.get('final_answer', 'No answer available')}")
            print(f"💾 Full details in: {json_filename}")
        else:
            print(f"\n❌ ANALYSIS FAILED")
            print(f"❌ Error: {results.get('pipeline_error', 'Unknown error')}")
            print(f"💾 Error details saved to: {json_filename}")


def main():
    """Main function to run the stipend and duration analysis."""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    PDF_PATH = "Suryansh_OL.docx (1) (1) (1).pdf"  # ⚠️ Replace with your document path
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    print("🚀 STIPEND AND DURATION ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"📄 Document: {PDF_PATH}")
    print(f"❓ Query: What is the stipend and total duration of internship?")
    print(f"🤖 Embedding Model: {EMBEDDING_MODEL}")
    print("=" * 60)
    
    # Validate configuration
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.strip() == "":
        print("❌ CONFIGURATION ERROR:")
        print("   Please set your Google API key in the GOOGLE_API_KEY variable")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ DOCUMENT ERROR:")
        print(f"   PDF file not found: {PDF_PATH}")
        print(f"   Please update the PDF_PATH variable to point to your document")
        return
    
    try:
        # Initialize and run pipeline
        tester = SimplePipelineTester(GOOGLE_API_KEY, EMBEDDING_MODEL)
        tester.pdf_path = PDF_PATH  # Update with configured path
        
        # Run complete analysis
        tester.run_and_save()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        print(f"❌ Pipeline failed with error: {str(e)}")


if __name__ == "__main__":
    main()