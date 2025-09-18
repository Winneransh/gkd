"""
final_synthesizer.py

Step 6: Synthesizes outputs from Steps 4A, 4B, and 5 into a single unified paragraph answer.
Saves complete pipeline results to JSON file for non-truncated analysis.
Updated to use HuggingFace embeddings and provide complete pipeline integration.
FIXED: String literal syntax errors corrected.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedFinalAnswerSynthesizer:
    """
    Synthesizes all pipeline outputs into a single, unified paragraph answer that
    consolidates single queries, hybrid queries, and all search angles into one coherent response.
    Compatible with HuggingFace embeddings and saves complete results to JSON.
    """
    
    def __init__(self, google_api_key: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the unified final answer synthesizer.
        
        Args:
            google_api_key: Google API key for Gemini
            embedding_model: HuggingFace embedding model name (for compatibility)
        """
        self.google_api_key = google_api_key
        self.embedding_model_name = embedding_model
        
        # Initialize Gemini 2.0 Flash LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.2,  # Slightly higher for natural language
            max_output_tokens=4096
        )
        
        # Create unified synthesis prompt
        self.unified_synthesis_prompt = self._create_unified_synthesis_prompt()
    
    def _create_unified_synthesis_prompt(self) -> PromptTemplate:
        """
        Create comprehensive synthesis prompt for unified single-paragraph answer generation.
        """
        
        prompt_template = """You are an expert legal document analyst who consolidates ALL analysis results into ONE comprehensive, flowing paragraph answer.

**USER'S ORIGINAL QUESTION:** {user_question}
**DOCUMENT TYPE:** {document_type}

**ALL ANALYSIS SOURCES TO CONSOLIDATE:**

**Single Query Results:**
{single_query_results}

**Hybrid Query Results:**
{hybrid_query_results}

**All Individual Search Angle Answers:**
{all_search_angle_answers}

**Consensus Evaluation Findings:**
{consensus_findings}

**Enhanced Resolution Results:**
{resolution_findings}

**UNIFIED SYNTHESIS INSTRUCTIONS:**

**1. COMPREHENSIVE CONSOLIDATION:**
- Analyze ALL sources: single queries, hybrid queries, search angles, consensus, resolution
- Extract the BEST and most ACCURATE information from across all sources
- Eliminate redundant, contradictory, or low-quality responses
- Prioritize specific, detailed answers with clear evidence
- Use enhanced resolution as authoritative for resolved contradictions

**2. DEDUPLICATION STRATEGY:**
- If multiple sources provide the same information, use it ONCE with highest confidence
- If hybrid queries repeat single query information, prioritize the clearer response
- If search angles give overlapping details, combine into comprehensive single mention
- Ignore "not found" responses when better sources have the information
- Eliminate vague or generic answers when specific details are available

**3. QUALITY ASSESSMENT:**
- HIGH QUALITY: Specific amounts, dates, terms with clear document references
- MEDIUM QUALITY: General information that addresses query but lacks specifics
- LOW QUALITY: Vague responses, contradictions, or "information not available"
- Use only HIGH and MEDIUM quality information, prioritizing HIGH quality

**4. SINGLE PARAGRAPH RESPONSE:**
- Create ONE comprehensive paragraph that flows naturally
- Include ALL relevant factual information that answers the user's question
- Write in conversational, natural language (no bullet points or lists)
- Present information authoritatively and confidently
- Ensure smooth transitions between different pieces of information

**5. INFORMATION INTEGRATION:**
- Start with direct answer to the user's specific question
- Add supporting details and context from the document
- Include all relevant specifics (amounts, dates, conditions, etc.)
- End with any additional relevant context or implications

**EXAMPLE OF DESIRED OUTPUT:**
"Based on your internship offer letter, you will receive a monthly stipend of INR 15,000 throughout the 3-month internship period running from July 1st to September 30th, 2024. Your work schedule will be Monday through Friday from 9 AM to 6 PM, with the flexibility to work remotely up to 2 days per week, and you'll be based primarily at the Bangalore office working on mobile app development projects. The position includes a completion certificate upon successful finish of the internship term, plus a performance-based bonus of up to INR 10,000 depending on your final evaluation, bringing your total potential compensation to INR 55,000 over the three-month period."

**SYNTHESIS APPROACH:**
1. Identify ALL factual details that answer the user's question across all sources
2. Remove duplicate information and prioritize most specific/accurate details
3. Organize information logically within one flowing paragraph
4. Present as definitive answer using the best available information
5. Write in natural, conversational tone without technical language

**CREATE ONE UNIFIED PARAGRAPH ANSWER:**"""

        return PromptTemplate(
            input_variables=["user_question", "document_type", "single_query_results", 
                           "hybrid_query_results", "all_search_angle_answers", 
                           "consensus_findings", "resolution_findings"],
            template=prompt_template
        )
    
    def synthesize_unified_answer(self, user_question: str, rag_output: Dict[str, Any], 
                                 consensus_output: Dict[str, Any], resolution_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synthesize unified single-paragraph answer from all pipeline outputs.
        """
        try:
            # Validate inputs
            if 'error' in rag_output:
                return self._create_error_response(user_question, "RAG processing failed", rag_output['error'])
            
            if 'error' in consensus_output:
                return self._create_error_response(user_question, "Consensus evaluation failed", consensus_output['error'])
            
            # Extract document information
            document_type = rag_output.get('document_type', 'Unknown')
            embedding_model = rag_output.get('embedding_model', self.embedding_model_name)
            
            # Organize all answers by type and extract useful information
            organized_results = self._organize_all_pipeline_results(rag_output, consensus_output, resolution_output)
            
            # Generate unified single paragraph answer
            unified_answer = self._generate_unified_paragraph_answer(
                user_question=user_question,
                document_type=document_type,
                organized_results=organized_results
            )
            
            # Calculate comprehensive quality metrics
            quality_metrics = self._calculate_unified_quality_metrics(
                rag_output, consensus_output, resolution_output, organized_results
            )
            
            return {
                'user_question': user_question,
                'document_type': document_type,
                'embedding_model': embedding_model,
                'unified_final_answer': unified_answer,
                'sources_analyzed': organized_results['source_summary'],
                'quality_metrics': quality_metrics,
                'synthesis_success': True,
                'synthesized_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in unified answer synthesis: {str(e)}")
            return self._create_error_response(user_question, "Synthesis failed", str(e))
    
    def _organize_all_pipeline_results(self, rag_output: Dict[str, Any], consensus_output: Dict[str, Any], 
                                      resolution_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Organize and categorize all pipeline results for synthesis.
        """
        individual_answers = rag_output.get('individual_answers', {})
        
        organized = {
            'single_query_answers': [],
            'hybrid_query_answers': [],
            'all_search_angle_answers': [],
            'consensus_answers': [],
            'resolution_answers': [],
            'source_summary': {
                'total_single_queries': 0,
                'total_hybrid_queries': 0,
                'total_search_angles': 0,
                'consensus_evaluations': 0,
                'resolutions_applied': 0
            }
        }
        
        # Process individual answers from RAG output
        for query_id, query_data in individual_answers.items():
            query_type = query_data.get('query_type', 'unknown')
            original_query = query_data.get('original_query', '')
            angle_answers = query_data.get('angle_answers', {})
            
            # Collect successful answers
            successful_answers = []
            for angle_id, angle_data in angle_answers.items():
                if angle_data.get('answer_success', False):
                    answer = angle_data.get('answer', '').strip()
                    if answer and answer != 'No relevant information found in the document.':
                        successful_answers.append(answer)
                        organized['all_search_angle_answers'].append(answer)
            
            # Categorize by query type
            if 'single' in query_type.lower() and successful_answers:
                organized['single_query_answers'].extend(successful_answers)
                organized['source_summary']['total_single_queries'] += 1
            elif 'hybrid' in query_type.lower() and successful_answers:
                organized['hybrid_query_answers'].extend(successful_answers)
                organized['source_summary']['total_hybrid_queries'] += 1
            
            organized['source_summary']['total_search_angles'] += len(successful_answers)
        
        # Process consensus evaluation results
        consensus_evaluations = consensus_output.get('consensus_evaluations', {})
        for query_id, evaluation in consensus_evaluations.items():
            if evaluation.get('evaluation_success', False):
                final_verdict = evaluation.get('final_verdict', {})
                verdict = final_verdict.get('verdict', '')
                
                if verdict in ['CORRECT', 'SATISFACTORY']:
                    final_answer = final_verdict.get('final_answer', '')
                    if final_answer and final_answer.strip():
                        organized['consensus_answers'].append(final_answer)
                        organized['source_summary']['consensus_evaluations'] += 1
        
        # Process resolution results
        if resolution_output and 'error' not in resolution_output:
            resolutions = resolution_output.get('resolutions', {})
            for query_id, resolution in resolutions.items():
                if resolution.get('resolution_success', False):
                    resolution_answer = resolution.get('resolution_answer', '')
                    if resolution_answer and resolution_answer.strip():
                        organized['resolution_answers'].append(resolution_answer)
                        organized['source_summary']['resolutions_applied'] += 1
        
        return organized
    
    def _generate_unified_paragraph_answer(self, user_question: str, document_type: str, 
                                          organized_results: Dict[str, Any]) -> str:
        """
        Generate unified single paragraph answer using LLM synthesis.
        """
        try:
            # Format all results for prompt
            single_query_text = "\n".join([f"- {answer}" for answer in organized_results['single_query_answers']])
            hybrid_query_text = "\n".join([f"- {answer}" for answer in organized_results['hybrid_query_answers']])
            all_angles_text = "\n".join([f"- {answer}" for answer in organized_results['all_search_angle_answers']])
            consensus_text = "\n".join([f"- {answer}" for answer in organized_results['consensus_answers']])
            resolution_text = "\n".join([f"- {answer}" for answer in organized_results['resolution_answers']])
            
            # Create synthesis prompt
            formatted_prompt = self.unified_synthesis_prompt.format(
                user_question=user_question,
                document_type=document_type,
                single_query_results=single_query_text or "No single query results available",
                hybrid_query_results=hybrid_query_text or "No hybrid query results available",
                all_search_angle_answers=all_angles_text or "No search angle answers available",
                consensus_findings=consensus_text or "No consensus findings available",
                resolution_findings=resolution_text or "No resolution findings available"
            )
            
            # Get unified answer from LLM
            response = self.llm.invoke(formatted_prompt)
            unified_answer = response.content.strip()
            
            # Clean and format as single paragraph
            unified_answer = self._ensure_single_paragraph(unified_answer)
            
            return unified_answer
            
        except Exception as e:
            logger.error(f"Error generating unified paragraph answer: {str(e)}")
            return "I apologize, but I encountered an error while analyzing your document. The information appears to be available, but I'm unable to present it clearly at the moment."
    
    def _ensure_single_paragraph(self, answer: str) -> str:
        """
        Ensure the answer is formatted as a single flowing paragraph.
        """
        # Remove line breaks and join into single paragraph
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        
        # Remove any bullet points or formatting
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.replace('•', '').replace('-', '').replace('*', '').strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Join into single paragraph with proper spacing
        single_paragraph = ' '.join(cleaned_lines)
        
        # Clean up multiple spaces
        import re
        single_paragraph = re.sub(r'\s+', ' ', single_paragraph)
        
        return single_paragraph.strip()
    
    def _calculate_unified_quality_metrics(self, rag_output: Dict[str, Any], consensus_output: Dict[str, Any], 
                                          resolution_output: Optional[Dict[str, Any]], organized_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics for the unified answer.
        """
        try:
            source_summary = organized_results['source_summary']
            
            # Calculate source utilization
            total_sources_available = (
                source_summary['total_single_queries'] + 
                source_summary['total_hybrid_queries'] + 
                source_summary['consensus_evaluations'] + 
                source_summary['resolutions_applied']
            )
            
            sources_with_content = sum([
                1 if organized_results['single_query_answers'] else 0,
                1 if organized_results['hybrid_query_answers'] else 0,
                1 if organized_results['consensus_answers'] else 0,
                1 if organized_results['resolution_answers'] else 0
            ])
            
            source_utilization = (sources_with_content / 4 * 100) if total_sources_available > 0 else 0
            
            # RAG success rate
            total_angles = rag_output.get('total_search_angles_processed', 0)
            successful_angles = rag_output.get('successful_answers', 0)
            rag_success_rate = (successful_angles / total_angles * 100) if total_angles > 0 else 0
            
            # Consensus quality
            consensus_evaluations = consensus_output.get('consensus_evaluations', {})
            high_quality_consensus = sum(1 for eval in consensus_evaluations.values() 
                                       if eval.get('final_verdict', {}).get('verdict') in ['CORRECT', 'SATISFACTORY'])
            consensus_quality = (high_quality_consensus / len(consensus_evaluations) * 100) if consensus_evaluations else 0
            
            # Overall synthesis quality score
            synthesis_quality = (
                (rag_success_rate * 0.4) + 
                (consensus_quality * 0.3) + 
                (source_utilization * 0.3)
            )
            
            return {
                'sources_utilized': {
                    'single_queries': len(organized_results['single_query_answers']),
                    'hybrid_queries': len(organized_results['hybrid_query_answers']),
                    'search_angles': len(organized_results['all_search_angle_answers']),
                    'consensus_findings': len(organized_results['consensus_answers']),
                    'resolutions': len(organized_results['resolution_answers'])
                },
                'processing_metrics': {
                    'rag_success_rate': round(rag_success_rate, 1),
                    'consensus_quality_rate': round(consensus_quality, 1),
                    'source_utilization_rate': round(source_utilization, 1),
                    'synthesis_quality_score': round(synthesis_quality, 1)
                },
                'answer_characteristics': {
                    'format': 'single_paragraph',
                    'completeness': 'High' if synthesis_quality > 80 else 'Medium' if synthesis_quality > 60 else 'Low',
                    'sources_consolidated': total_sources_available,
                    'deduplication_applied': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating unified quality metrics: {str(e)}")
            return {
                'synthesis_quality_score': 0,
                'completeness': 'Error'
            }
    
    def save_complete_results_to_json(self, complete_pipeline_result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save complete pipeline results to JSON file for full non-truncated analysis.
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"complete_legal_document_analysis_{timestamp}.json"
            
            if not filename.endswith('.json'):
                filename += '.json'
            
            # Save with proper formatting and encoding
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(complete_pipeline_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Complete pipeline results saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving complete results JSON: {str(e)}")
            return f"Error saving file: {str(e)}"
    
    def create_complete_pipeline_result(self, user_query: str, classification_result: Dict[str, Any],
                                       query_analysis: Dict[str, Any], search_angles: Dict[str, Any],
                                       rag_results: Dict[str, Any], consensus_results: Dict[str, Any],
                                       resolution_results: Dict[str, Any], unified_synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create complete pipeline result with all step outputs for JSON export.
        """
        # Calculate successful steps first
        successful_steps = sum([
            1 if classification_result.get('success', False) else 0,
            1 if 'error' not in query_analysis else 0,
            1 if 'error' not in search_angles else 0,
            1 if 'error' not in rag_results else 0,
            1 if 'error' not in consensus_results else 0,
            1 if 'error' not in resolution_results else 0
        ])
        
        return {
            'pipeline_metadata': {
                'user_query': user_query,
                'processing_timestamp': datetime.now().isoformat(),
                'pipeline_version': '2.0_unified_synthesis_huggingface',
                'embedding_model': unified_synthesis.get('embedding_model', self.embedding_model_name),
                'processing_stages': 6,
                'synthesis_type': 'unified_single_paragraph'
            },
            'user_question_analysis': {
                'original_question': user_query,
                'question_complexity': 'high' if len(user_query.split()) > 8 else 'medium' if len(user_query.split()) > 4 else 'simple',
                'expected_answer_components': user_query.count(' and ') + user_query.count(' or ') + 1
            },
            'step_1_document_classification': {
                'status': 'success' if classification_result.get('success', False) else 'failed',
                'document_type': classification_result.get('classification', {}).get('document_type', 'Unknown'),
                'confidence': classification_result.get('classification', {}).get('confidence', 0),
                'key_indicators': classification_result.get('classification', {}).get('key_indicators', []),
                'chunks_stored': classification_result.get('total_chunks', 0),
                'full_result': classification_result
            },
            'step_2_query_analysis': {
                'status': 'success' if 'error' not in query_analysis else 'failed',
                'single_queries_generated': len(query_analysis.get('single_queries', [])),
                'hybrid_queries_generated': len(query_analysis.get('hybrid_queries', [])),
                'total_query_components': len(query_analysis.get('single_queries', [])) + len(query_analysis.get('hybrid_queries', [])),
                'query_breakdown': {
                    'single_queries': query_analysis.get('single_queries', []),
                    'hybrid_queries': query_analysis.get('hybrid_queries', [])
                },
                'full_result': query_analysis
            },
            'step_3_search_angle_generation': {
                'status': 'success' if 'error' not in search_angles else 'failed',
                'total_search_angles_generated': search_angles.get('total_angles_generated', 0),
                'angles_per_query': 5,
                'search_angle_breakdown': search_angles.get('search_angles', {}),
                'full_result': search_angles
            },
            'step_4a_individual_rag_processing': {
                'status': 'success' if 'error' not in rag_results else 'failed',
                'search_angles_processed': rag_results.get('total_search_angles_processed', 0),
                'successful_answers': rag_results.get('successful_answers', 0),
                'success_rate': f"{(rag_results.get('successful_answers', 0) / max(rag_results.get('total_search_angles_processed', 1), 1) * 100):.1f}%",
                'individual_answers_detailed': rag_results.get('individual_answers', {}),
                'embedding_model_used': rag_results.get('embedding_model', 'Unknown'),
                'full_result': rag_results
            },
            'step_4b_consensus_evaluation': {
                'status': 'success' if 'error' not in consensus_results else 'failed',
                'queries_evaluated': len(consensus_results.get('consensus_evaluations', {})),
                'verdicts_summary': consensus_results.get('overall_summary', {}).get('verdicts_summary', {}),
                'confidence_distribution': consensus_results.get('overall_summary', {}).get('confidence_distribution', {}),
                'detailed_consensus_evaluations': consensus_results.get('consensus_evaluations', {}),
                'full_result': consensus_results
            },
            'step_5_enhanced_resolution': {
                'status': 'success' if 'error' not in resolution_results else 'failed',
                'contradictory_queries_found': resolution_results.get('contradictory_queries_found', 0),
                'successfully_resolved': resolution_results.get('successfully_resolved', 0),
                'enhanced_features_used': {
                    'web_searches_performed': resolution_results.get('web_searches_performed', 0),
                    'duration_calculations_performed': resolution_results.get('duration_calculations_performed', 0)
                },
                'detailed_resolutions': resolution_results.get('resolutions', {}),
                'full_result': resolution_results
            },
            'step_6_unified_final_synthesis': {
                'status': 'success' if unified_synthesis.get('synthesis_success', False) else 'failed',
                'synthesis_approach': 'unified_single_paragraph',
                'sources_analyzed': unified_synthesis.get('sources_analyzed', {}),
                'quality_metrics': unified_synthesis.get('quality_metrics', {}),
                'unified_answer': unified_synthesis.get('unified_final_answer', ''),
                'full_result': unified_synthesis
            },
            'final_consolidated_answer': {
                'user_question': user_query,
                'document_type': classification_result.get('classification', {}).get('document_type', 'Unknown'),
                'unified_paragraph_answer': unified_synthesis.get('unified_final_answer', 'Answer not available'),
                'answer_quality': unified_synthesis.get('quality_metrics', {}).get('answer_characteristics', {}).get('completeness', 'Unknown'),
                'sources_utilized': unified_synthesis.get('sources_analyzed', {}).get('total_single_queries', 0) + unified_synthesis.get('sources_analyzed', {}).get('total_hybrid_queries', 0),
                'confidence_indicators': {
                    'rag_success_rate': rag_results.get('successful_answers', 0) / max(rag_results.get('total_search_angles_processed', 1), 1) * 100,
                    'consensus_quality': len([e for e in consensus_results.get('consensus_evaluations', {}).values() if e.get('final_verdict', {}).get('verdict') in ['CORRECT', 'SATISFACTORY']]),
                    'resolution_applied': resolution_results.get('successfully_resolved', 0) > 0
                }
            },
            'pipeline_performance_summary': {
                'total_processing_steps': 6,
                'successful_steps': successful_steps,
                'overall_pipeline_success_rate': f"{(successful_steps / 6 * 100):.1f}%",
                'final_synthesis_quality': unified_synthesis.get('quality_metrics', {}).get('processing_metrics', {}).get('synthesis_quality_score', 0),
                'embedding_model_consistency': unified_synthesis.get('embedding_model', 'Unknown'),
                'pipeline_completion_timestamp': datetime.now().isoformat()
            }
        }
    
    def _create_error_response(self, user_question: str, error_type: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'error': error_type,
            'error_details': error_message,
            'user_question': user_question,
            'unified_final_answer': 'I apologize, but I encountered an error while processing your question. Please try rephrasing your question or check if the document contains the information you are looking for.',
            'synthesis_success': False,
            'synthesized_at': datetime.now().isoformat()
        }