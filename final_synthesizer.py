"""
final_synthesizer.py

Step 6: Synthesizes outputs from Steps 4A, 4B, and 5 into clean, coherent
paragraph answers for presentation to users.
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

class FinalAnswerSynthesizer:
    """
    Synthesizes all pipeline outputs into clean, coherent final answers
    formatted as natural paragraphs for user consumption.
    """
    
    def __init__(self, google_api_key: str):
        """
        Initialize the final answer synthesizer.
        
        Args:
            google_api_key: Google API key for Gemini
        """
        self.google_api_key = google_api_key
        
        # Initialize Gemini 2.5 Flash LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.2,  # Slightly higher for natural language
            max_output_tokens=3072
        )
        
        # Create synthesis prompt
        self.synthesis_prompt = self._create_synthesis_prompt()
    
    def _create_synthesis_prompt(self) -> PromptTemplate:
        """
        Create comprehensive synthesis prompt for final answer generation.
        
        Returns:
            PromptTemplate for final answer synthesis
        """
        
        prompt_template = """You are an expert legal document communication specialist. Your job is to synthesize complex analysis results into clear, natural, conversational answers that directly address the user's question.

**USER'S ORIGINAL QUESTION:** {user_question}
**DOCUMENT TYPE:** {document_type}

**ANALYSIS RESULTS TO SYNTHESIZE:**

**Individual Search Results (Step 4A):**
{individual_answers}

**Consensus Evaluation (Step 4B):**
{consensus_evaluation}

**Contradiction Resolution (Step 5):**
{contradiction_resolution}

**SYNTHESIS INSTRUCTIONS:**

**1. ANSWER STRUCTURE:**
- **Direct Answer**: Start with a clear, direct answer to the user's question
- **Supporting Details**: Provide specific information from the document
- **Additional Context**: Include relevant context or caveats if needed
- **Confidence Indication**: Subtly indicate the confidence level without technical jargon

**2. COMMUNICATION STYLE:**
- **Conversational**: Write as if speaking to someone who asked a question
- **Clear and Simple**: Avoid technical terms, jargon, or complex explanations
- **Natural Flow**: Use connecting phrases and smooth transitions
- **User-Focused**: Address exactly what the user asked for

**3. INFORMATION PRIORITIZATION:**
- **Primary**: Use consensus results and resolved contradictions as primary sources
- **Supporting**: Include relevant individual answers that support the main finding
- **Contextual**: Add helpful context from the document when relevant
- **Balanced**: Present complete picture while staying focused on the user's question

**4. CONFIDENCE HANDLING:**
- **High Confidence**: Present answer as definitive with supporting evidence
- **Medium Confidence**: Present answer with appropriate caveats ("based on the document" or "appears to be")
- **Low/Contradictory**: Acknowledge uncertainty and present what information is available
- **Resolved Contradictions**: Present the resolved answer with confidence

**5. DOCUMENT-SPECIFIC GUIDANCE:**

**OFFER LETTER:**
- Be specific about amounts, dates, and terms
- Explain benefits and conditions clearly
- Address work arrangements and expectations
- Clarify legal obligations when relevant

**HEALTH INSURANCE:**
- Clearly explain coverage and limitations
- Specify financial obligations (premiums, deductibles)
- Explain eligibility requirements
- Address geographic or temporal restrictions

**OTHER LEGAL DOCUMENTS:**
- Focus on the specific terms and conditions relevant to the query
- Explain legal implications in simple terms
- Provide context for understanding obligations or rights

**6. RESPONSE FORMAT:**
Write 2-4 natural paragraphs that flow conversationally. Do not use bullet points, numbered lists, or structured sections. Write as if you're explaining the answer to someone in person.

**EXAMPLE RESPONSE STYLE:**
"Based on your offer letter, the monthly stipend for your internship is INR 12,000. This amount is clearly stated in Section 5 of the document and will be paid monthly throughout your 3-month internship period. Additionally, you'll receive a completion bonus of INR 10,000 at the end of the internship, provided you successfully complete the full term.

The payment structure is straightforward - you'll receive the monthly stipend regularly, and the bonus is performance-based according to the document. This brings your total potential compensation to INR 46,000 over the three-month period."

**SYNTHESIS GUIDELINES:**
- If consensus is strong: Present answer confidently
- If contradictions were resolved: Use the resolved answer
- If multiple valid interpretations exist: Explain the different scenarios
- If information is missing: Acknowledge gaps honestly
- Always stay focused on answering the user's specific question

**SYNTHESIZE THE FINAL ANSWER:**"""

        return PromptTemplate(
            input_variables=["user_question", "document_type", "individual_answers", "consensus_evaluation", "contradiction_resolution"],
            template=prompt_template
        )
    
    def synthesize_final_answer(self, user_question: str, rag_output: Dict[str, Any], 
                               consensus_output: Dict[str, Any], resolution_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synthesize final answer from all pipeline outputs.
        
        Args:
            user_question: Original user question
            rag_output: Output from Step 4A (LLMEnhancedRAG)
            consensus_output: Output from Step 4B (ConsensusEvaluator)
            resolution_output: Output from Step 5 (ContradictionResolver), optional
            
        Returns:
            Dictionary containing synthesized final answer
        """
        try:
            # Validate inputs
            if 'error' in rag_output:
                return {
                    'error': 'RAG processing failed',
                    'rag_error': rag_output['error'],
                    'final_answer': 'I apologize, but I encountered an error while processing your question.',
                    'synthesized_at': datetime.now().isoformat()
                }
            
            if 'error' in consensus_output:
                return {
                    'error': 'Consensus evaluation failed',
                    'consensus_error': consensus_output['error'],
                    'final_answer': 'I apologize, but I encountered an error while analyzing the document.',
                    'synthesized_at': datetime.now().isoformat()
                }
            
            # Extract information for synthesis
            document_type = rag_output.get('document_type', 'Unknown')
            
            # Prepare individual answers summary
            individual_answers_summary = self._prepare_individual_answers_summary(rag_output)
            
            # Prepare consensus evaluation summary
            consensus_summary = self._prepare_consensus_summary(consensus_output)
            
            # Prepare contradiction resolution summary
            resolution_summary = self._prepare_resolution_summary(resolution_output) if resolution_output else "No contradictions required resolution."
            
            # Generate final synthesized answer
            synthesized_answer = self._generate_synthesized_answer(
                user_question=user_question,
                document_type=document_type,
                individual_answers=individual_answers_summary,
                consensus_evaluation=consensus_summary,
                contradiction_resolution=resolution_summary
            )
            
            # Determine overall confidence and quality metrics
            quality_metrics = self._calculate_quality_metrics(rag_output, consensus_output, resolution_output)
            
            return {
                'user_question': user_question,
                'document_type': document_type,
                'final_answer': synthesized_answer,
                'quality_metrics': quality_metrics,
                'synthesis_success': True,
                'synthesized_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in final answer synthesis: {str(e)}")
            return {
                'error': str(e),
                'user_question': user_question,
                'final_answer': 'I apologize, but I encountered an error while preparing your answer. Please try rephrasing your question.',
                'synthesis_success': False,
                'synthesized_at': datetime.now().isoformat()
            }
    
    def _prepare_individual_answers_summary(self, rag_output: Dict[str, Any]) -> str:
        """
        Prepare summary of individual answers from Step 4A.
        
        Args:
            rag_output: RAG output from Step 4A
            
        Returns:
            Formatted summary of individual answers
        """
        try:
            individual_answers = rag_output.get('individual_answers', {})
            
            summary_parts = []
            for query_id, query_data in individual_answers.items():
                original_query = query_data.get('original_query', 'Unknown Query')
                angle_answers = query_data.get('angle_answers', {})
                
                summary_parts.append(f"**{query_id} ({original_query}):**")
                
                for angle_id, angle_data in angle_answers.items():
                    if angle_data.get('answer_success', False):
                        answer = angle_data.get('answer', 'No answer')
                        summary_parts.append(f"  {angle_id}: {answer}")
                
                summary_parts.append("")  # Empty line for separation
            
            return "\n".join(summary_parts) if summary_parts else "No individual answers available."
            
        except Exception as e:
            logger.error(f"Error preparing individual answers summary: {str(e)}")
            return "Error summarizing individual answers."
    
    def _prepare_consensus_summary(self, consensus_output: Dict[str, Any]) -> str:
        """
        Prepare summary of consensus evaluation from Step 4B.
        
        Args:
            consensus_output: Consensus output from Step 4B
            
        Returns:
            Formatted summary of consensus evaluation
        """
        try:
            consensus_evaluations = consensus_output.get('consensus_evaluations', {})
            
            summary_parts = []
            for query_id, evaluation in consensus_evaluations.items():
                if evaluation.get('evaluation_success', False):
                    final_verdict = evaluation.get('final_verdict', {})
                    confidence_assessment = evaluation.get('confidence_assessment', {})
                    
                    verdict = final_verdict.get('verdict', 'Unknown')
                    final_answer = final_verdict.get('final_answer', 'No answer provided')
                    confidence_level = confidence_assessment.get('confidence_level', 'unknown')
                    reasoning = final_verdict.get('reasoning', 'No reasoning provided')
                    
                    summary_parts.append(f"**{query_id}:**")
                    summary_parts.append(f"  Verdict: {verdict}")
                    summary_parts.append(f"  Answer: {final_answer}")
                    summary_parts.append(f"  Confidence: {confidence_level}")
                    summary_parts.append(f"  Reasoning: {reasoning}")
                    summary_parts.append("")
            
            return "\n".join(summary_parts) if summary_parts else "No consensus evaluation available."
            
        except Exception as e:
            logger.error(f"Error preparing consensus summary: {str(e)}")
            return "Error summarizing consensus evaluation."
    
    def _prepare_resolution_summary(self, resolution_output: Dict[str, Any]) -> str:
        """
        Prepare summary of contradiction resolution from Step 5.
        
        Args:
            resolution_output: Resolution output from Step 5
            
        Returns:
            Formatted summary of contradiction resolution
        """
        try:
            if not resolution_output or 'error' in resolution_output:
                return "No contradiction resolution performed."
            
            resolutions = resolution_output.get('resolutions', {})
            
            if not resolutions:
                return "No contradictions required resolution."
            
            summary_parts = []
            for query_id, resolution in resolutions.items():
                if resolution.get('resolution_success', False):
                    resolution_answer = resolution.get('resolution_answer', 'No resolution provided')
                    
                    summary_parts.append(f"**{query_id} Resolution:**")
                    summary_parts.append(f"  {resolution_answer}")
                    summary_parts.append("")
            
            return "\n".join(summary_parts) if summary_parts else "Contradiction resolution completed with no specific results."
            
        except Exception as e:
            logger.error(f"Error preparing resolution summary: {str(e)}")
            return "Error summarizing contradiction resolution."
    
    def _generate_synthesized_answer(self, user_question: str, document_type: str, 
                                    individual_answers: str, consensus_evaluation: str, 
                                    contradiction_resolution: str) -> str:
        """
        Generate final synthesized answer using LLM.
        
        Args:
            user_question: Original user question
            document_type: Type of document
            individual_answers: Summary of individual answers
            consensus_evaluation: Summary of consensus evaluation
            contradiction_resolution: Summary of contradiction resolution
            
        Returns:
            Synthesized final answer
        """
        try:
            # Create synthesis prompt
            formatted_prompt = self.synthesis_prompt.format(
                user_question=user_question,
                document_type=document_type,
                individual_answers=individual_answers,
                consensus_evaluation=consensus_evaluation,
                contradiction_resolution=contradiction_resolution
            )
            
            # Get synthesized answer from LLM
            response = self.llm.invoke(formatted_prompt)
            synthesized_answer = response.content.strip()
            
            return synthesized_answer
            
        except Exception as e:
            logger.error(f"Error generating synthesized answer: {str(e)}")
            return f"I apologize, but I encountered an error while formulating your answer. The document appears to contain information relevant to your question, but I'm unable to present it clearly at the moment."
    
    def _calculate_quality_metrics(self, rag_output: Dict[str, Any], consensus_output: Dict[str, Any], 
                                  resolution_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate quality metrics for the synthesized answer.
        
        Args:
            rag_output: RAG output from Step 4A
            consensus_output: Consensus output from Step 4B
            resolution_output: Resolution output from Step 5
            
        Returns:
            Quality metrics dictionary
        """
        try:
            # RAG metrics
            rag_success_rate = 0
            total_rag_queries = rag_output.get('total_search_angles_processed', 0)
            successful_rag_queries = rag_output.get('successful_answers', 0)
            if total_rag_queries > 0:
                rag_success_rate = (successful_rag_queries / total_rag_queries) * 100
            
            # Consensus metrics
            consensus_evaluations = consensus_output.get('consensus_evaluations', {})
            high_confidence_count = 0
            medium_confidence_count = 0
            low_confidence_count = 0
            contradictory_count = 0
            
            for evaluation in consensus_evaluations.values():
                if evaluation.get('evaluation_success', False):
                    confidence_level = evaluation.get('confidence_assessment', {}).get('confidence_level', 'low')
                    verdict = evaluation.get('final_verdict', {}).get('verdict', 'INSUFFICIENT')
                    
                    if confidence_level == 'high' or verdict == 'CORRECT':
                        high_confidence_count += 1
                    elif confidence_level == 'medium' or verdict == 'SATISFACTORY':
                        medium_confidence_count += 1
                    elif verdict == 'CONTRADICTORY':
                        contradictory_count += 1
                    else:
                        low_confidence_count += 1
            
            # Resolution metrics
            resolution_success = False
            if resolution_output and 'error' not in resolution_output:
                successfully_resolved = resolution_output.get('successfully_resolved', 0)
                contradictory_queries_found = resolution_output.get('contradictory_queries_found', 0)
                if contradictory_queries_found > 0:
                    resolution_success = successfully_resolved == contradictory_queries_found
            
            # Overall quality score
            total_queries = len(consensus_evaluations)
            if total_queries > 0:
                quality_score = ((high_confidence_count * 100) + (medium_confidence_count * 75) + 
                               (low_confidence_count * 25)) / (total_queries * 100) * 100
            else:
                quality_score = 0
            
            return {
                'rag_success_rate': round(rag_success_rate, 1),
                'total_queries_analyzed': total_queries,
                'high_confidence_queries': high_confidence_count,
                'medium_confidence_queries': medium_confidence_count,
                'low_confidence_queries': low_confidence_count,
                'contradictory_queries': contradictory_count,
                'contradictions_resolved': resolution_success,
                'overall_quality_score': round(quality_score, 1),
                'analysis_completeness': 'Complete' if rag_success_rate > 80 else 'Partial'
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {str(e)}")
            return {
                'overall_quality_score': 0,
                'analysis_completeness': 'Error'
            }

# Example usage
if __name__ == "__main__":
    # API key for final synthesizer
    GOOGLE_API_KEY = "your_google_api_key_7_here"  # 7th API key for this component
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_7_here":
        print("Please set your Google API key for the final synthesizer")
        exit(1)
    
    # Initialize synthesizer
    synthesizer = FinalAnswerSynthesizer(GOOGLE_API_KEY)
    
    print("=== FINAL ANSWER SYNTHESIZER DEMO ===")
    print("This component synthesizes all analysis results into clean, natural answers:")
    print("1. Takes individual answers from Step 4A")
    print("2. Takes consensus evaluation from Step 4B") 
    print("3. Takes contradiction resolution from Step 5 (if any)")
    print("4. Creates natural, conversational paragraph answer")
    print("5. Includes quality metrics and confidence indicators")
    print("\nExample synthesis:")
    print("Input: Complex analysis with 15 individual answers and consensus")
    print("Output: 'Based on your offer letter, the monthly stipend is INR 12,000...'")