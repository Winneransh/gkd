"""
consensus_evaluator.py

Step 4B: Evaluates consensus across search angle answers, determines confidence scores,
and provides intelligent final verdicts considering search angle quality.
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsensusEvaluator:
    """
    Evaluates consensus across multiple search angle answers and provides intelligent
    final verdicts considering search angle quality and answer consistency.
    """
    
    def __init__(self, google_api_keys: List[str]):
        """
        Initialize the consensus evaluator with multiple API keys for rate limit handling.
        
        Args:
            google_api_keys: List of Google API keys to rotate through
        """
        self.google_api_keys = google_api_keys
        self.current_key_index = 0
        
        # Initialize Gemini 2.5 Flash LLM with first key
        self.llm = self._get_llm_instance()
        
        # Create consensus evaluation prompt
        self.consensus_prompt = self._create_consensus_evaluation_prompt()
    
    def _get_llm_instance(self) -> ChatGoogleGenerativeAI:
        """
        Get LLM instance with current API key.
        
        Returns:
            ChatGoogleGenerativeAI instance
        """
        current_key = self.google_api_keys[self.current_key_index]
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=current_key,
            temperature=0.1,
            max_output_tokens=6144
        )
    
    def _rotate_api_key(self):
        """
        Rotate to the next API key to handle rate limits.
        """
        self.current_key_index = (self.current_key_index + 1) % len(self.google_api_keys)
        self.llm = self._get_llm_instance()
        logger.info(f"Rotated to API key index: {self.current_key_index}")
    
    def _create_consensus_evaluation_prompt(self) -> PromptTemplate:
        """
        Create comprehensive system prompt for consensus evaluation.
        
        Returns:
            PromptTemplate for consensus evaluation
        """
        
        prompt_template = """You are an expert legal document consensus analyst. Your job is to evaluate multiple search angle answers for the same query and provide an intelligent final verdict considering both answer consistency and search angle quality.

**EVALUATION CONTEXT:**
Document Type: {document_type}
Original User Query: {original_query}
Generated Query Component: {generated_query}

**EVALUATION INPUTS:**

**Search Angles and Their Answers:**
{search_angles_and_answers}

**COMPREHENSIVE EVALUATION FRAMEWORK:**

**1. SEARCH ANGLE QUALITY ASSESSMENT:**
- **Relevance Check**: Do the search angles properly target the original query?
- **Coverage Analysis**: Do the angles cover different aspects appropriately?
- **Quality Scoring**: Are the search angles well-formulated or confusing/irrelevant?
- **Alignment Assessment**: How well do the search angles match the generated query component?

**2. ANSWER CONSISTENCY ANALYSIS:**
- **Direct Consensus**: How many angles provide the same/similar answers?
- **Factual Alignment**: Do the answers contain consistent factual information?
- **Contradiction Detection**: Are there conflicting answers about the same facts?
- **Information Completeness**: Do answers provide sufficient detail to address the query?

**3. INTELLIGENT CONFIDENCE SCORING:**
- **5/5 Consensus**: All 5 angles provide consistent, accurate answers
- **4/5 Consensus**: 4 angles agree, 1 differs (still high confidence if majority is strong)
- **3/5 Consensus**: 3 angles agree (moderate confidence, investigate differences)
- **2/5 Consensus**: Only 2 angles agree (low confidence, high uncertainty)
- **1/5 or 0/5**: No clear consensus (very low confidence)

**4. SEARCH ANGLE QUALITY BYPASS LOGIC:**
- **Scenario A**: Low numerical consensus (2/5 or 3/5) BUT search angles are poorly formulated/irrelevant 
  → If the few good answers are consistent and address the query well → Upgrade verdict to "SATISFACTORY"
- **Scenario B**: High numerical consensus (4/5 or 5/5) BUT answers are contradictory on core facts
  → If search angles are well-formulated → Flag as "CONTRADICTORY" requiring investigation
- **Scenario C**: Mixed results but some angles provide clear, definitive answers
  → Prioritize quality answers over quantity

**5. VERDICT DETERMINATION LOGIC:**

**HIGH CONFIDENCE (80-100%):**
- 4+ angles provide consistent, factual answers
- Search angles are well-aligned with the query
- No significant contradictions on core facts
- Sufficient information to fully address the query

**MEDIUM CONFIDENCE (60-79%):**
- 3+ angles provide consistent answers with minor variations
- Search angles are reasonably aligned
- Some ambiguity but no major contradictions
- Adequate information to address the query

**SATISFACTORY (Bypass Low Confidence):**
- Numerical consensus may be low (2-3/5) BUT
- Search angles are poorly formulated/irrelevant causing confusion
- The few relevant answers are consistent and accurate
- Clear information available despite search angle issues

**LOW CONFIDENCE (30-59%):**
- Only 2 angles provide similar answers
- Search angles are reasonably well-formulated
- Significant gaps in information or minor contradictions
- Insufficient information to fully address the query

**CONTRADICTORY (Requires Investigation):**
- Answers contain conflicting factual information
- Search angles are well-formulated and relevant
- Multiple sources provide different facts about the same thing
- Investigation needed to resolve conflicts

**INSUFFICIENT (Below 30%):**
- 0-1 angles provide relevant answers
- Search angles may be poorly formulated
- No clear information available to address the query

**OUTPUT FORMAT:**
Provide your evaluation in the following JSON structure:

{{
  "search_angle_quality": {{
    "relevance_score": 85,
    "coverage_score": 90,
    "alignment_score": 88,
    "overall_quality": "high|medium|low",
    "quality_issues": ["list any issues with search angles"]
  }},
  "answer_consistency": {{
    "consensus_count": "4/5",
    "consensus_percentage": 80,
    "consistent_answers": ["answer1", "answer2", "answer3", "answer4"],
    "conflicting_answers": ["conflicting_answer"],
    "factual_contradictions": ["any contradictory facts"]
  }},
  "confidence_assessment": {{
    "numerical_confidence": 80,
    "confidence_level": "high|medium|satisfactory|low|contradictory|insufficient",
    "bypass_applied": false,
    "bypass_reason": "explanation if bypass was applied"
  }},
  "final_verdict": {{
    "verdict": "CORRECT|SATISFACTORY|CONTRADICTORY|INSUFFICIENT",
    "final_answer": "the consolidated final answer to the original query",
    "supporting_evidence": ["evidence points supporting the answer"],
    "reasoning": "detailed explanation of the verdict decision",
    "recommendations": ["any recommendations for improvement"]
  }},
  "detailed_analysis": {{
    "angle_by_angle_review": {{
      "A1": {{"relevance": "high|medium|low", "answer_quality": "assessment"}},
      "A2": {{"relevance": "high|medium|low", "answer_quality": "assessment"}},
      "A3": {{"relevance": "high|medium|low", "answer_quality": "assessment"}},
      "A4": {{"relevance": "high|medium|low", "answer_quality": "assessment"}},
      "A5": {{"relevance": "high|medium|low", "answer_quality": "assessment"}}
    }},
    "consensus_breakdown": "detailed explanation of how consensus was determined",
    "quality_vs_consensus": "analysis of whether search angle quality affected consensus"
  }}
}}

**EVALUATION INSTRUCTIONS:**
1. First assess the quality and relevance of each search angle
2. Analyze the consistency and quality of answers across all angles
3. Calculate numerical consensus and identify patterns
4. Apply intelligent bypass logic if search angle quality is poor but answers are good
5. Determine final verdict considering both consensus and search angle quality
6. Provide clear reasoning for the verdict decision

**PERFORM YOUR EVALUATION:**"""

        return PromptTemplate(
            input_variables=["document_type", "original_query", "generated_query", "search_angles_and_answers"],
            template=prompt_template
        )
    
    def evaluate_consensus(self, rag_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate consensus across all query answers from Step 4A.
        
        Args:
            rag_output: Complete output from LLMEnhancedRAG.process_all_search_angles()
            
        Returns:
            Dictionary containing consensus evaluation results for all queries
        """
        try:
            # Validate input
            if 'error' in rag_output:
                return {
                    'error': 'RAG processing failed',
                    'rag_error': rag_output['error'],
                    'consensus_evaluations': {},
                    'evaluated_at': datetime.now().isoformat()
                }
            
            document_type = rag_output.get('document_type', 'Unknown')
            original_query = rag_output.get('original_query', '')
            individual_answers = rag_output.get('individual_answers', {})
            
            evaluation_results = {
                'document_type': document_type,
                'original_query': original_query,
                'consensus_evaluations': {},
                'overall_summary': {},
                'queries_needing_refinement': [],
                'evaluated_at': datetime.now().isoformat()
            }
            
            # Evaluate consensus for each query
            for query_id, query_data in individual_answers.items():
                logger.info(f"Evaluating consensus for query {query_id}")
                
                try:
                    # Perform consensus evaluation for this query
                    consensus_result = self._evaluate_single_query_consensus(
                        query_id=query_id,
                        query_data=query_data,
                        document_type=document_type,
                        original_user_query=original_query
                    )
                    
                    evaluation_results['consensus_evaluations'][query_id] = consensus_result
                    
                    # Check if this query needs refinement
                    verdict = consensus_result.get('final_verdict', {}).get('verdict', 'INSUFFICIENT')
                    confidence_level = consensus_result.get('confidence_assessment', {}).get('confidence_level', 'low')
                    
                    if verdict in ['CONTRADICTORY', 'INSUFFICIENT'] or confidence_level == 'low':
                        evaluation_results['queries_needing_refinement'].append({
                            'query_id': query_id,
                            'verdict': verdict,
                            'confidence_level': confidence_level,
                            'issues': consensus_result.get('final_verdict', {}).get('recommendations', [])
                        })
                    
                    # Rotate API key after each query to manage rate limits
                    self._rotate_api_key()
                    
                except Exception as e:
                    logger.error(f"Error evaluating consensus for query {query_id}: {str(e)}")
                    evaluation_results['consensus_evaluations'][query_id] = {
                        'error': str(e),
                        'final_verdict': {
                            'verdict': 'INSUFFICIENT',
                            'final_answer': 'Evaluation failed',
                            'reasoning': f'Error during evaluation: {str(e)}'
                        }
                    }
                    # Still rotate API key on error
                    self._rotate_api_key()
            
            # Generate overall summary
            evaluation_results['overall_summary'] = self._generate_overall_summary(evaluation_results['consensus_evaluations'])
            
            logger.info(f"Completed consensus evaluation for {len(individual_answers)} queries")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in consensus evaluation: {str(e)}")
            return {
                'error': str(e),
                'document_type': rag_output.get('document_type', 'Unknown'),
                'consensus_evaluations': {},
                'evaluated_at': datetime.now().isoformat()
            }
    
    def _evaluate_single_query_consensus(self, query_id: str, query_data: Dict[str, Any], 
                                        document_type: str, original_user_query: str) -> Dict[str, Any]:
        """
        Evaluate consensus for a single query's 5 search angle answers.
        
        Args:
            query_id: Query identifier
            query_data: Query data with angle answers
            document_type: Type of document
            original_user_query: Original user query
            
        Returns:
            Consensus evaluation result for this query
        """
        try:
            original_query = query_data.get('original_query', '')
            angle_answers = query_data.get('angle_answers', {})
            
            # Prepare search angles and answers for evaluation
            search_angles_and_answers = self._format_angles_and_answers(angle_answers)
            
            # Create evaluation prompt
            formatted_prompt = self.consensus_prompt.format(
                document_type=document_type,
                original_query=original_user_query,
                generated_query=original_query,
                search_angles_and_answers=search_angles_and_answers
            )
            
            # Get consensus evaluation from Gemini
            response = self.llm.invoke(formatted_prompt)
            evaluation_text = response.content
            
            # Parse the evaluation response
            parsed_evaluation = self._parse_consensus_evaluation(evaluation_text)
            
            # Add metadata
            parsed_evaluation.update({
                'query_id': query_id,
                'original_query': original_query,
                'total_angles_evaluated': len(angle_answers),
                'evaluation_success': True
            })
            
            return parsed_evaluation
            
        except Exception as e:
            logger.error(f"Error in single query consensus evaluation: {str(e)}")
            return {
                'query_id': query_id,
                'error': str(e),
                'final_verdict': {
                    'verdict': 'INSUFFICIENT',
                    'final_answer': 'Consensus evaluation failed',
                    'reasoning': f'Error during consensus evaluation: {str(e)}'
                },
                'evaluation_success': False
            }
    
    def _format_angles_and_answers(self, angle_answers: Dict[str, Any]) -> str:
        """
        Format search angles and their answers for LLM evaluation.
        
        Args:
            angle_answers: Dictionary of angle answers
            
        Returns:
            Formatted string for evaluation
        """
        formatted_parts = []
        
        for angle_id, angle_data in angle_answers.items():
            formatted_parts.append(f"**{angle_id}:**")
            formatted_parts.append(f"Search Query: {angle_data.get('search_query', 'N/A')}")
            formatted_parts.append(f"Angle Focus: {angle_data.get('angle_focus', 'N/A')}")
            formatted_parts.append(f"Keywords: {', '.join(angle_data.get('angle_keywords', []))}")
            formatted_parts.append(f"Answer: {angle_data.get('answer', 'No answer provided')}")
            formatted_parts.append(f"Success: {angle_data.get('answer_success', False)}")
            formatted_parts.append("")  # Empty line for separation
        
        return "\n".join(formatted_parts)
    
    def _parse_consensus_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """
        Parse the consensus evaluation response from Gemini.
        
        Args:
            evaluation_text: Raw evaluation response
            
        Returns:
            Parsed evaluation dictionary
        """
        try:
            # Try to extract JSON from the response
            json_start = evaluation_text.find('{')
            json_end = evaluation_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_text = evaluation_text[json_start:json_end]
                parsed_json = json.loads(json_text)
                
                # Validate required fields and provide defaults
                required_fields = {
                    'search_angle_quality': {},
                    'answer_consistency': {},
                    'confidence_assessment': {},
                    'final_verdict': {},
                    'detailed_analysis': {}
                }
                
                for field in required_fields:
                    if field not in parsed_json:
                        parsed_json[field] = required_fields[field]
                
                return parsed_json
            else:
                return self._fallback_consensus_parse(evaluation_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in consensus evaluation: {str(e)}")
            return self._fallback_consensus_parse(evaluation_text)
        except Exception as e:
            logger.error(f"Error parsing consensus evaluation: {str(e)}")
            return self._fallback_consensus_parse(evaluation_text)
    
    def _fallback_consensus_parse(self, evaluation_text: str) -> Dict[str, Any]:
        """
        Fallback parsing when JSON extraction fails.
        
        Args:
            evaluation_text: Raw evaluation text
            
        Returns:
            Basic consensus structure
        """
        return {
            'search_angle_quality': {
                'overall_quality': 'unknown',
                'quality_issues': ['JSON parsing failed']
            },
            'answer_consistency': {
                'consensus_count': '0/5',
                'consensus_percentage': 0
            },
            'confidence_assessment': {
                'numerical_confidence': 0,
                'confidence_level': 'insufficient',
                'bypass_applied': False
            },
            'final_verdict': {
                'verdict': 'INSUFFICIENT',
                'final_answer': 'Evaluation parsing failed',
                'reasoning': 'Could not parse consensus evaluation response'
            },
            'detailed_analysis': {
                'consensus_breakdown': 'Parsing failed'
            },
            'parse_error': 'JSON extraction failed',
            'raw_response': evaluation_text
        }
    
    def _generate_overall_summary(self, consensus_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall summary across all consensus evaluations.
        
        Args:
            consensus_evaluations: All consensus evaluation results
            
        Returns:
            Overall summary dictionary
        """
        if not consensus_evaluations:
            return {
                'total_queries': 0,
                'verdicts_summary': {},
                'confidence_distribution': {},
                'success_rate': 0
            }
        
        verdict_counts = {}
        confidence_levels = []
        successful_evaluations = 0
        
        for evaluation in consensus_evaluations.values():
            if evaluation.get('evaluation_success', False):
                successful_evaluations += 1
                
                verdict = evaluation.get('final_verdict', {}).get('verdict', 'INSUFFICIENT')
                confidence_level = evaluation.get('confidence_assessment', {}).get('confidence_level', 'low')
                
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                confidence_levels.append(confidence_level)
        
        # Calculate confidence distribution
        confidence_distribution = {}
        for level in confidence_levels:
            confidence_distribution[level] = confidence_distribution.get(level, 0) + 1
        
        return {
            'total_queries': len(consensus_evaluations),
            'successful_evaluations': successful_evaluations,
            'success_rate': (successful_evaluations / len(consensus_evaluations) * 100) if consensus_evaluations else 0,
            'verdicts_summary': verdict_counts,
            'confidence_distribution': confidence_distribution,
            'ready_for_synthesis': verdict_counts.get('CORRECT', 0) + verdict_counts.get('SATISFACTORY', 0) > 0
        }

# Example usage
if __name__ == "__main__":
    # Separate API keys for each component to avoid rate limits
    GOOGLE_API_KEYS = [
        # Key 5: consensus_evaluator.py
    ]
    
    # Filter out placeholder keys
    valid_keys = [key for key in GOOGLE_API_KEYS if key != "your_google_api_key_1_here" and not key.endswith("_here")]
    
    if len(valid_keys) < 5:
        print("Please set all 5 Google API keys in the GOOGLE_API_KEYS list")
        print("Each component needs its own API key to avoid rate limits:")
        print("- Key 1: document_classifier.py")
        print("- Key 2: query_analyzer.py") 
        print("- Key 3: search_query_generator.py")
        print("- Key 4: llm_enhanced_rag.py")
        print("- Key 5: consensus_evaluator.py")
        exit(1)
    
    # Import previous components
    import os
    from document_classifier import LegalDocumentClassifier
    from query_analyzer import LegalDocumentQueryAnalyzer
    from search_query_generator import SearchQueryGenerator
    from llm_enhanced_rag import LLMEnhancedRAG
    
    # Initialize each component with its own dedicated API key
    classifier = LegalDocumentClassifier(valid_keys[0])    # API Key 1
    analyzer = LegalDocumentQueryAnalyzer(valid_keys[1])   # API Key 2
    generator = SearchQueryGenerator(valid_keys[2])        # API Key 3
    rag_system = LLMEnhancedRAG(valid_keys[3])            # API Key 4
    evaluator = ConsensusEvaluator([valid_keys[4]])        # API Key 5
    
    # Complete workflow with consensus evaluation
    pdf_path = "Suryansh_OL.docx (1) (1) (1).pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        print("=== COMPLETE WORKFLOW WITH CONSENSUS EVALUATION (Steps 1-4B) ===")
        
        # Steps 1-4A: Same as before
        print("\n1. CLASSIFYING DOCUMENT...")
        classification_result = classifier.process_pdf_and_classify(pdf_path)
        
        if classification_result['success']:
            print(f"   Document Type: {classification_result['classification']['document_type']}")
            
            print("\n2. ANALYZING USER QUERY...")
            user_query = "What is the monthly stipend amount and working hours?"
            query_analysis = analyzer.analyze_query(user_query, classification_result)
            
            if 'error' not in query_analysis:
                print(f"   Single Queries: {len(query_analysis['single_queries'])}")
                print(f"   Hybrid Queries: {len(query_analysis['hybrid_queries'])}")
                
                print("\n3. GENERATING SEARCH ANGLES...")
                search_angles = generator.generate_search_angles(query_analysis)
                
                if 'error' not in search_angles:
                    print(f"   Total Search Angles: {search_angles['total_angles_generated']}")
                    
                    print("\n4A. PROCESSING INDIVIDUAL SEARCH ANGLES...")
                    rag_results = rag_system.process_all_search_angles(search_angles)
                    
                    if 'error' not in rag_results:
                        print(f"   Individual Answers Generated: {rag_results['successful_answers']}")
                        
                        # Step 4B: Consensus Evaluation
                        print("\n4B. EVALUATING CONSENSUS ACROSS ANSWERS...")
                        consensus_results = evaluator.evaluate_consensus(rag_results)
                        
                        if 'error' not in consensus_results:
                            summary = consensus_results['overall_summary']
                            print(f"   Queries Evaluated: {summary['total_queries']}")
                            print(f"   Success Rate: {summary['success_rate']:.1f}%")
                            print(f"   Ready for Synthesis: {summary['ready_for_synthesis']}")
                            
                            # Show consensus results
                            print("\n   Consensus Results:")
                            for query_id, consensus in consensus_results['consensus_evaluations'].items():
                                if consensus.get('evaluation_success', False):
                                    verdict = consensus['final_verdict']
                                    confidence = consensus['confidence_assessment']
                                    print(f"   Query {query_id}:")
                                    print(f"     Verdict: {verdict.get('verdict', 'Unknown')}")
                                    print(f"     Confidence: {confidence.get('confidence_level', 'Unknown')}")
                                    print(f"     Answer: {verdict.get('final_answer', 'N/A')[:100]}...")
                                    if confidence.get('bypass_applied', False):
                                        print(f"     Bypass Applied: {confidence.get('bypass_reason', 'N/A')}")
                            
                            # Show refinement needs
                            refinement_needed = consensus_results.get('queries_needing_refinement', [])
                            if refinement_needed:
                                print(f"\n   Queries Needing Refinement: {len(refinement_needed)}")
                                for item in refinement_needed:
                                    print(f"     {item['query_id']}: {item['verdict']} ({item['confidence_level']})")
                        else:
                            print(f"   Error: {consensus_results['error']}")
                    else:
                        print(f"   Error: {rag_results['error']}")
                else:
                    print(f"   Error: {search_angles['error']}")
            else:
                print(f"   Error: {query_analysis['error']}")
        else:
            print(f"   Error: {classification_result.get('error', 'Unknown error')}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("\n=== DEMO NOTES ===")
        print("This consensus evaluation system:")
        print("- Analyzes all 15 individual answers (5 per query)")
        print("- Evaluates search angle quality and answer consistency")
        print("- Applies intelligent bypass logic for poor search angles")
        print("- Provides verdicts: CORRECT/SATISFACTORY/CONTRADICTORY/INSUFFICIENT")
        print("- Uses multiple API keys to handle rate limits")
        print("- Ready for final synthesis step")