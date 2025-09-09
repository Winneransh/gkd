"""
contradiction_resolver.py

Step 5: Resolves contradictory findings by generating targeted searches
based on contradiction reasons and providing final definitive answers.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContradictionResolver:
    """
    Resolves contradictory findings by generating targeted searches based on
    the specific contradictions identified and providing final definitive answers.
    """
    
    def __init__(self, google_api_key: str, chroma_persist_directory: str = "./chroma_db"):
        """
        Initialize the contradiction resolver.
        
        Args:
            google_api_key: Google API key for Gemini
            chroma_persist_directory: Directory where ChromaDB is persisted
        """
        self.google_api_key = google_api_key
        self.chroma_persist_directory = chroma_persist_directory
        
        # Initialize Gemini 2.5 Flash LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=4096
        )
        
        # Initialize embeddings for ChromaDB
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Create prompt templates
        self.contradiction_search_prompt = self._create_contradiction_search_prompt()
        self.resolution_prompt = self._create_resolution_prompt()
    
    def _create_contradiction_search_prompt(self) -> PromptTemplate:
        """
        Create prompt for generating targeted searches to resolve contradictions.
        
        Returns:
            PromptTemplate for contradiction-focused search generation
        """
        
        prompt_template = """You are an expert legal document researcher specializing in resolving contradictory information. Your job is to generate targeted search queries that will help resolve specific contradictions found in previous analysis.

**DOCUMENT TYPE:** {document_type}
**ORIGINAL QUERY:** {original_query}
**CONTRADICTORY FINDINGS:** {contradictory_findings}
**CONTRADICTION REASONS:** {contradiction_reasons}

**CONTRADICTION RESOLUTION STRATEGY:**

**1. IDENTIFY THE CORE CONFLICT:**
- What exactly are the contradictory statements about?
- Are they about amounts, dates, conditions, or terms?
- Which specific aspects need clarification?

**2. GENERATE TARGETED SEARCHES:**
- Create searches that specifically target the conflicting information
- Look for definitive statements, official clauses, or authoritative sections
- Focus on exact terms, amounts, dates, or conditions mentioned in contradictions
- Search for clarifying context around the conflicting information

**3. DOCUMENT-SPECIFIC SEARCH STRATEGIES:**

**OFFER LETTER:**
- For salary/compensation conflicts: Search for exact amounts, payment terms, bonus structure
- For duration conflicts: Search for specific start dates, contract periods, probation terms
- For work arrangement conflicts: Search for location, hours, remote work policies
- For legal term conflicts: Search for termination clauses, notice periods, obligations

**HEALTH INSURANCE:**
- For coverage conflicts: Search for specific procedure coverage, exclusions, limitations
- For cost conflicts: Search for exact premium amounts, deductibles, co-payment terms
- For eligibility conflicts: Search for age requirements, pre-condition clauses, family coverage
- For geographic conflicts: Search for coverage areas, network restrictions, location limits

**OTHER LEGAL DOCUMENTS:**
- Focus on the specific legal terms, amounts, dates, or conditions that are contradictory
- Search for definitive clauses, official statements, or authoritative sections

**4. SEARCH QUERY CHARACTERISTICS:**
- **Specific**: Target the exact contradictory elements
- **Authoritative**: Look for official terms, clauses, or definitive statements
- **Contextual**: Include surrounding context that might clarify the contradiction
- **Precise**: Use exact terminology from the contradictory findings

**OUTPUT FORMAT:**
Generate exactly 5 targeted search queries in JSON format:

{{
  "targeted_searches": [
    {{
      "search_query": "specific targeted search query 1",
      "focus": "what_this_search_targets",
      "rationale": "why this search will help resolve the contradiction"
    }},
    {{
      "search_query": "specific targeted search query 2", 
      "focus": "what_this_search_targets",
      "rationale": "why this search will help resolve the contradiction"
    }},
    {{
      "search_query": "specific targeted search query 3",
      "focus": "what_this_search_targets", 
      "rationale": "why this search will help resolve the contradiction"
    }},
    {{
      "search_query": "specific targeted search query 4",
      "focus": "what_this_search_targets",
      "rationale": "why this search will help resolve the contradiction"
    }},
    {{
      "search_query": "specific targeted search query 5",
      "focus": "what_this_search_targets",
      "rationale": "why this search will help resolve the contradiction"
    }}
  ]
}}

**GENERATE TARGETED SEARCHES:**"""

        return PromptTemplate(
            input_variables=["document_type", "original_query", "contradictory_findings", "contradiction_reasons"],
            template=prompt_template
        )
    
    def _create_resolution_prompt(self) -> PromptTemplate:
        """
        Create prompt for resolving contradictions using retrieved context.
        
        Returns:
            PromptTemplate for contradiction resolution
        """
        
        prompt_template = """You are an expert legal document analyst specializing in resolving contradictory information. Your job is to analyze the retrieved context and provide a definitive, authoritative answer that resolves the contradiction.

**DOCUMENT TYPE:** {document_type}
**ORIGINAL QUERY:** {original_query}
**CONTRADICTORY FINDINGS:** {contradictory_findings}
**CONTRADICTION REASONS:** {contradiction_reasons}

**RESOLUTION INSTRUCTIONS:**

**1. ANALYZE THE RETRIEVED CONTEXT:**
- Look for definitive, authoritative statements in the document
- Identify which contradictory finding is correct based on the evidence
- Find official clauses, exact amounts, precise dates, or clear terms
- Determine if both contradictory statements have merit in different contexts

**2. RESOLUTION STRATEGIES:**

**DEFINITIVE RESOLUTION:**
- One contradictory finding is clearly correct based on authoritative document text
- Provide the correct information with supporting evidence
- Explain why the other finding was incorrect

**CONTEXTUAL RESOLUTION:**  
- Both findings are correct but apply to different contexts/conditions
- Explain the different contexts where each applies
- Provide comprehensive answer covering all scenarios

**CLARIFICATION RESOLUTION:**
- The contradiction stems from ambiguous language in the document
- Provide the most reasonable interpretation based on document context
- Note any remaining ambiguity that cannot be definitively resolved

**INSUFFICIENT INFORMATION:**
- The retrieved context still doesn't provide enough information to resolve
- State what information is missing
- Provide best available answer with caveats

**3. RESPONSE REQUIREMENTS:**
- **Be Definitive**: Provide clear, authoritative answer when possible
- **Use Evidence**: Quote specific document text that resolves the contradiction
- **Explain Resolution**: Clearly state how the contradiction is resolved
- **Address Both Sides**: Acknowledge both contradictory findings and explain the resolution
- **Be Precise**: Use exact amounts, dates, terms from the document

**RETRIEVED CONTEXT:**
{retrieved_context}

**RESPONSE FORMAT:**
```
**CONTRADICTION RESOLUTION:**

**Final Answer:** [Definitive answer to the original query]

**Resolution Type:** [Definitive/Contextual/Clarification/Insufficient]

**Supporting Evidence:** [Specific quotes or references from the document that resolve the contradiction]

**Contradiction Explanation:** [Clear explanation of how the contradiction is resolved - which finding was correct/incorrect and why]

**Additional Context:** [Any relevant context or caveats that provide complete understanding]
```

**RESOLVE THE CONTRADICTION:**"""

        return PromptTemplate(
            input_variables=["document_type", "original_query", "contradictory_findings", "contradiction_reasons", "retrieved_context"],
            template=prompt_template
        )
    
    def resolve_contradictions(self, consensus_output: Dict[str, Any], collection_name: str = "legal_documents") -> Dict[str, Any]:
        """
        Resolve contradictory findings from consensus evaluation.
        
        Args:
            consensus_output: Complete output from ConsensusEvaluator.evaluate_consensus()
            collection_name: ChromaDB collection name to search in
            
        Returns:
            Dictionary containing contradiction resolution results
        """
        try:
            # Validate input
            if 'error' in consensus_output:
                return {
                    'error': 'Consensus evaluation failed',
                    'consensus_error': consensus_output['error'],
                    'resolutions': {},
                    'resolved_at': datetime.now().isoformat()
                }
            
            # Initialize ChromaDB vector store
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )
            
            document_type = consensus_output.get('document_type', 'Unknown')
            original_query = consensus_output.get('original_query', '')
            consensus_evaluations = consensus_output.get('consensus_evaluations', {})
            
            resolution_results = {
                'document_type': document_type,
                'original_query': original_query,
                'collection_name': collection_name,
                'resolutions': {},
                'contradictory_queries_found': 0,
                'successfully_resolved': 0,
                'resolved_at': datetime.now().isoformat()
            }
            
            # Find queries that need contradiction resolution
            contradictory_queries = []
            for query_id, evaluation in consensus_evaluations.items():
                if evaluation.get('evaluation_success', False):
                    verdict = evaluation.get('final_verdict', {}).get('verdict', '')
                    confidence_level = evaluation.get('confidence_assessment', {}).get('confidence_level', '')
                    
                    # Check if this query needs contradiction resolution
                    if verdict == 'CONTRADICTORY' or confidence_level == 'contradictory':
                        contradictory_queries.append({
                            'query_id': query_id,
                            'evaluation': evaluation
                        })
            
            resolution_results['contradictory_queries_found'] = len(contradictory_queries)
            
            if not contradictory_queries:
                logger.info("No contradictory queries found - no resolution needed")
                return resolution_results
            
            # Resolve each contradictory query
            for query_info in contradictory_queries:
                query_id = query_info['query_id']
                evaluation = query_info['evaluation']
                
                logger.info(f"Resolving contradiction for query {query_id}")
                
                try:
                    # Resolve this contradictory query
                    resolution = self._resolve_single_contradiction(
                        query_id=query_id,
                        evaluation=evaluation,
                        vectorstore=vectorstore,
                        document_type=document_type,
                        original_user_query=original_query
                    )
                    
                    resolution_results['resolutions'][query_id] = resolution
                    
                    if resolution.get('resolution_success', False):
                        resolution_results['successfully_resolved'] += 1
                    
                except Exception as e:
                    logger.error(f"Error resolving contradiction for query {query_id}: {str(e)}")
                    resolution_results['resolutions'][query_id] = {
                        'query_id': query_id,
                        'error': str(e),
                        'resolution_success': False,
                        'final_answer': 'Resolution failed due to error'
                    }
            
            logger.info(f"Contradiction resolution completed. Resolved {resolution_results['successfully_resolved']}/{resolution_results['contradictory_queries_found']} queries")
            
            return resolution_results
            
        except Exception as e:
            logger.error(f"Error in contradiction resolution: {str(e)}")
            return {
                'error': str(e),
                'document_type': consensus_output.get('document_type', 'Unknown'),
                'resolutions': {},
                'resolved_at': datetime.now().isoformat()
            }
    
    def _resolve_single_contradiction(self, query_id: str, evaluation: Dict[str, Any], 
                                     vectorstore: Chroma, document_type: str, original_user_query: str) -> Dict[str, Any]:
        """
        Resolve contradiction for a single query.
        
        Args:
            query_id: Query identifier
            evaluation: Consensus evaluation result
            vectorstore: ChromaDB vector store instance
            document_type: Type of document
            original_user_query: Original user query
            
        Returns:
            Resolution result for this query
        """
        try:
            # Extract contradiction information
            original_query = evaluation.get('original_query', '')
            contradictory_findings = self._extract_contradictory_findings(evaluation)
            contradiction_reasons = self._extract_contradiction_reasons(evaluation)
            
            # Step 1: Generate targeted searches for contradiction resolution
            targeted_searches = self._generate_targeted_searches(
                document_type=document_type,
                original_query=original_query,
                contradictory_findings=contradictory_findings,
                contradiction_reasons=contradiction_reasons
            )
            
            # Step 2: Execute targeted searches
            retrieved_context = self._execute_targeted_searches(
                targeted_searches=targeted_searches,
                vectorstore=vectorstore
            )
            
            # Step 3: Resolve contradiction using LLM
            if retrieved_context:
                resolution_answer = self._get_resolution_answer(
                    document_type=document_type,
                    original_query=original_query,
                    contradictory_findings=contradictory_findings,
                    contradiction_reasons=contradiction_reasons,
                    retrieved_context=retrieved_context
                )
                
                return {
                    'query_id': query_id,
                    'original_query': original_query,
                    'contradictory_findings': contradictory_findings,
                    'contradiction_reasons': contradiction_reasons,
                    'targeted_searches_used': len(targeted_searches),
                    'resolution_answer': resolution_answer,
                    'resolution_success': True
                }
            else:
                return {
                    'query_id': query_id,
                    'original_query': original_query,
                    'error': 'No relevant context retrieved for contradiction resolution',
                    'resolution_success': False
                }
                
        except Exception as e:
            logger.error(f"Error resolving single contradiction: {str(e)}")
            return {
                'query_id': query_id,
                'error': str(e),
                'resolution_success': False
            }
    
    def _extract_contradictory_findings(self, evaluation: Dict[str, Any]) -> List[str]:
        """
        Extract contradictory findings from consensus evaluation.
        
        Args:
            evaluation: Consensus evaluation result
            
        Returns:
            List of contradictory findings
        """
        try:
            answer_consistency = evaluation.get('answer_consistency', {})
            conflicting_answers = answer_consistency.get('conflicting_answers', [])
            factual_contradictions = answer_consistency.get('factual_contradictions', [])
            
            contradictory_findings = []
            contradictory_findings.extend(conflicting_answers)
            contradictory_findings.extend(factual_contradictions)
            
            return contradictory_findings if contradictory_findings else ['Multiple conflicting answers found']
            
        except Exception:
            return ['Contradictory information detected']
    
    def _extract_contradiction_reasons(self, evaluation: Dict[str, Any]) -> str:
        """
        Extract reasons for contradiction from consensus evaluation.
        
        Args:
            evaluation: Consensus evaluation result
            
        Returns:
            Contradiction reasons string
        """
        try:
            final_verdict = evaluation.get('final_verdict', {})
            reasoning = final_verdict.get('reasoning', '')
            
            detailed_analysis = evaluation.get('detailed_analysis', {})
            consensus_breakdown = detailed_analysis.get('consensus_breakdown', '')
            
            reasons = []
            if reasoning:
                reasons.append(reasoning)
            if consensus_breakdown:
                reasons.append(consensus_breakdown)
            
            return ' | '.join(reasons) if reasons else 'Contradictory answers found in consensus analysis'
            
        except Exception:
            return 'Contradiction detected during consensus evaluation'
    
    def _generate_targeted_searches(self, document_type: str, original_query: str, 
                                   contradictory_findings: List[str], contradiction_reasons: str) -> List[str]:
        """
        Generate targeted search queries to resolve contradictions.
        
        Args:
            document_type: Type of document
            original_query: Original query
            contradictory_findings: List of contradictory findings
            contradiction_reasons: Reasons for contradiction
            
        Returns:
            List of targeted search queries
        """
        try:
            # Create search generation prompt
            formatted_prompt = self.contradiction_search_prompt.format(
                document_type=document_type,
                original_query=original_query,
                contradictory_findings=json.dumps(contradictory_findings),
                contradiction_reasons=contradiction_reasons
            )
            
            # Get targeted searches from LLM
            response = self.llm.invoke(formatted_prompt)
            search_text = response.content
            
            # Parse targeted searches
            targeted_searches = self._parse_targeted_searches(search_text)
            
            return targeted_searches
            
        except Exception as e:
            logger.error(f"Error generating targeted searches: {str(e)}")
            # Fallback searches based on original query
            return [
                f"{original_query} exact amount",
                f"{original_query} official terms",
                f"{original_query} specific details",
                f"{original_query} authoritative clause",
                f"{original_query} definitive information"
            ]
    
    def _parse_targeted_searches(self, search_text: str) -> List[str]:
        """
        Parse targeted searches from LLM response.
        
        Args:
            search_text: Raw LLM response
            
        Returns:
            List of targeted search queries
        """
        try:
            # Try to extract JSON
            json_start = search_text.find('{')
            json_end = search_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_text = search_text[json_start:json_end]
                parsed_json = json.loads(json_text)
                
                targeted_searches = []
                for item in parsed_json.get('targeted_searches', []):
                    query = item.get('search_query', '')
                    if query:
                        targeted_searches.append(query)
                
                return targeted_searches if targeted_searches else []
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error parsing targeted searches: {str(e)}")
            return []
    
    def _execute_targeted_searches(self, targeted_searches: List[str], vectorstore: Chroma) -> str:
        """
        Execute targeted searches and compile context.
        
        Args:
            targeted_searches: List of targeted search queries
            vectorstore: ChromaDB vector store instance
            
        Returns:
            Compiled context from all searches
        """
        try:
            all_retrieved_docs = []
            
            for search_query in targeted_searches:
                docs = vectorstore.similarity_search(search_query, k=3)
                all_retrieved_docs.extend(docs)
            
            # Deduplicate documents
            unique_docs = self._deduplicate_docs(all_retrieved_docs)
            
            # Prepare context
            if unique_docs:
                context_parts = []
                for i, doc in enumerate(unique_docs[:8]):  # Use top 8 most relevant
                    context_parts.append(f"**Context {i+1}:**")
                    context_parts.append(doc.page_content)
                    context_parts.append("")
                
                return "\n".join(context_parts)
            else:
                return "No relevant context found"
                
        except Exception as e:
            logger.error(f"Error executing targeted searches: {str(e)}")
            return "Error retrieving context"
    
    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents.
        
        Args:
            docs: List of documents
            
        Returns:
            Deduplicated documents
        """
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _get_resolution_answer(self, document_type: str, original_query: str, 
                              contradictory_findings: List[str], contradiction_reasons: str, 
                              retrieved_context: str) -> str:
        """
        Get resolution answer from LLM using retrieved context.
        
        Args:
            document_type: Type of document
            original_query: Original query
            contradictory_findings: List of contradictory findings
            contradiction_reasons: Reasons for contradiction
            retrieved_context: Retrieved context from targeted searches
            
        Returns:
            Resolution answer string
        """
        try:
            # Create resolution prompt
            formatted_prompt = self.resolution_prompt.format(
                document_type=document_type,
                original_query=original_query,
                contradictory_findings=json.dumps(contradictory_findings),
                contradiction_reasons=contradiction_reasons,
                retrieved_context=retrieved_context
            )
            
            # Get resolution from LLM
            response = self.llm.invoke(formatted_prompt)
            resolution_answer = response.content.strip()
            
            return resolution_answer
            
        except Exception as e:
            logger.error(f"Error getting resolution answer: {str(e)}")
            return f"Error generating resolution: {str(e)}"

# Example usage
if __name__ == "__main__":
    # API key for contradiction resolver
    GOOGLE_API_KEY = "your_google_api_key_6_here"  # 6th API key for this component
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_6_here":
        print("Please set your Google API key for the contradiction resolver")
        exit(1)
    
    # Import all previous components
    import os
    from document_classifier import LegalDocumentClassifier
    from query_analyzer import LegalDocumentQueryAnalyzer
    from search_query_generator import SearchQueryGenerator
    from llm_enhanced_rag import LLMEnhancedRAG
    from consensus_evaluator import ConsensusEvaluator
    
    # For demo purposes - would need all 6 API keys
    print("=== CONTRADICTION RESOLVER DEMO ===")
    print("This component resolves contradictory findings by:")
    print("1. Identifying queries marked as 'CONTRADICTORY' from Step 4B")
    print("2. Generating 5 targeted searches based on contradiction reasons")
    print("3. Executing enhanced LLM RAG to find definitive answers")
    print("4. Providing final resolution with supporting evidence")
    print("\nExample workflow:")
    print("- Step 4B finds: 'Conflicting stipend amounts: 12000 vs 15000'")
    print("- Step 5 generates: 'exact monthly stipend amount INR official'")
    print("- Step 5 searches and finds: 'Monthly stipend is INR 12,000'")
    print("- Step 5 resolves: 'DEFINITIVE: Monthly stipend is INR 12,000'")