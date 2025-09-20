"""
enhanced_contradiction_resolver.py

Enhanced Step 5: Resolves contradictory findings with web search grounding for legal queries
and includes specialized duration calculation tools for contract/internship terms.
Updated to use HuggingFace embeddings and integrate with the complete pipeline.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from google import genai
from google.genai import types

# Chroma client factory
from chroma_client_factory import ChromaClientFactory
from chroma_config import ChromaConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DurationCalculator:
    """
    Specialized tool for calculating duration from contract/internship terms
    using LLM-enhanced extraction without regex.
    """
    
    def __init__(self, google_api_key: str):
        """Initialize duration calculator with Gemini."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        self.duration_extraction_prompt = self._create_duration_extraction_prompt()
    
    def _create_duration_extraction_prompt(self) -> PromptTemplate:
        """Create prompt for extracting duration information."""
        
        prompt_template = """You are a legal document duration extraction specialist. Your job is to extract and calculate duration information from document text.

**DOCUMENT CONTEXT:**
{retrieved_context}

**USER QUERY:**
{user_query}

**DURATION EXTRACTION TASK:**
Extract all duration-related information from the document that relates to the user's query. Look for:

1. **START DATES:**
   - Explicit dates in any format (DD/MM/YYYY, MM-DD-YYYY, Month DD, YYYY, etc.)
   - Relative dates ("from the date of joining", "effective immediately", etc.)
   - Academic terms ("semester starting", "academic year beginning", etc.)

2. **END DATES:**
   - Explicit end dates in any format
   - Duration expressions ("for 3 months", "6-month period", "until completion")
   - Conditional endings ("upon project completion", "subject to performance")

3. **DURATION EXPRESSIONS:**
   - Months ("3 months", "six months", "quarter", "semester")
   - Years ("1 year", "annual", "yearly")
   - Weeks/Days ("90 days", "12 weeks")
   - Academic terms ("one semester", "academic year")

4. **CONTRACT TERMS:**
   - Probation periods
   - Notice periods  
   - Renewal terms
   - Extension clauses

**OUTPUT FORMAT:**
Provide extraction results in JSON format:

{{
  "duration_found": true/false,
  "start_date": {{
    "text_found": "exact text mentioning start date",
    "interpreted_date": "YYYY-MM-DD or 'not_specified'",
    "date_type": "explicit/relative/academic_term"
  }},
  "end_date": {{
    "text_found": "exact text mentioning end date", 
    "interpreted_date": "YYYY-MM-DD or 'not_specified'",
    "date_type": "explicit/relative/duration_based"
  }},
  "duration_expressions": [
    {{
      "text_found": "exact text mentioning duration",
      "duration_value": "number extracted",
      "duration_unit": "months/years/days/weeks",
      "duration_type": "total_contract/probation/notice/other"
    }}
  ],
  "calculated_duration": {{
    "total_months": "number of months calculated",
    "total_days": "number of days calculated", 
    "duration_breakdown": "detailed explanation of calculation",
    "confidence_level": "high/medium/low"
  }},
  "special_conditions": [
    "any conditional terms affecting duration"
  ],
  "calculation_notes": "explanation of how duration was determined"
}}

**EXTRACTION RULES:**
- Extract ALL duration mentions, even if they seem redundant
- Convert everything to months and days for standardization
- Handle ambiguous dates with best interpretation
- Note any conditional or variable duration terms
- If multiple durations exist, identify which applies to the user's query

**PERFORM EXTRACTION:**"""

        return PromptTemplate(
            input_variables=["retrieved_context", "user_query"],
            template=prompt_template
        )
    
    def extract_and_calculate_duration(self, retrieved_context: str, user_query: str) -> Dict[str, Any]:
        """Extract duration information using LLM and calculate totals."""
        try:
            # Create extraction prompt
            formatted_prompt = self.duration_extraction_prompt.format(
                retrieved_context=retrieved_context,
                user_query=user_query
            )
            
            # Get extraction from LLM
            response = self.llm.invoke(formatted_prompt)
            extraction_text = response.content
            
            # Parse JSON response
            duration_data = self._parse_duration_extraction(extraction_text)
            
            # Enhanced calculation with flexible date parsing
            if duration_data.get('duration_found', False):
                enhanced_calc = self._enhanced_duration_calculation(duration_data)
                duration_data['enhanced_calculation'] = enhanced_calc
            
            return duration_data
            
        except Exception as e:
            logger.error(f"Duration extraction error: {str(e)}")
            return {
                'duration_found': False,
                'error': str(e),
                'fallback_message': 'Could not extract duration information'
            }
    
    def _parse_duration_extraction(self, extraction_text: str) -> Dict[str, Any]:
        """Parse LLM extraction response."""
        try:
            # Extract JSON from response
            json_start = extraction_text.find('{')
            json_end = extraction_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_text = extraction_text[json_start:json_end]
                parsed_data = json.loads(json_text)
                return parsed_data
            else:
                return {'duration_found': False, 'error': 'No valid JSON found'}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return {'duration_found': False, 'error': 'JSON parsing failed'}
    
    def _enhanced_duration_calculation(self, duration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced duration calculation with flexible date parsing."""
        try:
            calc_result = {
                'calculation_method': 'enhanced_flexible',
                'total_months': 0,
                'total_days': 0,
                'calculation_details': [],
                'confidence': 'medium'
            }
            
            # Method 1: Direct duration expressions
            duration_expressions = duration_data.get('duration_expressions', [])
            if duration_expressions:
                for expr in duration_expressions:
                    try:
                        value = float(expr.get('duration_value', 0))
                        unit = expr.get('duration_unit', '').lower()
                        
                        if 'month' in unit:
                            calc_result['total_months'] += value
                            calc_result['calculation_details'].append(f"Added {value} months from '{expr.get('text_found', '')}'")
                        elif 'year' in unit:
                            calc_result['total_months'] += (value * 12)
                            calc_result['calculation_details'].append(f"Added {value} years ({value * 12} months) from '{expr.get('text_found', '')}'")
                        elif 'day' in unit:
                            calc_result['total_days'] += value
                            calc_result['calculation_details'].append(f"Added {value} days from '{expr.get('text_found', '')}'")
                        elif 'week' in unit:
                            calc_result['total_days'] += (value * 7)
                            calc_result['calculation_details'].append(f"Added {value} weeks ({value * 7} days) from '{expr.get('text_found', '')}'")
                    except (ValueError, TypeError):
                        continue
            
            # Method 2: Start and end date calculation
            start_date = duration_data.get('start_date', {})
            end_date = duration_data.get('end_date', {})
            
            if (start_date.get('interpreted_date') and start_date.get('interpreted_date') != 'not_specified' and
                end_date.get('interpreted_date') and end_date.get('interpreted_date') != 'not_specified'):
                
                try:
                    start_dt = parser.parse(start_date['interpreted_date'])
                    end_dt = parser.parse(end_date['interpreted_date'])
                    
                    # Calculate difference
                    delta = relativedelta(end_dt, start_dt)
                    date_calc_months = delta.years * 12 + delta.months
                    date_calc_days = delta.days
                    
                    # If we have both methods, use the more specific one
                    if calc_result['total_months'] == 0:
                        calc_result['total_months'] = date_calc_months
                        calc_result['total_days'] += date_calc_days
                        calc_result['calculation_details'].append(f"Calculated from start date {start_date['interpreted_date']} to end date {end_date['interpreted_date']}")
                        calc_result['confidence'] = 'high'
                    else:
                        # Compare methods for validation
                        diff_months = abs(calc_result['total_months'] - date_calc_months)
                        if diff_months <= 1:  # Within 1 month tolerance
                            calc_result['confidence'] = 'high'
                            calc_result['calculation_details'].append(f"Validated: Duration expression matches date calculation ({date_calc_months} months)")
                        else:
                            calc_result['calculation_details'].append(f"Warning: Duration expression ({calc_result['total_months']} months) differs from date calculation ({date_calc_months} months)")
                            calc_result['confidence'] = 'medium'
                
                except Exception as e:
                    calc_result['calculation_details'].append(f"Date parsing error: {str(e)}")
            
            # Convert excess days to months
            if calc_result['total_days'] >= 30:
                additional_months = calc_result['total_days'] // 30
                remaining_days = calc_result['total_days'] % 30
                calc_result['total_months'] += additional_months
                calc_result['total_days'] = remaining_days
                calc_result['calculation_details'].append(f"Converted {additional_months * 30} days to {additional_months} months")
            
            # Generate summary
            if calc_result['total_months'] > 0 or calc_result['total_days'] > 0:
                summary_parts = []
                if calc_result['total_months'] > 0:
                    summary_parts.append(f"{calc_result['total_months']} month{'s' if calc_result['total_months'] != 1 else ''}")
                if calc_result['total_days'] > 0:
                    summary_parts.append(f"{calc_result['total_days']} day{'s' if calc_result['total_days'] != 1 else ''}")
                
                calc_result['duration_summary'] = " and ".join(summary_parts)
                calc_result['total_duration_text'] = calc_result['duration_summary']
            else:
                calc_result['duration_summary'] = "Duration could not be calculated"
                calc_result['total_duration_text'] = "Unknown duration"
            
            return calc_result
            
        except Exception as e:
            logger.error(f"Enhanced calculation error: {str(e)}")
            return {
                'calculation_method': 'failed',
                'error': str(e),
                'total_duration_text': 'Calculation failed'
            }

class EnhancedContradictionResolver:
    """
    Enhanced contradiction resolver with web search grounding for legal queries
    and specialized duration calculation capabilities.
    Updated to use HuggingFace embeddings for pipeline consistency.
    """
    
    def __init__(self, google_api_key: str, chroma_config: Optional[ChromaConfig] = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the enhanced contradiction resolver.
        
        Args:
            google_api_key: Google API key for Gemini
            chroma_config: Chroma cloud configuration
            embedding_model: HuggingFace embedding model name
        """
        self.google_api_key = google_api_key
        self.embedding_model_name = embedding_model
        
        # Initialize Chroma configuration for cloud service
        self.chroma_config = chroma_config or ChromaConfig.from_environment()
        self.chroma_factory = ChromaClientFactory(self.chroma_config)
        
        # Initialize Gemini client for grounding search
        self.gemini_client = genai.Client(api_key=google_api_key)
        
        # Configure grounding tool for legal searches
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        self.gemini_grounding_config = types.GenerateContentConfig(
            tools=[self.grounding_tool],
            temperature=0.1
        )
        
        # Initialize regular LLM for processing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=4096
        )
        
        # Initialize HuggingFace embeddings (replacing Gemini embeddings)
        logger.info(f"Loading HuggingFace embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize duration calculator
        self.duration_calculator = DurationCalculator(google_api_key)
        
        # Create prompt templates
        self.legal_search_prompt = self._create_legal_search_prompt()
        self.contradiction_search_prompt = self._create_contradiction_search_prompt()
        self.resolution_prompt = self._create_resolution_prompt()
        self.query_classifier_prompt = self._create_query_classifier_prompt()
    
    def get_vectorstore(self, collection_name: str = "legal_documents") -> Chroma:
        """
        Get ChromaDB vectorstore with HuggingFace embeddings from cloud service.
        
        Args:
            collection_name: ChromaDB collection name
            
        Returns:
            Chroma vectorstore instance
        """
        return self.chroma_factory.get_vectorstore(
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
    
    def _create_query_classifier_prompt(self) -> PromptTemplate:
        """Create prompt for classifying query types."""
        
        prompt_template = """You are a legal query classifier. Analyze the user query and determine what type of legal information they're seeking.

**USER QUERY:** {user_query}
**DOCUMENT TYPE:** {document_type}
**CONTRADICTORY FINDINGS:** {contradictory_findings}

**CLASSIFICATION TASK:**
Determine if this query falls into any of these special categories:

1. **DURATION/TENURE QUERIES:**
   - Internship duration, contract length, employment period
   - Probation period, notice period, project timeline
   - Keywords: duration, period, length, time, months, years, tenure, term

2. **LEGAL SECTION/CLAUSE QUERIES:**
   - Specific legal provisions, rights, obligations
   - Contract clauses, policy sections, legal terms
   - Requires external legal research beyond document content

3. **CONTRADICTION RESOLUTION:**
   - Multiple conflicting answers found in document analysis
   - Needs targeted search to resolve specific contradictions

**OUTPUT FORMAT:**
{{
  "query_type": "duration_calculation|legal_research|contradiction_resolution|standard_document_query",
  "requires_web_search": true/false,
  "requires_duration_calculation": true/false,
  "confidence": 0.8,
  "reasoning": "explanation of classification",
  "search_keywords": ["keyword1", "keyword2"]
}}

**CLASSIFY THE QUERY:**"""

        return PromptTemplate(
            input_variables=["user_query", "document_type", "contradictory_findings"],
            template=prompt_template
        )
    
    def _create_legal_search_prompt(self) -> PromptTemplate:
        """Create prompt for legal web searches."""
        
        prompt_template = """You are a legal research specialist. Search for authoritative information about the user's legal query.

**QUERY CONTEXT:**
Document Type: {document_type}
User Question: {user_query}
Specific Legal Topic: {search_keywords}

**SEARCH FOCUS:**
Find authoritative, current information about:
- Legal requirements and regulations
- Standard practices in {document_type} documents
- Rights and obligations related to the query
- Legal precedents or established interpretations
- Regulatory compliance requirements

**SEARCH STRATEGY:**
- Target government websites (.gov domains)
- Look for legal authorities and regulatory bodies
- Find legal databases and official documentation
- Search for legal advice and professional guidance
- Include jurisdiction-specific information (India/local laws)

**SEARCH QUERIES TO USE:**
- "{user_query} legal requirements India"
- "{document_type} {search_keywords} law regulations"
- "legal rights {search_keywords} India employment law"
- "standard {document_type} {search_keywords} provisions"
- "{search_keywords} legal compliance requirements"

Provide comprehensive legal research with authoritative sources and current legal standards.
"""

        return PromptTemplate(
            input_variables=["document_type", "user_query", "search_keywords"],
            template=prompt_template
        )
    
    def _create_contradiction_search_prompt(self) -> PromptTemplate:
        """Create prompt for contradiction-specific searches."""
        
        prompt_template = """You are a contradiction research specialist. Search for specific information to resolve contradictory findings.

**CONTRADICTION CONTEXT:**
Document Type: {document_type}
Original Query: {original_query}
Contradictory Findings: {contradictory_findings}
Contradiction Reasons: {contradiction_reasons}

**TARGETED SEARCH FOCUS:**
Find specific, authoritative information to resolve:
- Conflicting amounts, dates, or terms
- Different interpretations of the same clause
- Varying conditions or requirements
- Unclear or ambiguous language

**SEARCH STRATEGY:**
- Look for official clarifications and interpretations
- Find standard industry practices
- Search for legal precedents resolving similar issues
- Target authoritative sources for definitive answers

Search for definitive, authoritative information that can conclusively resolve the identified contradictions.
"""

        return PromptTemplate(
            input_variables=["document_type", "original_query", "contradictory_findings", "contradiction_reasons"],
            template=prompt_template
        )
    
    def _create_resolution_prompt(self) -> PromptTemplate:
        """Create enhanced resolution prompt with web search integration."""
        
        prompt_template = """You are an expert legal document analyst with access to both document content and external legal research.

**RESOLUTION CONTEXT:**
Document Type: {document_type}
Original Query: {original_query}
Query Type: {query_type}

**INFORMATION SOURCES:**

**1. DOCUMENT CONTENT:**
{retrieved_context}

**2. EXTERNAL LEGAL RESEARCH:**
{web_search_results}

**3. CONTRADICTION ANALYSIS:**
Contradictory Findings: {contradictory_findings}
Contradiction Reasons: {contradiction_reasons}

**4. DURATION CALCULATION:**
{duration_calculation_results}

**RESOLUTION INSTRUCTIONS:**

**FOR DURATION QUERIES:**
- Use duration calculation results as primary source
- Cross-reference with document content for validation
- Provide specific timeframes with clear breakdown

**FOR LEGAL RESEARCH QUERIES:**
- Integrate document content with external legal research
- Cite authoritative sources when available
- Explain legal requirements and standard practices

**FOR CONTRADICTION RESOLUTION:**
- Use both document analysis and external research
- Provide definitive resolution with supporting evidence
- Explain why one interpretation is correct

**RESPONSE FORMAT:**
```
**COMPREHENSIVE ANSWER:**

[Definitive answer to the original query integrating all available information]

**RESOLUTION TYPE:** [Duration Calculation/Legal Research/Contradiction Resolution/Standard Analysis]

**SUPPORTING EVIDENCE:**
- Document Evidence: [Specific quotes or references from the document]
- External Research: [Key findings from web search with sources if available]
- Duration Analysis: [Calculation details if applicable]

**LEGAL CONTEXT:** [Relevant legal background or requirements if applicable]

**CONFIDENCE LEVEL:** [High/Medium/Low with explanation]

**ADDITIONAL NOTES:** [Any caveats, limitations, or recommendations]
```

**PROVIDE COMPREHENSIVE RESOLUTION:**"""

        return PromptTemplate(
            input_variables=["document_type", "original_query", "query_type", "retrieved_context", 
                           "web_search_results", "contradictory_findings", "contradiction_reasons", 
                           "duration_calculation_results"],
            template=prompt_template
        )
    
    def resolve_contradictions(self, consensus_output: Dict[str, Any], collection_name: str = "legal_documents") -> Dict[str, Any]:
        """
        Enhanced contradiction resolution with web search and duration calculation.
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
            
            # Initialize ChromaDB vector store with HuggingFace embeddings
            vectorstore = self.get_vectorstore(collection_name)
            
            document_type = consensus_output.get('document_type', 'Unknown')
            original_query = consensus_output.get('original_query', '')
            embedding_model = consensus_output.get('embedding_model', self.embedding_model_name)
            consensus_evaluations = consensus_output.get('consensus_evaluations', {})
            
            resolution_results = {
                'document_type': document_type,
                'original_query': original_query,
                'embedding_model': embedding_model,
                'collection_name': collection_name,
                'resolutions': {},
                'contradictory_queries_found': 0,
                'successfully_resolved': 0,
                'web_searches_performed': 0,
                'duration_calculations_performed': 0,
                'resolved_at': datetime.now().isoformat()
            }
            
            logger.info(f"Enhanced resolution using {embedding_model} embeddings")
            
            # Find queries that need resolution
            contradictory_queries = []
            for query_id, evaluation in consensus_evaluations.items():
                if evaluation.get('evaluation_success', False):
                    verdict = evaluation.get('final_verdict', {}).get('verdict', '')
                    confidence_level = evaluation.get('confidence_assessment', {}).get('confidence_level', '')
                    
                    # Check if this query needs enhanced resolution
                    if verdict == 'CONTRADICTORY' or confidence_level == 'contradictory' or confidence_level == 'low':
                        contradictory_queries.append({
                            'query_id': query_id,
                            'evaluation': evaluation
                        })
            
            resolution_results['contradictory_queries_found'] = len(contradictory_queries)
            
            if not contradictory_queries:
                logger.info("No contradictory queries found - no resolution needed")
                return resolution_results
            
            # Resolve each query with enhanced capabilities
            for query_info in contradictory_queries:
                query_id = query_info['query_id']
                evaluation = query_info['evaluation']
                
                logger.info(f"Enhanced resolution for query {query_id}")
                
                try:
                    resolution = self._enhanced_resolve_single_query(
                        query_id=query_id,
                        evaluation=evaluation,
                        vectorstore=vectorstore,
                        document_type=document_type,
                        original_user_query=original_query
                    )
                    
                    resolution_results['resolutions'][query_id] = resolution
                    
                    if resolution.get('resolution_success', False):
                        resolution_results['successfully_resolved'] += 1
                    
                    # Track enhanced features used
                    if resolution.get('web_search_performed', False):
                        resolution_results['web_searches_performed'] += 1
                    if resolution.get('duration_calculation_performed', False):
                        resolution_results['duration_calculations_performed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error in enhanced resolution for query {query_id}: {str(e)}")
                    resolution_results['resolutions'][query_id] = {
                        'query_id': query_id,
                        'error': str(e),
                        'resolution_success': False,
                        'final_answer': 'Enhanced resolution failed due to error'
                    }
            
            logger.info(f"Enhanced resolution completed. Resolved {resolution_results['successfully_resolved']}/{resolution_results['contradictory_queries_found']} queries")
            logger.info(f"Web searches: {resolution_results['web_searches_performed']}, Duration calculations: {resolution_results['duration_calculations_performed']}")
            
            return resolution_results
            
        except Exception as e:
            logger.error(f"Error in enhanced contradiction resolution: {str(e)}")
            return {
                'error': str(e),
                'document_type': consensus_output.get('document_type', 'Unknown'),
                'embedding_model': consensus_output.get('embedding_model', self.embedding_model_name),
                'resolutions': {},
                'resolved_at': datetime.now().isoformat()
            }
    
    def _enhanced_resolve_single_query(self, query_id: str, evaluation: Dict[str, Any], 
                                     vectorstore: Chroma, document_type: str, original_user_query: str) -> Dict[str, Any]:
        """Enhanced resolution for a single query with all capabilities."""
        try:
            # Extract information
            original_query = evaluation.get('original_query', '')
            contradictory_findings = self._extract_contradictory_findings(evaluation)
            contradiction_reasons = self._extract_contradiction_reasons(evaluation)
            
            # Step 1: Classify query type
            query_classification = self._classify_query_type(
                user_query=original_query,
                document_type=document_type,
                contradictory_findings=contradictory_findings
            )
            
            # Step 2: Get document context
            retrieved_context = self._get_document_context(
                query=original_query,
                vectorstore=vectorstore
            )
            
            # Step 3: Enhanced processing based on query type
            web_search_results = ""
            duration_calculation_results = ""
            web_search_performed = False
            duration_calculation_performed = False
            
            # Web search if needed
            if query_classification.get('requires_web_search', False):
                web_search_results = self._perform_legal_web_search(
                    user_query=original_query,
                    document_type=document_type,
                    search_keywords=query_classification.get('search_keywords', []),
                    contradictory_findings=contradictory_findings,
                    contradiction_reasons=contradiction_reasons
                )
                web_search_performed = True
            
            # Duration calculation if needed
            if query_classification.get('requires_duration_calculation', False):
                duration_data = self.duration_calculator.extract_and_calculate_duration(
                    retrieved_context=retrieved_context,
                    user_query=original_query
                )
                duration_calculation_results = json.dumps(duration_data, indent=2)
                duration_calculation_performed = True
            
            # Step 4: Generate comprehensive resolution
            resolution_answer = self._get_enhanced_resolution_answer(
                document_type=document_type,
                original_query=original_query,
                query_type=query_classification.get('query_type', 'standard'),
                retrieved_context=retrieved_context,
                web_search_results=web_search_results,
                contradictory_findings=contradictory_findings,
                contradiction_reasons=contradiction_reasons,
                duration_calculation_results=duration_calculation_results
            )
            
            return {
                'query_id': query_id,
                'original_query': original_query,
                'query_classification': query_classification,
                'contradictory_findings': contradictory_findings,
                'contradiction_reasons': contradiction_reasons,
                'web_search_performed': web_search_performed,
                'duration_calculation_performed': duration_calculation_performed,
                'resolution_answer': resolution_answer,
                'resolution_success': True,
                'enhanced_features_used': {
                    'web_search': web_search_performed,
                    'duration_calculation': duration_calculation_performed,
                    'query_classification': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced single query resolution: {str(e)}")
            return {
                'query_id': query_id,
                'error': str(e),
                'resolution_success': False
            }
    
    def _classify_query_type(self, user_query: str, document_type: str, contradictory_findings: List[str]) -> Dict[str, Any]:
        """Classify the query to determine processing approach."""
        try:
            formatted_prompt = self.query_classifier_prompt.format(
                user_query=user_query,
                document_type=document_type,
                contradictory_findings=json.dumps(contradictory_findings)
            )
            
            response = self.llm.invoke(formatted_prompt)
            classification_text = response.content
            
            # Parse classification
            try:
                json_start = classification_text.find('{')
                json_end = classification_text.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    json_text = classification_text[json_start:json_end]
                    classification = json.loads(json_text)
                    return classification
                else:
                    return {'query_type': 'standard_document_query', 'requires_web_search': False, 'requires_duration_calculation': False}
            except json.JSONDecodeError:
                return {'query_type': 'standard_document_query', 'requires_web_search': False, 'requires_duration_calculation': False}
                
        except Exception as e:
            logger.error(f"Query classification error: {str(e)}")
            return {'query_type': 'standard_document_query', 'requires_web_search': False, 'requires_duration_calculation': False}
    
    def _perform_legal_web_search(self, user_query: str, document_type: str, search_keywords: List[str], 
                                 contradictory_findings: List[str], contradiction_reasons: str) -> str:
        """Perform web search using Gemini grounding."""
        try:
            # Determine search type
            if contradictory_findings:
                # Use contradiction-specific search
                search_prompt = self.contradiction_search_prompt.format(
                    document_type=document_type,
                    original_query=user_query,
                    contradictory_findings=json.dumps(contradictory_findings),
                    contradiction_reasons=contradiction_reasons
                )
            else:
                # Use general legal search
                search_prompt = self.legal_search_prompt.format(
                    document_type=document_type,
                    user_query=user_query,
                    search_keywords=' '.join(search_keywords)
                )
            
            # Perform grounding search
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=search_prompt,
                config=self.gemini_grounding_config,
            )
            
            search_results = response.text
            
            # Extract sources if available
            sources_info = ""
            if response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                if hasattr(metadata, 'grounding_chunks'):
                    sources = [
                        f"Source: {getattr(chunk.web, 'title', 'Unknown')} - {getattr(chunk.web, 'uri', 'Unknown')}"
                        for chunk in metadata.grounding_chunks
                    ]
                    if sources:
                        sources_info = "\n\nSources consulted:\n" + "\n".join(sources)
            
            return search_results + sources_info
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return f"Web search failed: {str(e)}"
    
    def _get_document_context(self, query: str, vectorstore: Chroma) -> str:
        """Retrieve relevant context from document."""
        try:
            docs = vectorstore.similarity_search(query, k=5)
            
            if docs:
                context_parts = []
                for i, doc in enumerate(docs):
                    context_parts.append(f"**Document Section {i+1}:**")
                    context_parts.append(doc.page_content)
                    context_parts.append("")
                
                return "\n".join(context_parts)
            else:
                return "No relevant document context found."
                
        except Exception as e:
            logger.error(f"Document context retrieval error: {str(e)}")
            return f"Context retrieval error: {str(e)}"
    
    def _get_enhanced_resolution_answer(self, document_type: str, original_query: str, query_type: str,
                                      retrieved_context: str, web_search_results: str, 
                                      contradictory_findings: List[str], contradiction_reasons: str,
                                      duration_calculation_results: str) -> str:
        """Generate comprehensive resolution using all available information."""
        try:
            formatted_prompt = self.resolution_prompt.format(
                document_type=document_type,
                original_query=original_query,
                query_type=query_type,
                retrieved_context=retrieved_context,
                web_search_results=web_search_results,
                contradictory_findings=json.dumps(contradictory_findings),
                contradiction_reasons=contradiction_reasons,
                duration_calculation_results=duration_calculation_results
            )
            
            response = self.llm.invoke(formatted_prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Enhanced resolution generation error: {str(e)}")
            return f"Error generating enhanced resolution: {str(e)}"
    
    def _extract_contradictory_findings(self, evaluation: Dict[str, Any]) -> List[str]:
        """Extract contradictory findings from consensus evaluation."""
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
        """Extract reasons for contradiction from consensus evaluation."""
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

# Example usage with complete pipeline integration
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get API key from environment
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.strip() == "":
        print("Please set your Google API key")
        exit(1)
    
    # Import all previous components
    import os
    from document_classifier import LegalDocumentClassifier
    from query_analyzer import LegalDocumentQueryAnalyzer
    from search_query_generator import SearchQueryGenerator
    from llm_enhanced_rag import LLMEnhancedRAG
    from consensus_evaluator import ConsensusEvaluator
    
    # Initialize all components with HuggingFace embeddings
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    classifier = LegalDocumentClassifier(GOOGLE_API_KEY, embedding_model=embedding_model)
    analyzer = LegalDocumentQueryAnalyzer(GOOGLE_API_KEY)
    generator = SearchQueryGenerator(GOOGLE_API_KEY)
    rag_system = LLMEnhancedRAG(GOOGLE_API_KEY, embedding_model=embedding_model)
    evaluator = ConsensusEvaluator([GOOGLE_API_KEY], embedding_model=embedding_model)
    resolver = EnhancedContradictionResolver(GOOGLE_API_KEY, embedding_model=embedding_model)
    
    # Complete workflow with enhanced contradiction resolution
    pdf_path = "Suryansh_OL.docx (1) (1) (1).pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        print("=== COMPLETE WORKFLOW WITH ENHANCED RESOLUTION (Steps 1-5) ===")
        print(f"Using HuggingFace Embedding Model: {embedding_model}")
        
        # Step 1: Document Classification
        print("\n1. CLASSIFYING DOCUMENT...")
        classification_result = classifier.process_pdf_and_classify(pdf_path)
        
        if classification_result['success']:
            print(f"   Document Type: {classification_result['classification']['document_type']}")
            print(f"   Embedding Model: {classification_result['embedding_model']}")
            
            # Step 2: Query Analysis
            print("\n2. ANALYZING USER QUERY...")
            user_query = "What is the monthly stipend amount and internship duration?"
            query_analysis = analyzer.analyze_query(user_query, classification_result)
            
            if 'error' not in query_analysis:
                print(f"   Single Queries: {len(query_analysis['single_queries'])}")
                print(f"   Hybrid Queries: {len(query_analysis['hybrid_queries'])}")
                
                # Step 3: Search Angle Generation
                print("\n3. GENERATING SEARCH ANGLES...")
                search_angles = generator.generate_search_angles(query_analysis)
                
                if 'error' not in search_angles:
                    print(f"   Total Search Angles: {search_angles['total_angles_generated']}")
                    
                    # Step 4A: Individual RAG Processing
                    print("\n4A. PROCESSING INDIVIDUAL SEARCH ANGLES...")
                    rag_results = rag_system.process_all_search_angles(search_angles)
                    
                    if 'error' not in rag_results:
                        print(f"   Individual Answers Generated: {rag_results['successful_answers']}")
                        print(f"   Embedding Model: {rag_results['embedding_model']}")
                        
                        # Step 4B: Consensus Evaluation
                        print("\n4B. EVALUATING CONSENSUS ACROSS ANSWERS...")
                        consensus_results = evaluator.evaluate_consensus(rag_results)
                        
                        if 'error' not in consensus_results:
                            summary = consensus_results['overall_summary']
                            print(f"   Queries Evaluated: {summary['total_queries']}")
                            print(f"   Success Rate: {summary['success_rate']:.1f}%")
                            print(f"   Embedding Model: {consensus_results['embedding_model']}")
                            
                            # Step 5: Enhanced Contradiction Resolution
                            print("\n5. ENHANCED CONTRADICTION RESOLUTION...")
                            resolution_results = resolver.resolve_contradictions(consensus_results)
                            
                            if 'error' not in resolution_results:
                                print(f"   Contradictory Queries Found: {resolution_results['contradictory_queries_found']}")
                                print(f"   Successfully Resolved: {resolution_results['successfully_resolved']}")
                                print(f"   Web Searches Performed: {resolution_results['web_searches_performed']}")
                                print(f"   Duration Calculations: {resolution_results['duration_calculations_performed']}")
                                print(f"   Embedding Model: {resolution_results['embedding_model']}")
                                
                                # Show resolution results
                                if resolution_results['resolutions']:
                                    print("\n   Enhanced Resolution Results:")
                                    for query_id, resolution in resolution_results['resolutions'].items():
                                        if resolution.get('resolution_success', False):
                                            features_used = resolution.get('enhanced_features_used', {})
                                            print(f"   Query {query_id}:")
                                            print(f"     Query Type: {resolution.get('query_classification', {}).get('query_type', 'unknown')}")
                                            print(f"     Web Search: {features_used.get('web_search', False)}")
                                            print(f"     Duration Calc: {features_used.get('duration_calculation', False)}")
                                            print(f"     Resolution: {resolution.get('resolution_answer', 'N/A')[:150]}...")
                                        else:
                                            print(f"   Query {query_id}: Resolution failed - {resolution.get('error', 'Unknown')}")
                                else:
                                    print("\n   No contradictory queries requiring enhanced resolution")
                                    
                                print(f"\n✅ COMPLETE PIPELINE SUCCESS!")
                                print(f"   Total Processing Time: Complete")
                                print(f"   Embedding Consistency: {embedding_model}")
                                print(f"   Enhanced Features: Web Search + Duration Calculation")
                            else:
                                print(f"   Error: {resolution_results['error']}")
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
        print("\n=== ENHANCED PIPELINE FEATURES ===")
        print("This complete pipeline now includes:")
        print()
        print("🔧 CONSISTENT ARCHITECTURE:")
        print("   ✅ HuggingFace embeddings across all components")
        print("   ✅ Gemini 2.0 Flash for LLM reasoning")
        print("   ✅ ChromaDB with unified embedding model")
        print("   ✅ Pipeline-wide compatibility tracking")
        print()
        print("🚀 ENHANCED RESOLUTION FEATURES:")
        print("   ✅ Intelligent query classification")
        print("   ✅ Web search grounding for legal research")
        print("   ✅ Advanced duration calculation tools")
        print("   ✅ Multi-source information synthesis")
        print()
        print("📊 PROCESSING CAPABILITIES:")
        print("   • Document Classification → Search Angles → RAG Processing")
        print("   • Consensus Evaluation → Enhanced Resolution")
        print("   • Duration extraction with flexible date parsing")
        print("   • Legal research with authoritative sources")
        print("   • Contradiction resolution with evidence synthesis")
        print()
        print("🎯 SPECIALIZED QUERY HANDLING:")
        print("   📅 Duration Queries: 'How long is the internship?'")
        print("   ⚖️  Legal Queries: 'What are my termination rights?'")
        print("   🔄 Contradictions: 'Found conflicting stipend amounts'")
        print("   📋 Standard Queries: 'What is the job location?'")
        print()
        print("💡 READY FOR PRODUCTION:")
        print("   • All components use same embedding model")
        print("   • Error handling and fallback mechanisms")
        print("   • Comprehensive logging and tracking")
        print("   • Enhanced features activate automatically")
        print("   • Full pipeline integration maintained")
    
    print("\n" + "="*60)
    print("INTEGRATION INSTRUCTIONS")
    print("="*60)
    print("1. Replace old imports with:")
    print("   from enhanced_contradiction_resolver import EnhancedContradictionResolver")
    print()
    print("2. Initialize with same embedding model:")
    print("   resolver = EnhancedContradictionResolver(")
    print("       GOOGLE_API_KEY,")
    print("       embedding_model='sentence-transformers/all-MiniLM-L6-v2'")
    print("   )")
    print()
    print("3. Use same interface:")
    print("   resolution_results = resolver.resolve_contradictions(consensus_results)")
    print()
    print("4. Enhanced features activate automatically based on:")
    print("   • Query content analysis")
    print("   • Contradiction patterns")
    print("   • Document type classification")
    print()
    print("5. New output includes:")
    print("   • web_searches_performed: count")
    print("   • duration_calculations_performed: count")
    print("   • enhanced_features_used: detailed breakdown")
    print("   • embedding_model: consistency tracking")