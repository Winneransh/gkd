"""
search_query_generator.py

Generates multiple search angles for each query component to ensure comprehensive
information retrieval and enable consensus validation.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchQueryGenerator:
    """
    Generates multiple search angles for each query to ensure comprehensive
    information retrieval from legal documents.
    """
    
    def __init__(self, google_api_key: str):
        """
        Initialize the search query generator.
        
        Args:
            google_api_key: Google API key for Gemini
        """
        self.google_api_key = google_api_key
        
        # Initialize Gemini 2.5 Flash LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=4096
        )
        
        # Create search generation prompt template
        self.search_generation_prompt = self._create_search_generation_prompt()
    
    def _create_search_generation_prompt(self) -> PromptTemplate:
        """
        Create a comprehensive system prompt for generating search angles.
        
        Returns:
            PromptTemplate for search angle generation
        """
        
        prompt_template = """You are a search query optimization expert specializing in legal document analysis. Your job is to generate 5 different search angles for each input query to approach the same information need from different perspectives.

**DOCUMENT TYPE CONTEXT:** {document_type}

**SEARCH ANGLE GENERATION PRINCIPLES:**

Each search angle should:
1. **Use different keywords and phrasings** - Avoid repetition, use synonyms and alternative terms
2. **Focus on a specific aspect** - Each angle targets a different facet of the same query
3. **Be optimized for semantic search** - Use natural language that matches document content
4. **Avoid redundancy** - While maintaining complementary coverage
5. **Cover multiple perspectives** - Exact matches, category matches, alternative phrasings, edge cases, broader context

**SEARCH ANGLE CATEGORIES:**
1. **Exact Match**: Direct terminology and specific phrases
2. **Category/Conceptual**: Broader category or concept-based terms
3. **Alternative Phrasing**: Different ways to express the same query
4. **Edge Cases**: Boundary conditions, exceptions, special circumstances
5. **Contextual/Broader**: Related context that might contain relevant information

**DOCUMENT-SPECIFIC SEARCH STRATEGIES:**

**OFFER LETTER:**
- Position angles: job title, role, responsibilities, department, level
- Compensation angles: salary, stipend, bonus, benefits, payment terms
- Temporal angles: start date, duration, probation, notice period, contract term
- Work arrangement angles: remote, location, hours, flexibility, reporting
- Legal angles: terms, conditions, obligations, rights, termination

**HEALTH INSURANCE:**
- Medical angles: procedures, treatments, conditions, coverage, exclusions
- Financial angles: premiums, deductibles, co-pays, limits, reimbursement
- Eligibility angles: age, demographics, pre-conditions, family status
- Geographic angles: coverage area, network, providers, location restrictions
- Temporal angles: waiting periods, effective dates, policy duration

**RENTAL AGREEMENT:**
- Property angles: type, location, condition, amenities, specifications
- Financial angles: rent, deposits, fees, utilities, escalations, penalties
- Legal angles: terms, obligations, rights, restrictions, termination
- Temporal angles: lease term, renewal, notice periods, move dates
- Usage angles: occupancy, pets, modifications, permitted use

**OTHER LEGAL DOCUMENTS:**
- Adapt search strategies based on document type and query context
- Focus on legal terminology, financial terms, dates, parties, obligations

**EXAMPLES:**

**Example 1 - Offer Letter Compensation Query:**
Input Query: "monthly stipend amount"
Document Type: Offer Letter

Search Angles:
1. Exact: "monthly stipend INR amount payment"
2. Categorical: "compensation salary monthly payment terms"
3. Alternative: "stipend per month remuneration allowance"
4. Contextual: "intern payment structure monthly compensation"
5. Financial: "monthly financial benefit stipend salary"

**Example 2 - Offer Letter Duration Query:**
Input Query: "internship duration period"
Document Type: Offer Letter

Search Angles:
1. Exact: "internship duration 3 months period"
2. Temporal: "contract term length timeline months"
3. Alternative: "internship tenure duration time period"
4. Contextual: "appointment period contract duration term"
5. Specific: "3 months internship duration length"

**Example 3 - Health Insurance Coverage Query:**
Input Query: "ACL surgery coverage"
Document Type: Health Insurance Policy

Search Angles:
1. Exact: "ACL reconstruction surgery coverage benefits"
2. Medical: "anterior cruciate ligament surgical procedure covered"
3. Categorical: "orthopedic knee surgery procedures coverage"
4. Alternative: "ligament repair surgery medical coverage"
5. Contextual: "surgical procedures orthopedic benefits ACL"

**OUTPUT FORMAT:**
For each input query, provide exactly 5 search angles in the following JSON structure:

{{
  "search_angles": [
    {{
      "angle_id": "A1",
      "query": "search query text optimized for semantic search",
      "focus": "exact_match|categorical|alternative_phrasing|edge_cases|contextual",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }},
    {{
      "angle_id": "A2",
      "query": "different search query approach",
      "focus": "focus_category",
      "keywords": ["different", "keywords", "list"]
    }}
    // ... 3 more angles
  ]
}}

**INPUT QUERY TO PROCESS:**
Document Type: {document_type}
Query Component: {query_component}
Original Query: "{original_query}"
Keywords: {query_keywords}

**GENERATE 5 SEARCH ANGLES:**"""

        return PromptTemplate(
            input_variables=["document_type", "query_component", "original_query", "query_keywords"],
            template=prompt_template
        )
    
    def generate_search_angles(self, query_analysis_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate search angles for all queries from the query analyzer output.
        
        Args:
            query_analysis_output: Complete output from LegalDocumentQueryAnalyzer.analyze_query()
            
        Returns:
            Dictionary containing search angles for all queries
        """
        try:
            # Validate input
            if 'error' in query_analysis_output:
                return {
                    'error': 'Query analysis failed',
                    'analysis_error': query_analysis_output['error'],
                    'search_angles': {},
                    'generated_at': datetime.now().isoformat()
                }
            
            document_type = query_analysis_output.get('document_type', 'Unknown')
            single_queries = query_analysis_output.get('single_queries', [])
            hybrid_queries = query_analysis_output.get('hybrid_queries', [])
            
            search_angles_result = {
                'document_type': document_type,
                'original_query': query_analysis_output.get('original_query', ''),
                'search_angles': {},
                'total_angles_generated': 0,
                'generated_at': datetime.now().isoformat()
            }
            
            # Generate angles for single queries
            for query in single_queries:
                angles = self._generate_angles_for_query(
                    query=query,
                    document_type=document_type,
                    query_type='single'
                )
                search_angles_result['search_angles'][query['id']] = angles
                search_angles_result['total_angles_generated'] += len(angles.get('search_angles', []))
            
            # Generate angles for hybrid queries
            for query in hybrid_queries:
                angles = self._generate_angles_for_query(
                    query=query,
                    document_type=document_type,
                    query_type='hybrid'
                )
                search_angles_result['search_angles'][query['id']] = angles
                search_angles_result['total_angles_generated'] += len(angles.get('search_angles', []))
            
            logger.info(f"Generated {search_angles_result['total_angles_generated']} search angles for {len(single_queries + hybrid_queries)} queries")
            
            return search_angles_result
            
        except Exception as e:
            logger.error(f"Error generating search angles: {str(e)}")
            return {
                'error': str(e),
                'document_type': query_analysis_output.get('document_type', 'Unknown'),
                'search_angles': {},
                'generated_at': datetime.now().isoformat()
            }
    
    def _generate_angles_for_query(self, query: Dict[str, Any], document_type: str, query_type: str) -> Dict[str, Any]:
        """
        Generate 5 search angles for a single query.
        
        Args:
            query: Single query dictionary from analyzer
            document_type: Type of document
            query_type: 'single' or 'hybrid'
            
        Returns:
            Dictionary containing 5 search angles for the query
        """
        try:
            # Prepare input for prompt
            query_text = query.get('query', '')
            query_component = query.get('component', 'general') if query_type == 'single' else 'hybrid'
            query_keywords = query.get('keywords', [])
            
            # Create the prompt
            formatted_prompt = self.search_generation_prompt.format(
                document_type=document_type,
                query_component=query_component,
                original_query=query_text,
                query_keywords=json.dumps(query_keywords)
            )
            
            # Get search angles from Gemini
            response = self.llm.invoke(formatted_prompt)
            angles_text = response.content
            
            # Parse the response
            parsed_angles = self._parse_search_angles_response(angles_text)
            
            # Add metadata
            result = {
                'query_id': query.get('id', 'unknown'),
                'query_type': query_type,
                'original_query': query_text,
                'component': query_component,
                'search_angles': parsed_angles.get('search_angles', []),
                'generation_success': len(parsed_angles.get('search_angles', [])) == 5
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating angles for query {query.get('id', 'unknown')}: {str(e)}")
            return {
                'query_id': query.get('id', 'unknown'),
                'query_type': query_type,
                'original_query': query.get('query', ''),
                'error': str(e),
                'search_angles': [],
                'generation_success': False
            }
    
    def _parse_search_angles_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the search angles response from Gemini.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed search angles dictionary
        """
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_text = response_text[json_start:json_end]
                parsed_json = json.loads(json_text)
                
                # Validate structure
                if 'search_angles' not in parsed_json:
                    parsed_json['search_angles'] = []
                
                # Ensure we have 5 angles
                angles = parsed_json['search_angles']
                if len(angles) != 5:
                    logger.warning(f"Expected 5 search angles, got {len(angles)}")
                
                # Add angle IDs if missing
                for i, angle in enumerate(angles):
                    if 'angle_id' not in angle:
                        angle['angle_id'] = f"A{i+1}"
                
                return parsed_json
            else:
                return self._fallback_angles_parse(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return self._fallback_angles_parse(response_text)
        except Exception as e:
            logger.error(f"Error parsing search angles response: {str(e)}")
            return {
                'search_angles': [],
                'parse_error': str(e),
                'raw_response': response_text
            }
    
    def _fallback_angles_parse(self, response_text: str) -> Dict[str, Any]:
        """
        Fallback parsing when JSON parsing fails.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Basic search angles structure
        """
        return {
            'search_angles': [],
            'parse_method': 'fallback',
            'raw_response': response_text,
            'note': 'JSON parsing failed, manual review needed'
        }
    
    def get_all_search_queries(self, search_angles_output: Dict[str, Any]) -> List[str]:
        """
        Extract all search query strings from the search angles output.
        
        Args:
            search_angles_output: Output from generate_search_angles method
            
        Returns:
            List of all search query strings
        """
        all_queries = []
        
        search_angles = search_angles_output.get('search_angles', {})
        
        for query_id, query_data in search_angles.items():
            angles = query_data.get('search_angles', [])
            for angle in angles:
                query_text = angle.get('query', '')
                if query_text:
                    all_queries.append(query_text)
        
        return all_queries
    
    def get_search_summary(self, search_angles_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the search angles generation.
        
        Args:
            search_angles_output: Output from generate_search_angles method
            
        Returns:
            Summary dictionary
        """
        search_angles = search_angles_output.get('search_angles', {})
        
        total_queries = len(search_angles)
        total_angles = search_angles_output.get('total_angles_generated', 0)
        successful_generations = sum(1 for q in search_angles.values() if q.get('generation_success', False))
        
        return {
            'document_type': search_angles_output.get('document_type', 'Unknown'),
            'original_query': search_angles_output.get('original_query', ''),
            'total_queries_processed': total_queries,
            'total_search_angles_generated': total_angles,
            'successful_generations': successful_generations,
            'generation_success_rate': (successful_generations / total_queries * 100) if total_queries > 0 else 0,
            'average_angles_per_query': total_angles / total_queries if total_queries > 0 else 0
        }

# Example usage
if __name__ == "__main__":
    # Initialize components
    GOOGLE_API_KEY = ""  # Replace with your actual API key
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        print("Please set your Google API key")
        exit(1)
    
    # Import previous components
    from document_classifier import LegalDocumentClassifier
    from query_analyzer import LegalDocumentQueryAnalyzer
    
    # Initialize all components
    classifier = LegalDocumentClassifier(GOOGLE_API_KEY)
    analyzer = LegalDocumentQueryAnalyzer(GOOGLE_API_KEY)
    generator = SearchQueryGenerator(GOOGLE_API_KEY)
    
    # Example end-to-end workflow
    pdf_path = "Suryansh_OL.docx (1) (1) (1).pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        print("=== COMPLETE END-TO-END WORKFLOW ===")
        
        # Step 1: Classify document
        print("\n1. CLASSIFYING DOCUMENT...")
        classification_result = classifier.process_pdf_and_classify(pdf_path)
        
        if classification_result['success']:
            print(f"   Document Type: {classification_result['classification']['document_type']}")
            
            # Step 2: Analyze user query
            print("\n2. ANALYZING USER QUERY...")
            user_query = "What is the monthly stipend amount and duration?"
            query_analysis = analyzer.analyze_query(user_query, classification_result)
            
            if 'error' not in query_analysis:
                print(f"   Single Queries: {len(query_analysis['single_queries'])}")
                print(f"   Hybrid Queries: {len(query_analysis['hybrid_queries'])}")
                
                # Step 3: Generate search angles
                print("\n3. GENERATING SEARCH ANGLES...")
                search_angles = generator.generate_search_angles(query_analysis)
                
                if 'error' not in search_angles:
                    print(f"   Total Search Angles: {search_angles['total_angles_generated']}")
                    
                    # Display some examples
                    print("\n   Example Search Angles:")
                    for query_id, query_data in list(search_angles['search_angles'].items())[:2]:
                        print(f"   Query {query_id}:")
                        for angle in query_data.get('search_angles', [])[:3]:
                            print(f"     - {angle.get('query', 'N/A')}")
                    
                    # Get summary
                    summary = generator.get_search_summary(search_angles)
                    print(f"\n   Success Rate: {summary['generation_success_rate']:.1f}%")
                else:
                    print(f"   Error: {search_angles['error']}")
            else:
                print(f"   Error: {query_analysis['error']}")
        else:
            print(f"   Error: {classification_result.get('error', 'Unknown error')}")
    else:
        print(f"PDF file not found: {pdf_path}")
        
        # Demo with mock data for the Zeko AI offer letter
        print("\n=== DEMO WITH ZEKO AI OFFER LETTER ===")
        
        # Mock classification result for offer letter
        mock_classification = {
            'success': True,
            'document_name': 'zeko_ai_offer_letter.pdf',
            'classification': {
                'document_type': 'Offer Letter',
                'confidence': 98,
                'key_indicators': ['offer of employment', 'internship', 'stipend', 'terms and conditions'],
                'reasoning': 'Document contains employment offer terminology and internship details'
            }
        }
        
        # Test queries specific to the offer letter
        test_queries = [
            "What is the monthly stipend amount?",
            "How long is the probation period?",
            "What are the working hours and remote work policy?",
            "What is the notice period for leaving?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            # Step 2: Analyze query
            analysis = analyzer.analyze_query(query, mock_classification)
            
            # Step 3: Generate search angles
            if 'error' not in analysis:
                angles = generator.generate_search_angles(analysis)
                
                if 'error' not in angles:
                    print(f"Generated {angles['total_angles_generated']} search angles")
                    
                    # Show first few angles as example
                    first_query_id = list(angles['search_angles'].keys())[0]
                    first_query_angles = angles['search_angles'][first_query_id]['search_angles'][:3]
                    
                    for i, angle in enumerate(first_query_angles):
                        print(f"  {i+1}. {angle.get('query', 'N/A')}")