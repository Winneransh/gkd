"""
query_analyzer.py

Generalized legal document query analyzer that breaks down complex queries 
into components and identifies interdependencies based on document type.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentQueryAnalyzer:
    """
    Analyzes user queries for different types of legal documents and breaks them
    down into components with interdependencies.
    """
    
    def __init__(self, google_api_key: str):
        """
        Initialize the query analyzer.
        
        Args:
            google_api_key: Google API key for Gemini
        """
        self.google_api_key = google_api_key
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=4096
        )
        
        # Create document-specific prompt templates
        self.query_analysis_prompt = self._create_query_analysis_prompt()
    
    def _create_query_analysis_prompt(self) -> PromptTemplate:
        """
        Create a comprehensive system prompt for legal document query analysis.
        
        Returns:
            PromptTemplate for query analysis
        """
        
        prompt_template = """You are an expert legal document query analysis specialist. Your job is to break down complex user queries about legal documents into individual components and identify potential interdependencies between these components.

**DOCUMENT TYPE CONTEXT:** {document_type}

**ANALYSIS FRAMEWORK BY DOCUMENT TYPE:**

**1. HEALTH INSURANCE POLICY:**
- Medical Components: procedures, treatments, conditions, medications
- Demographics: age, gender, family status, pre-existing conditions
- Geographic: coverage areas, network providers, location restrictions
- Temporal: policy duration, waiting periods, effective dates, renewal dates
- Financial: premiums, deductibles, co-pays, coverage limits

**2. OFFER LETTER:**
- Position Components: job title, department, level, responsibilities
- Compensation: salary, bonuses, equity, benefits package
- Temporal: start date, probation period, contract duration
- Location: work location, remote work options, relocation
- Legal: terms and conditions, at-will employment, non-compete

**3. DIVORCE PAPERS:**
- Asset Components: property division, financial accounts, investments
- Custody: child custody, visitation rights, decision-making authority
- Support: alimony, child support, spousal maintenance
- Temporal: separation date, filing date, final decree timeline
- Legal: grounds for divorce, jurisdiction, legal representation

**4. PRENUPTIAL AGREEMENT:**
- Asset Components: separate property, marital property definitions
- Financial: income, debts, inheritance, business ownership
- Legal: enforceability conditions, modification terms, termination
- Temporal: execution date, marriage date, review periods
- Geographic: jurisdiction, applicable state laws

**5. WILL/TESTAMENT:**
- Beneficiary Components: heirs, beneficiaries, charitable organizations
- Asset Components: real estate, personal property, financial accounts
- Legal: executor appointment, guardianship, trust provisions
- Temporal: execution date, revision dates, probate timeline
- Conditions: contingent bequests, residuary clauses, special provisions

**6. LOAN AGREEMENT:**
- Financial Components: principal, interest rate, fees, payment amounts
- Temporal: loan term, payment schedule, maturity date, grace periods
- Collateral: secured assets, guarantees, insurance requirements
- Legal: default conditions, acceleration clauses, governing law
- Parties: borrower qualifications, lender requirements, guarantors

**7. RENTAL/LEASE AGREEMENT:**
- Property Components: address, type, condition, included amenities
- Financial: rent amount, deposits, fees, utilities, escalations
- Temporal: lease term, renewal options, notice periods, move-in/out dates
- Legal: tenant rights, landlord obligations, default conditions
- Usage: permitted use, occupancy limits, pet policies, modifications

**8. COMPANY MERGER/RESOLUTION:**
- Corporate Components: entities involved, ownership structure, governance
- Financial: valuation, consideration, payment terms, escrow
- Legal: regulatory approvals, compliance requirements, representations
- Temporal: transaction timeline, closing conditions, effective dates
- Operational: integration plans, employee matters, contract assignments

**9. OFFICE POLICY:**
- Employee Components: covered personnel, roles, departments
- Behavioral: conduct standards, performance expectations, compliance
- Procedural: workflows, approval processes, reporting mechanisms
- Temporal: effective dates, review periods, policy updates
- Legal: enforcement, disciplinary actions, grievance procedures

**10. NON-DISCLOSURE AGREEMENT (NDA):**
- Information Components: confidential information definition, scope
- Parties: disclosing party, receiving party, third parties
- Temporal: effective period, term duration, survival clauses
- Legal: obligations, restrictions, remedies, governing law
- Practical: permitted uses, return/destruction requirements

**QUERY ANALYSIS INSTRUCTIONS:**

1. **Component Identification**: Break the query into distinct components based on the document type
2. **Keyword Extraction**: Identify relevant keywords for each component
3. **Interdependency Analysis**: Determine how components might interact or depend on each other
4. **Single Query Generation**: Create focused queries for individual components
5. **Hybrid Query Generation**: Create combined queries for interdependent scenarios

**EXAMPLE ANALYSES:**

**Example 1 - Health Insurance:**
User Query: "47-year-old female, ACL reconstruction surgery, Mumbai, 4-month-old policy"
Document Type: Health Insurance Policy

Single Queries:
- Q1 (Medical): "ACL reconstruction surgery coverage"
- Q2 (Demographics): "47-year-old female eligibility"
- Q3 (Geographic): "Mumbai coverage area"
- Q4 (Temporal): "4-month policy waiting period"

Hybrid Queries:
- H1 (Medical + Demographics): "ACL reconstruction coverage for 47-year-old female"
- H2 (Medical + Temporal): "ACL reconstruction with 4-month policy"

**Example 2 - Offer Letter:**
User Query: "Software Engineer position, $85k salary, remote work, 3-month probation"
Document Type: Offer Letter

Single Queries:
- Q1 (Position): "Software Engineer role requirements"
- Q2 (Compensation): "$85,000 salary package details"
- Q3 (Location): "Remote work arrangements"
- Q4 (Temporal): "3-month probation period terms"

Hybrid Queries:
- H1 (Position + Compensation): "Software Engineer salary $85k compensation"
- H2 (Location + Position): "Remote Software Engineer position terms"

**Example 3 - Rental Agreement:**
User Query: "2-bedroom apartment, $1200 monthly rent, 1-year lease, pet policy"
Document Type: Rental/Lease Agreement

Single Queries:
- Q1 (Property): "2-bedroom apartment specifications"
- Q2 (Financial): "$1200 monthly rent payment terms"
- Q3 (Temporal): "1-year lease duration"
- Q4 (Usage): "Pet policy and restrictions"

Hybrid Queries:
- H1 (Property + Financial): "2-bedroom apartment $1200 rent terms"
- H2 (Usage + Financial): "Pet policy additional fees rent"

**OUTPUT FORMAT:**
Provide your analysis in the following JSON structure:

{{
  "single_queries": [
    {{
      "id": "Q1",
      "component": "component_type",
      "query": "focused query text",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ],
  "hybrid_queries": [
    {{
      "id": "H1",
      "dependencies": ["Q1", "Q2"],
      "query": "combined query text",
      "reason": "explanation of interdependency",
      "keywords": ["combined", "keywords", "list"]
    }}
  ]
}}

**USER QUERY TO ANALYZE:**
Document Type: {document_type}
User Query: "{user_query}"

**ANALYSIS:**"""

        return PromptTemplate(
            input_variables=["document_type", "user_query"],
            template=prompt_template
        )
    
    def analyze_query(self, user_query: str, document_type: str) -> Dict[str, Any]:
        """
        Analyze user query and break it down into components based on document type.
        
        Args:
            user_query: The user's question about the document
            document_type: Type of legal document from classifier
            
        Returns:
            Dictionary containing single and hybrid queries
        """
        try:
            # Create the prompt
            formatted_prompt = self.query_analysis_prompt.format(
                document_type=document_type,
                user_query=user_query
            )
            
            # Get analysis from Gemini
            response = self.llm.invoke(formatted_prompt)
            analysis_text = response.content
            
            # Parse the JSON response
            parsed_result = self._parse_analysis_response(analysis_text)
            
            # Add metadata
            parsed_result.update({
                'document_type': document_type,
                'original_query': user_query,
                'analyzed_at': datetime.now().isoformat(),
                'total_single_queries': len(parsed_result.get('single_queries', [])),
                'total_hybrid_queries': len(parsed_result.get('hybrid_queries', []))
            })
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {
                'error': str(e),
                'document_type': document_type,
                'original_query': user_query,
                'single_queries': [],
                'hybrid_queries': [],
                'analyzed_at': datetime.now().isoformat()
            }
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the analysis response from Gemini.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed analysis dictionary
        """
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_text = response_text[json_start:json_end]
                parsed_json = json.loads(json_text)
                
                # Validate the structure
                if 'single_queries' not in parsed_json:
                    parsed_json['single_queries'] = []
                if 'hybrid_queries' not in parsed_json:
                    parsed_json['hybrid_queries'] = []
                
                return parsed_json
            else:
                # Fallback parsing if JSON extraction fails
                return self._fallback_parse(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return self._fallback_parse(response_text)
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return {
                'single_queries': [],
                'hybrid_queries': [],
                'parse_error': str(e),
                'raw_response': response_text
            }
    
    def _fallback_parse(self, response_text: str) -> Dict[str, Any]:
        """
        Fallback parsing method when JSON parsing fails.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Basic parsed structure
        """
        return {
            'single_queries': [],
            'hybrid_queries': [],
            'parse_method': 'fallback',
            'raw_response': response_text,
            'note': 'JSON parsing failed, manual review needed'
        }
    
    def get_query_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the query analysis.
        
        Args:
            analysis_result: Result from analyze_query method
            
        Returns:
            Summary dictionary
        """
        single_queries = analysis_result.get('single_queries', [])
        hybrid_queries = analysis_result.get('hybrid_queries', [])
        
        # Extract all components
        components = list(set([q.get('component', 'unknown') for q in single_queries]))
        
        # Extract all keywords
        all_keywords = []
        for q in single_queries + hybrid_queries:
            all_keywords.extend(q.get('keywords', []))
        unique_keywords = list(set(all_keywords))
        
        return {
            'document_type': analysis_result.get('document_type', 'Unknown'),
            'original_query': analysis_result.get('original_query', ''),
            'total_components': len(components),
            'components_identified': components,
            'total_keywords': len(unique_keywords),
            'unique_keywords': unique_keywords,
            'single_queries_count': len(single_queries),
            'hybrid_queries_count': len(hybrid_queries),
            'complexity_score': len(single_queries) + (len(hybrid_queries) * 1.5),
            'has_interdependencies': len(hybrid_queries) > 0
        }

# Example usage
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get API key from environment
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.strip() == "":
        print("Please set your Google API key")
        exit(1)
    
    # Import the classifier
    from document_classifier import LegalDocumentClassifier
    
    # Initialize both components
    classifier = LegalDocumentClassifier(GOOGLE_API_KEY)
    analyzer = LegalDocumentQueryAnalyzer(GOOGLE_API_KEY)
    
    # Example workflow
    pdf_path = "Suryansh_OL.docx (1) (1) (1).pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        # Step 1: Classify document
        print("=== STEP 1: CLASSIFYING DOCUMENT ===")
        classification_result = classifier.process_pdf_and_classify(pdf_path)
        
        if classification_result['success']:
            print(f"Document Type: {classification_result['classification']['document_type']}")
            print(f"Confidence: {classification_result['classification']['confidence']}%")
            
            # Step 2: Analyze user queries using classifier output
            print("\n=== STEP 2: ANALYZING QUERIES ===")
            
            test_queries = [
               "what is the stipend for the position mentioned in the offer letter?"
            ]
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                
                # Pass the complete classifier output to analyzer
                analysis_result = analyzer.analyze_query(query, classification_result)
                
                if 'error' not in analysis_result:
                    print(f"Single Queries: {len(analysis_result['single_queries'])}")
                    for q in analysis_result['single_queries']:
                        print(f"  - {q['query']} ({q['component']})")
                    
                    if analysis_result['hybrid_queries']:
                        print(f"Hybrid Queries: {len(analysis_result['hybrid_queries'])}")
                        for h in analysis_result['hybrid_queries']:
                            print(f"  - {h['query']}")
                else:
                    print(f"Error: {analysis_result['error']}")
        else:
            print(f"Classification failed: {classification_result.get('error', 'Unknown error')}")
    else:
        print(f"PDF file not found: {pdf_path}")
        
        # Demo with mock classification result
        print("\n=== DEMO WITH MOCK DATA ===")
        mock_classification = {
            'success': True,
            'document_name': 'sample_health_policy.pdf',
            'classification': {
                'document_type': 'Health Insurance Policy',
                'confidence': 95,
                'key_indicators': ['health insurance', 'premium', 'coverage'],
                'reasoning': 'Document contains health insurance terminology'
            }
        }
        
        query = "47-year-old female, ACL surgery, 4-month policy"
        analysis_result = analyzer.analyze_query(query, mock_classification)
        
        print(f"Query: {query}")
        print(f"Document Type: {analysis_result['document_type']}")
        print(f"Single Queries: {len(analysis_result['single_queries'])}")
        print(f"Hybrid Queries: {len(analysis_result['hybrid_queries'])}")