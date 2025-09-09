"""
llm_enhanced_rag.py

LLM-Enhanced RAG System that processes each individual search angle separately
and provides individual answers without any evaluation or confidence scoring.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEnhancedRAG:
    """
    LLM-Enhanced RAG system that processes each search angle individually
    and provides separate answers for each search angle.
    """
    
    def __init__(self, google_api_key: str, chroma_persist_directory: str = "./chroma_db"):
        """
        Initialize the LLM-Enhanced RAG system.
        
        Args:
            google_api_key: Google API key for Gemini
            chroma_persist_directory: Directory where ChromaDB is persisted
        """
        self.google_api_key = google_api_key
        self.chroma_persist_directory = chroma_persist_directory
        
        # Initialize Gemini 2.5 Flash LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=4096
        )
        
        # Initialize embeddings for ChromaDB
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Rate limiting variables
        self.processed_angles_count = 0
        self.minute_start_time = time.time()
        self.max_angles_per_minute = 5
        
        # Create RAG prompt templates
        self.rag_prompt = self._create_rag_prompt()
        self.query_enhancement_prompt = self._create_query_enhancement_prompt()
    
    def _check_rate_limit(self):
        """
        Check and enforce rate limiting for Gemini API calls.
        Only allows 5 search angles to be processed per minute.
        """
        current_time = time.time()
        elapsed_time = current_time - self.minute_start_time
        
        # If a minute has passed, reset the counter
        if elapsed_time >= 60:
            self.processed_angles_count = 0
            self.minute_start_time = current_time
            logger.info("Rate limit counter reset - new minute started")
        
        # If we've reached the limit, wait until the minute is up
        if self.processed_angles_count >= self.max_angles_per_minute:
            time_to_wait = 60 - elapsed_time
            if time_to_wait > 0:
                logger.info(f"Rate limit reached. Waiting {time_to_wait:.1f} seconds...")
                time.sleep(time_to_wait)
                # Reset after waiting
                self.processed_angles_count = 0
                self.minute_start_time = time.time()
    
    def _increment_rate_limit_counter(self):
        """Increment the counter for processed angles."""
        self.processed_angles_count += 1
        logger.info(f"Processed angles this minute: {self.processed_angles_count}/{self.max_angles_per_minute}")
    
    def _create_query_enhancement_prompt(self) -> PromptTemplate:
        """
        Create prompt for enhancing individual search queries using LLM.
        
        Returns:
            PromptTemplate for query enhancement
        """
        
        prompt_template = """You are a legal document search expert. Your job is to enhance a single search query to maximize retrieval from legal document databases.

**DOCUMENT TYPE:** {document_type}
**SEARCH QUERY:** {search_query}

**ENHANCEMENT INSTRUCTIONS:**

1. **Identify Key Terms**: Extract the main concepts from the search query
2. **Add Synonyms**: Include alternative terms and variations
3. **Include Specific Terms**: Add domain-specific terminology
4. **Consider Context**: Think about how this information appears in legal documents

**DOCUMENT-SPECIFIC ENHANCEMENTS:**

**OFFER LETTER:**
- For compensation: Add currency (INR), payment terms (monthly, annual), synonyms (salary, stipend, remuneration)
- For duration: Add time units (months, years), contract terms (term, period, tenure)
- For work: Add location terms (remote, office, hybrid), time terms (hours, schedule)

**HEALTH INSURANCE:**
- For coverage: Add medical terms, procedure names, benefit terminology
- For costs: Add financial terms (premium, deductible, co-pay, amount)
- For eligibility: Add demographic terms (age, gender, conditions)

**OTHER LEGAL DOCUMENTS:**
- Add relevant legal terminology and specific domain terms

**OUTPUT FORMAT:**
Provide exactly 3 enhanced search variations in JSON format:

{{
  "enhanced_queries": [
    "enhanced search query variation 1",
    "enhanced search query variation 2", 
    "enhanced search query variation 3"
  ]
}}

**ENHANCE THE QUERY:**"""

        return PromptTemplate(
            input_variables=["document_type", "search_query"],
            template=prompt_template
        )
    
    def _create_rag_prompt(self) -> PromptTemplate:
        """
        Create RAG prompt for answering individual search angle queries.
        
        Returns:
            PromptTemplate for RAG answering
        """
        
        prompt_template = """You are an expert legal document analyst. Your job is to answer a specific search query about a legal document using the provided context.

**DOCUMENT TYPE:** {document_type}
**SEARCH QUERY:** {search_query}

**INSTRUCTIONS:**

1. **Focus on the Specific Query**: Answer exactly what the search query is asking for
2. **Use Retrieved Context**: Base your answer on the provided document context only
3. **Be Specific and Direct**: Provide concrete information when available
4. **Quote Relevant Text**: Include relevant quotes from the document when helpful
5. **Stay Factual**: Only state what is explicitly mentioned in the context

**ANSWERING GUIDELINES:**

**For Offer Letters:**
- Compensation queries: Provide exact amounts, currency, frequency
- Position queries: State job title, role, department details
- Duration queries: Specify exact timeframes, start dates, contract terms
- Work arrangement queries: Detail location, hours, flexibility terms
- Legal queries: State specific terms, conditions, obligations

**For Health Insurance:**
- Coverage queries: Specify what is/isn't covered, procedures, conditions
- Cost queries: Provide exact amounts for premiums, deductibles, limits
- Eligibility queries: State specific requirements, restrictions, criteria
- Geographic queries: Detail coverage areas, network information
- Temporal queries: Specify waiting periods, effective dates, terms

**For Other Legal Documents:**
- Provide specific terms, conditions, amounts, dates as mentioned in context
- Quote relevant sections when helpful for clarity

**RESPONSE FORMAT:**
Provide a direct, factual answer focusing specifically on what the search query asks for. Do not include confidence levels, evaluations, or additional analysis.

**DOCUMENT CONTEXT:**
{context}

**PROVIDE YOUR ANSWER:**"""

        return PromptTemplate(
            input_variables=["document_type", "search_query", "context"],
            template=prompt_template
        )
    
    def process_all_search_angles(self, search_angles_output: Dict[str, Any], collection_name: str = "legal_documents") -> Dict[str, Any]:
        """
        Process each individual search angle separately and provide individual answers.
        
        Args:
            search_angles_output: Complete output from SearchQueryGenerator.generate_search_angles()
            collection_name: ChromaDB collection name to search in
            
        Returns:
            Dictionary containing individual answers for each search angle
        """
        try:
            # Validate input
            if 'error' in search_angles_output:
                return {
                    'error': 'Search angle generation failed',
                    'generation_error': search_angles_output['error'],
                    'individual_answers': {},
                    'processed_at': datetime.now().isoformat()
                }
            
            # Initialize ChromaDB vector store
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )
            
            document_type = search_angles_output.get('document_type', 'Unknown')
            original_query = search_angles_output.get('original_query', '')
            search_angles = search_angles_output.get('search_angles', {})
            
            results = {
                'document_type': document_type,
                'original_query': original_query,
                'collection_name': collection_name,
                'individual_answers': {},
                'total_search_angles_processed': 0,
                'successful_answers': 0,
                'processed_at': datetime.now().isoformat()
            }
            
            # Process each query and its search angles
            for query_id, query_data in search_angles.items():
                logger.info(f"Processing search angles for query {query_id}")
                
                original_query_text = query_data.get('original_query', '')
                search_angle_list = query_data.get('search_angles', [])
                
                # Initialize storage for this query's angles
                results['individual_answers'][query_id] = {
                    'original_query': original_query_text,
                    'query_type': query_data.get('query_type', 'unknown'),
                    'angle_answers': {},
                    'total_angles': len(search_angle_list),
                    'successful_angles': 0
                }
                
                # Process each individual search angle
                for angle in search_angle_list:
                    # Check rate limit before processing each angle
                    self._check_rate_limit()
                    
                    angle_id = angle.get('angle_id', 'unknown')
                    search_query = angle.get('query', '')
                    
                    logger.info(f"Processing angle {angle_id}: {search_query}")
                    
                    # Get answer for this individual search angle
                    angle_answer = self._get_answer_for_single_angle(
                        search_query=search_query,
                        angle_data=angle,
                        vectorstore=vectorstore,
                        document_type=document_type
                    )
                    
                    # Increment rate limit counter after processing
                    self._increment_rate_limit_counter()
                    
                    # Store the answer
                    results['individual_answers'][query_id]['angle_answers'][angle_id] = angle_answer
                    results['total_search_angles_processed'] += 1
                    
                    if angle_answer.get('answer_success', False):
                        results['individual_answers'][query_id]['successful_angles'] += 1
                        results['successful_answers'] += 1
            
            logger.info(f"Completed processing {results['total_search_angles_processed']} search angles. Success rate: {results['successful_answers']}/{results['total_search_angles_processed']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in LLM-enhanced RAG processing: {str(e)}")
            return {
                'error': str(e),
                'document_type': search_angles_output.get('document_type', 'Unknown'),
                'individual_answers': {},
                'processed_at': datetime.now().isoformat()
            }
    
    def _get_answer_for_single_angle(self, search_query: str, angle_data: Dict[str, Any], 
                                    vectorstore: Chroma, document_type: str) -> Dict[str, Any]:
        """
        Get LLM-enhanced answer for a single search angle.
        
        Args:
            search_query: The search query for this angle
            angle_data: Angle metadata (focus, keywords, etc.)
            vectorstore: ChromaDB vector store instance
            document_type: Type of document
            
        Returns:
            Individual answer result for this search angle
        """
        try:
            # Step 1: Enhance the search query using LLM
            enhanced_queries = self._enhance_single_query(search_query, document_type)
            
            # Step 2: Retrieve documents using original + enhanced queries
            all_queries_to_search = [search_query] + enhanced_queries
            retrieved_docs = []
            
            for query in all_queries_to_search:
                docs = vectorstore.similarity_search(query, k=3)  # Get 3 docs per query
                retrieved_docs.extend(docs)
            
            # Step 3: Deduplicate and prepare context
            unique_docs = self._deduplicate_docs(retrieved_docs)
            
            if not unique_docs:
                return {
                    'angle_id': angle_data.get('angle_id', 'unknown'),
                    'search_query': search_query,
                    'enhanced_queries': enhanced_queries,
                    'answer': 'No relevant information found in the document.',
                    'documents_found': 0,
                    'answer_success': False,
                    'error': 'No documents retrieved'
                }
            
            # Step 4: Get LLM answer using retrieved context
            context = self._prepare_context(unique_docs)
            llm_answer = self._get_llm_answer_for_angle(search_query, context, document_type)
            
            return {
                'angle_id': angle_data.get('angle_id', 'unknown'),
                'search_query': search_query,
                'angle_focus': angle_data.get('focus', 'unknown'),
                'angle_keywords': angle_data.get('keywords', []),
                'enhanced_queries': enhanced_queries,
                'answer': llm_answer,
                'documents_found': len(unique_docs),
                'answer_success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing angle {angle_data.get('angle_id', 'unknown')}: {str(e)}")
            return {
                'angle_id': angle_data.get('angle_id', 'unknown'),
                'search_query': search_query,
                'error': str(e),
                'answer': 'Error occurred while processing this search angle.',
                'answer_success': False
            }
    
    def _enhance_single_query(self, search_query: str, document_type: str) -> List[str]:
        """
        Enhance a single search query using LLM.
        
        Args:
            search_query: Original search query
            document_type: Type of document
            
        Returns:
            List of enhanced search queries
        """
        try:
            # Create enhancement prompt
            formatted_prompt = self.query_enhancement_prompt.format(
                document_type=document_type,
                search_query=search_query
            )
            
            # Get enhanced queries from LLM
            response = self.llm.invoke(formatted_prompt)
            enhancement_text = response.content
            
            # Parse enhanced queries
            enhanced_queries = self._parse_enhanced_queries(enhancement_text)
            
            return enhanced_queries
            
        except Exception as e:
            logger.error(f"Error enhancing query '{search_query}': {str(e)}")
            return []  # Return empty list if enhancement fails
    
    def _parse_enhanced_queries(self, response_text: str) -> List[str]:
        """
        Parse enhanced queries from LLM response.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            List of enhanced query strings
        """
        try:
            # Try to extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_text = response_text[json_start:json_end]
                parsed_json = json.loads(json_text)
                
                enhanced_queries = parsed_json.get('enhanced_queries', [])
                return enhanced_queries if isinstance(enhanced_queries, list) else []
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error parsing enhanced queries: {str(e)}")
            return []
    
    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents.
        
        Args:
            docs: List of retrieved documents
            
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
        
        return unique_docs[:5]  # Return top 5 unique documents
    
    def _prepare_context(self, docs: List[Document]) -> str:
        """
        Prepare document context for LLM.
        
        Args:
            docs: List of documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(docs):
            context_parts.append(f"**Document Section {i+1}:**")
            context_parts.append(doc.page_content)
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _get_llm_answer_for_angle(self, search_query: str, context: str, document_type: str) -> str:
        """
        Get direct answer from LLM for a single search angle.
        
        Args:
            search_query: The search query
            context: Retrieved document context
            document_type: Type of document
            
        Returns:
            Direct answer string
        """
        try:
            # Create RAG prompt
            formatted_prompt = self.rag_prompt.format(
                document_type=document_type,
                search_query=search_query,
                context=context
            )
            
            # Get answer from LLM
            response = self.llm.invoke(formatted_prompt)
            answer = response.content.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error getting LLM answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def get_processing_summary(self, rag_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of the RAG processing results.
        
        Args:
            rag_output: Output from process_all_search_angles method
            
        Returns:
            Summary dictionary
        """
        individual_answers = rag_output.get('individual_answers', {})
        
        total_queries = len(individual_answers)
        total_angles = rag_output.get('total_search_angles_processed', 0)
        successful_angles = rag_output.get('successful_answers', 0)
        
        query_breakdown = {}
        for query_id, query_data in individual_answers.items():
            query_breakdown[query_id] = {
                'total_angles': query_data.get('total_angles', 0),
                'successful_angles': query_data.get('successful_angles', 0),
                'success_rate': (query_data.get('successful_angles', 0) / query_data.get('total_angles', 1)) * 100
            }
        
        return {
            'document_type': rag_output.get('document_type', 'Unknown'),
            'original_query': rag_output.get('original_query', ''),
            'total_queries': total_queries,
            'total_search_angles': total_angles,
            'successful_answers': successful_angles,
            'overall_success_rate': (successful_angles / total_angles * 100) if total_angles > 0 else 0,
            'query_breakdown': query_breakdown,
            'ready_for_evaluation': successful_angles > 0
        }

# Example usage
if __name__ == "__main__":
    # Initialize components
    GOOGLE_API_KEY = "AIzaSyAUFUq_6OHjo1UNWdFxsXZi5AB8HSnt0DU"  # Replace with your actual API key
    

    
    # Import previous components
    import os
    from query_analyzer import LegalDocumentQueryAnalyzer
    from search_query_generator import SearchQueryGenerator
    
    # Initialize components (skipping document classifier)
    analyzer = LegalDocumentQueryAnalyzer(GOOGLE_API_KEY)
    generator = SearchQueryGenerator(GOOGLE_API_KEY)
    rag_system = LLMEnhancedRAG(GOOGLE_API_KEY)
    
    # Check if ChromaDB folder exists
    chroma_db_path = "./chroma_db"
    
    if os.path.exists(chroma_db_path):
        print("=== INDIVIDUAL SEARCH ANGLE PROCESSING (Using Pre-existing ChromaDB) ===")
        
        # Skip Step 1: Document classification - use pre-existing ChromaDB
        print("\n1. USING PRE-EXISTING CHROMADB...")
        print(f"   ChromaDB Path: {chroma_db_path}")
        print(f"   Document Type: Offer Letter (pre-defined)")
        
        # Create mock classification result for query analyzer
        classification_result = {
            'success': True,
            'classification': {
                'document_type': 'offer letter',
                'confidence': 1.0,
                'reasoning': 'Using pre-existing ChromaDB with offer letter documents'
            },
            'document_text': 'Pre-existing document content in ChromaDB',
            'processed_at': datetime.now().isoformat()
        }
        
        print("\n2. ANALYZING USER QUERY...")
        user_query = "What is the monthly stipend amount and duration of internship?"
        query_analysis = analyzer.analyze_query(user_query, classification_result)
        
        if 'error' not in query_analysis:
            print(f"   Single Queries: {len(query_analysis['single_queries'])}")
            print(f"   Hybrid Queries: {len(query_analysis['hybrid_queries'])}")
            
            print("\n3. GENERATING SEARCH ANGLES...")
            search_angles = generator.generate_search_angles(query_analysis)
            
            if 'error' not in search_angles:
                print(f"   Total Search Angles: {search_angles['total_angles_generated']}")
                
                # Step 4A: Process each search angle individually
                print("\n4A. PROCESSING INDIVIDUAL SEARCH ANGLES (Rate Limited: 5 per minute)...")
                rag_results = rag_system.process_all_search_angles(search_angles)
                
                if 'error' not in rag_results:
                    print(f"   Search Angles Processed: {rag_results['total_search_angles_processed']}")
                    print(f"   Successful Answers: {rag_results['successful_answers']}")
                    
                    # Show individual answers
                    print("\n   Individual Search Angle Answers:")
                    for query_id, query_data in rag_results['individual_answers'].items():
                        print(f"\n   Query {query_id}: {query_data['original_query']}")
                        
                        for angle_id, angle_answer in query_data['angle_answers'].items():
                            if angle_answer.get('answer_success', False):
                                print(f"     {angle_id}: {angle_answer['answer'][:100]}...")
                            else:
                                print(f"     {angle_id}: Error - {angle_answer.get('error', 'Unknown')}")
                    
                    # Get summary
                    summary = rag_system.get_processing_summary(rag_results)
                    print(f"\n   Overall Success Rate: {summary['overall_success_rate']:.1f}%")
                    print(f"   Ready for Evaluation: {summary['ready_for_evaluation']}")
                else:
                    print(f"   Error: {rag_results['error']}")
            else:
                print(f"   Error: {search_angles['error']}")
        else:
            print(f"   Error: {query_analysis['error']}")
    else:
        print(f"ChromaDB folder not found: {chroma_db_path}")
        print("\n=== DEMO NOTES ===")
        print("This system now uses pre-existing ChromaDB and skips document classification:")
        print("- Uses pre-existing ChromaDB folder instead of processing new PDFs")
        print("- Document type set to 'offer letter' directly")
        print("- 2 Single Queries + 1 Hybrid = 3 queries")
        print("- 5 search angles per query = 15 total search angles")
        print("- Only 5 search angles processed per minute (Gemini rate limit)")
        print("- Each angle gets its own LLM-enhanced answer")
        print("- System automatically waits when rate limit is reached")
        print("- Ready for Step 4B evaluation of the individual answers")