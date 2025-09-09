"""
document_classifier.py

This module handles PDF text extraction, chunking, ChromaDB storage with Gemini embeddings,
and document type classification using Gemini API with advanced prompting techniques.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# PDF Processing
import fitz  # PyMuPDF
import pdfplumber

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentClassifier:
    """
    Handles PDF processing, ChromaDB storage, and document type classification.
    """
    
    def __init__(self, google_api_key: str, chroma_persist_directory: str = "./chroma_db"):
        """
        Initialize the document classifier.
        
        Args:
            google_api_key: Google API key for Gemini
            chroma_persist_directory: Directory to persist ChromaDB
        """
        self.google_api_key = google_api_key
        self.chroma_persist_directory = chroma_persist_directory
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Initialize Gemini LLM for classification
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Document classification prompt template
        self.classification_prompt = self._create_classification_prompt()
    
    def _create_classification_prompt(self) -> PromptTemplate:
        """
        Create an advanced system prompt for document type classification using few-shot prompting.
        
        Returns:
            PromptTemplate for document classification
        """
        
        prompt_template = """You are an expert legal document classifier with extensive experience in analyzing various types of legal and business documents. Your task is to accurately identify the type of document based on its content, language patterns, legal terminology, and structural elements.

**DOCUMENT TYPES TO CLASSIFY:**
1. **Health Insurance Policy** - Medical coverage, premiums, deductibles, coverage limits
2. **Offer Letter** - Job offers, salary, benefits, start date, employment terms
3. **Divorce Papers** - Divorce petition, custody, alimony, asset division
4. **Prenuptial Agreement** - Pre-marriage asset protection, financial arrangements
5. **Will/Testament** - Estate planning, beneficiaries, asset distribution after death
6. **Loan Agreement** - Borrowing terms, interest rates, repayment schedule, collateral
7. **Rental/Lease Agreement** - Property rental, lease terms, rent amount, tenant obligations
8. **Company Merger/Resolution** - Corporate mergers, acquisitions, board resolutions
9. **Office Policy** - Workplace rules, employee handbook, company procedures
10. **Non-Disclosure Agreement (NDA)** - Confidentiality, trade secrets, information protection
11. **Other Legal Document** - Any other legal document not fitting above categories

**CLASSIFICATION EXAMPLES:**

**Example 1:**
Document Content: "This Health Insurance Policy provides coverage for medical expenses including hospitalization, surgical procedures, and prescription medications. The annual premium is $2,400 with a $500 deductible. Coverage includes preventive care, emergency services, and specialist consultations..."
Classification: Health Insurance Policy
Confidence: 95%
Key Indicators: "Health Insurance Policy", "premium", "deductible", "medical expenses", "hospitalization"

**Example 2:**
Document Content: "We are pleased to extend this offer of employment for the position of Software Engineer at TechCorp Inc. Your starting salary will be $85,000 annually. Benefits include health insurance, 401k matching, and 15 days paid vacation. Your start date is March 1st, 2024..."
Classification: Offer Letter
Confidence: 98%
Key Indicators: "offer of employment", "starting salary", "benefits", "start date", position title

**Example 3:**
Document Content: "This Lease Agreement is entered into between Landlord John Smith and Tenant Sarah Johnson for the property located at 123 Main Street. Monthly rent is $1,200 due on the first of each month. Lease term is 12 months beginning January 1st, 2024..."
Classification: Rental/Lease Agreement
Confidence: 96%
Key Indicators: "Lease Agreement", "Landlord", "Tenant", "monthly rent", "lease term"

**Example 4:**
Document Content: "LAST WILL AND TESTAMENT of Robert Williams. I hereby revoke all previous wills. I bequeath my entire estate to my children equally. I appoint my brother as executor of this will. In witness whereof, I have signed this will..."
Classification: Will/Testament
Confidence: 99%
Key Indicators: "LAST WILL AND TESTAMENT", "bequeath", "estate", "executor", legal witnessing language

**Example 5:**
Document Content: "LOAN AGREEMENT between First National Bank (Lender) and Michael Davis (Borrower). Principal amount: $50,000. Interest rate: 6.5% annually. Repayment term: 60 months. Monthly payment: $978. Collateral: 2020 Honda Civic..."
Classification: Loan Agreement
Confidence: 97%
Key Indicators: "LOAN AGREEMENT", "Principal amount", "Interest rate", "Repayment term", "Collateral"

**Example 6:**
Document Content: "MUTUAL NON-DISCLOSURE AGREEMENT between ABC Corp and XYZ Inc. Both parties agree to maintain confidentiality of proprietary information. Trade secrets and business information shall not be disclosed to third parties..."
Classification: Non-Disclosure Agreement (NDA)
Confidence: 94%
Key Indicators: "NON-DISCLOSURE AGREEMENT", "confidentiality", "proprietary information", "trade secrets"

**ANALYSIS INSTRUCTIONS:**
1. **Content Analysis**: Examine key terms, legal language, and document structure
2. **Context Clues**: Look for specific terminology unique to each document type
3. **Structural Elements**: Identify document formatting and legal clauses
4. **Parties Involved**: Analyze relationships between parties mentioned
5. **Purpose and Intent**: Determine the primary purpose of the document
6. **Legal Language**: Recognize standard legal phrases and clauses
7. **Financial Terms**: Identify monetary amounts, payment terms, or financial obligations

**CONFIDENCE SCORING:**
- 90-100%: Very clear indicators, unmistakable document type
- 80-89%: Strong indicators with minor ambiguity
- 70-79%: Good indicators but some conflicting elements
- 60-69%: Moderate confidence, some unclear aspects
- Below 60%: Low confidence, multiple document types possible

**OUTPUT FORMAT:**
Document Type: [Exact classification from the 11 types above]
Confidence: [Percentage]%
Key Indicators: [List 3-5 most important phrases/terms that led to this classification]
Reasoning: [Brief explanation of why this classification was chosen]

**DOCUMENT TO CLASSIFY:**
{document_text}

**CLASSIFICATION:**"""

        return PromptTemplate(
            input_variables=["document_text"],
            template=prompt_template
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            full_text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
            
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def chunk_and_store_document(self, text: str, document_name: str, collection_name: str = "legal_documents") -> Tuple[List[Document], Chroma]:
        """
        Chunk document text and store in ChromaDB.
        
        Args:
            text: Full document text
            document_name: Name/identifier for the document
            collection_name: ChromaDB collection name
            
        Returns:
            Tuple of (document_chunks, vectorstore)
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'document_name': document_name,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'source': 'pdf_document',
                        'created_at': datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            # Create or load ChromaDB collection
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )
            
            # Add documents to vector store
            vectorstore.add_documents(documents)
            vectorstore.persist()
            
            logger.info(f"Successfully stored {len(documents)} chunks in ChromaDB collection '{collection_name}'")
            
            return documents, vectorstore
            
        except Exception as e:
            logger.error(f"Error chunking and storing document: {str(e)}")
            raise
    
    def classify_document(self, text: str) -> Dict[str, any]:
        """
        Classify the document type using Gemini API with advanced prompting.
        
        Args:
            text: Full document text or representative sample
            
        Returns:
            Dictionary containing classification results
        """
        try:
            # Truncate text if too long (keep first 4000 characters for classification)
            sample_text = text[:4000] if len(text) > 4000 else text
            
            # Create the prompt
            formatted_prompt = self.classification_prompt.format(document_text=sample_text)
            
            # Get classification from Gemini
            response = self.llm.invoke(formatted_prompt)
            classification_text = response.content
            
            # Parse the response
            parsed_result = self._parse_classification_response(classification_text)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return {
                'document_type': 'Classification Error',
                'confidence': 0,
                'key_indicators': [],
                'reasoning': f'Error during classification: {str(e)}',
                'raw_response': ''
            }
    
    def _parse_classification_response(self, response_text: str) -> Dict[str, any]:
        """
        Parse the classification response from Gemini.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed classification dictionary
        """
        try:
            # Initialize default values
            result = {
                'document_type': 'Unknown',
                'confidence': 0,
                'key_indicators': [],
                'reasoning': '',
                'raw_response': response_text
            }
            
            # Extract document type
            doc_type_match = re.search(r'Document Type:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
            if doc_type_match:
                result['document_type'] = doc_type_match.group(1).strip()
            
            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*(\d+)%', response_text, re.IGNORECASE)
            if confidence_match:
                result['confidence'] = int(confidence_match.group(1))
            
            # Extract key indicators
            indicators_match = re.search(r'Key Indicators:\s*(.+?)(?:\nReasoning:|$)', response_text, re.IGNORECASE | re.DOTALL)
            if indicators_match:
                indicators_text = indicators_match.group(1).strip()
                # Split by common delimiters and clean up
                indicators = [ind.strip().strip('"').strip("'") for ind in re.split(r'[,;]', indicators_text)]
                result['key_indicators'] = [ind for ind in indicators if ind]
            
            # Extract reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+?)$', response_text, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing classification response: {str(e)}")
            return {
                'document_type': 'Parse Error',
                'confidence': 0,
                'key_indicators': [],
                'reasoning': f'Error parsing response: {str(e)}',
                'raw_response': response_text
            }
    
    def process_pdf_and_classify(self, pdf_path: str, collection_name: str = "legal_documents") -> Dict[str, any]:
        """
        Complete pipeline: extract text, store in ChromaDB, and classify document type.
        
        Args:
            pdf_path: Path to PDF file
            collection_name: ChromaDB collection name
            
        Returns:
            Dictionary containing all results
        """
        logger.info(f"Starting processing for PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            document_text = self.extract_text_from_pdf(pdf_path)
            logger.info(f"Extracted {len(document_text)} characters from PDF")
            
            # Get document name from file path
            document_name = os.path.basename(pdf_path)
            
            # Chunk and store in ChromaDB
            documents, vectorstore = self.chunk_and_store_document(
                document_text, 
                document_name, 
                collection_name
            )
            
            # Classify document type
            classification = self.classify_document(document_text)
            
            # Compile results
            results = {
                'document_name': document_name,
                'pdf_path': pdf_path,
                'text_length': len(document_text),
                'total_chunks': len(documents),
                'collection_name': collection_name,
                'classification': classification,
                'processed_at': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info(f"Successfully processed and classified document as: {classification['document_type']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in complete processing pipeline: {str(e)}")
            return {
                'document_name': os.path.basename(pdf_path) if pdf_path else 'Unknown',
                'pdf_path': pdf_path,
                'error': str(e),
                'success': False,
                'processed_at': datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    GOOGLE_API_KEY = ""  # Replace with your actual API key
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        print("Please set your Google API key")
        exit(1)
    
    classifier = LegalDocumentClassifier(GOOGLE_API_KEY)
    
    # Process a PDF file
    pdf_path = "Suryansh_OL.docx (1) (1) (1).pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        results = classifier.process_pdf_and_classify(pdf_path)
        
        if results['success']:
            print(f"\n=== DOCUMENT CLASSIFICATION RESULTS ===")
            print(f"Document: {results['document_name']}")
            print(f"Type: {results['classification']['document_type']}")
            print(f"Confidence: {results['classification']['confidence']}%")
            print(f"Key Indicators: {', '.join(results['classification']['key_indicators'])}")
            print(f"Reasoning: {results['classification']['reasoning']}")
            print(f"Total Chunks Stored: {results['total_chunks']}")
        else:
            print(f"Error processing document: {results.get('error', 'Unknown error')}")
    else:

        print(f"PDF file not found: {pdf_path}")
