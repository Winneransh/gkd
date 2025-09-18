"""
document_classifier.py

This module handles PDF text extraction, chunking, ChromaDB storage with HuggingFace embeddings,
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentClassifier:
    """
    Handles PDF processing, ChromaDB storage with HuggingFace embeddings, and document type classification with Gemini.
    """
    
    def __init__(self, google_api_key: str, chroma_persist_directory: str = "./chroma_db", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the document classifier.
        
        Args:
            google_api_key: Google API key for Gemini
            chroma_persist_directory: Directory to persist ChromaDB
            embedding_model: HuggingFace embedding model name
        """
        self.google_api_key = google_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize HuggingFace embeddings
        logger.info(f"Loading HuggingFace embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Gemini LLM for classification
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Updated to latest model
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
        Chunk document text and store in ChromaDB using HuggingFace embeddings.
        
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
                        'embedding_model': self.embedding_model_name,
                        'created_at': datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            # Create or load ChromaDB collection with HuggingFace embeddings
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )
            
            # Add documents to vector store
            vectorstore.add_documents(documents)
            vectorstore.persist()
            
            logger.info(f"Successfully stored {len(documents)} chunks in ChromaDB collection '{collection_name}' using {self.embedding_model_name}")
            
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
    
    def search_similar_documents(self, query: str, collection_name: str = "legal_documents", k: int = 5) -> List[Document]:
        """
        Search for similar document chunks using HuggingFace embeddings.
        
        Args:
            query: Search query
            collection_name: ChromaDB collection name
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        try:
            # Load existing vectorstore
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )
            
            # Perform similarity search
            similar_docs = vectorstore.similarity_search(query, k=k)
            
            logger.info(f"Found {len(similar_docs)} similar documents for query: '{query}'")
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    def rag_classify_with_context(self, text: str, collection_name: str = "legal_documents") -> Dict[str, any]:
        """
        Enhanced classification using RAG - retrieve similar documents for context before classification.
        
        Args:
            text: Document text to classify
            collection_name: ChromaDB collection name
            
        Returns:
            Dictionary containing classification results with RAG context
        """
        try:
            # Get a sample of the text for similarity search
            sample_text = text[:1000] if len(text) > 1000 else text
            
            # Search for similar documents
            similar_docs = self.search_similar_documents(sample_text, collection_name, k=3)
            
            # Create context from similar documents
            context = ""
            if similar_docs:
                context = "\n\n**SIMILAR DOCUMENT EXAMPLES FROM DATABASE:**\n"
                for i, doc in enumerate(similar_docs, 1):
                    context += f"\nExample {i} (from {doc.metadata.get('document_name', 'Unknown')}):\n"
                    context += f"{doc.page_content[:300]}...\n"
            
            # Enhanced prompt with RAG context
            enhanced_prompt = self.classification_prompt.template.replace(
                "**DOCUMENT TO CLASSIFY:**",
                f"{context}\n\n**DOCUMENT TO CLASSIFY:**"
            )
            
            enhanced_template = PromptTemplate(
                input_variables=["document_text"],
                template=enhanced_prompt
            )
            
            # Truncate text if too long
            sample_text = text[:4000] if len(text) > 4000 else text
            
            # Create the enhanced prompt
            formatted_prompt = enhanced_template.format(document_text=sample_text)
            
            # Get classification from Gemini with RAG context
            response = self.llm.invoke(formatted_prompt)
            classification_text = response.content
            
            # Parse the response
            parsed_result = self._parse_classification_response(classification_text)
            parsed_result['rag_context_used'] = len(similar_docs) > 0
            parsed_result['similar_documents_count'] = len(similar_docs)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in RAG classification: {str(e)}")
            # Fallback to regular classification
            return self.classify_document(text)
    
    def process_pdf_and_classify(self, pdf_path: str, collection_name: str = "legal_documents", use_rag: bool = True) -> Dict[str, any]:
        """
        Complete pipeline: extract text, store in ChromaDB with HuggingFace embeddings, and classify document type with Gemini.
        
        Args:
            pdf_path: Path to PDF file
            collection_name: ChromaDB collection name
            use_rag: Whether to use RAG-enhanced classification
            
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
            
            # Chunk and store in ChromaDB with HuggingFace embeddings
            documents, vectorstore = self.chunk_and_store_document(
                document_text, 
                document_name, 
                collection_name
            )
            
            # Classify document type (with or without RAG)
            if use_rag:
                classification = self.rag_classify_with_context(document_text, collection_name)
            else:
                classification = self.classify_document(document_text)
            
            # Compile results
            results = {
                'document_name': document_name,
                'pdf_path': pdf_path,
                'text_length': len(document_text),
                'total_chunks': len(documents),
                'collection_name': collection_name,
                'embedding_model': self.embedding_model_name,
                'classification_method': 'RAG-enhanced' if use_rag else 'Direct',
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
    # Initialize classifier with HuggingFace embeddings
    GOOGLE_API_KEY = "AIzaSyCAm0TLde3cRtzSTyEScq6CQKJofriwVJI"  # Replace with your actual API key
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        print("Please set your Google API key")
        exit(1)
    
    # You can choose different HuggingFace embedding models:
    # "sentence-transformers/all-MiniLM-L6-v2" (default, lightweight)
    # "sentence-transformers/all-mpnet-base-v2" (better quality, larger)
    # "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" (good for Q&A)
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
    
    classifier = LegalDocumentClassifier(
        GOOGLE_API_KEY,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Process a PDF file
    pdf_path = "Suryansh_OL.docx (1) (1) (1).pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        # Process with RAG-enhanced classification
        results = classifier.process_pdf_and_classify(pdf_path, use_rag=True)
        
        if results['success']:
            print(f"\n=== DOCUMENT CLASSIFICATION RESULTS ===")
            print(f"Document: {results['document_name']}")
            print(f"Embedding Model: {results['embedding_model']}")
            print(f"Classification Method: {results['classification_method']}")
            print(f"Type: {results['classification']['document_type']}")
            print(f"Confidence: {results['classification']['confidence']}%")
            print(f"Key Indicators: {', '.join(results['classification']['key_indicators'])}")
            print(f"Reasoning: {results['classification']['reasoning']}")
            print(f"Total Chunks Stored: {results['total_chunks']}")
            
            if results['classification'].get('rag_context_used'):
                print(f"RAG Context: Used {results['classification']['similar_documents_count']} similar documents")
        else:
            print(f"Error processing document: {results.get('error', 'Unknown error')}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("\nExample: Search similar documents")
        similar_docs = classifier.search_similar_documents("insurance policy premium", k=3)
        for i, doc in enumerate(similar_docs, 1):
            print(f"\nSimilar Doc {i}: {doc.metadata.get('document_name', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")