import os
import sys
import logging
import traceback
import codecs
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Load environment variables with explicit path and BOM handling
try:
    # Get the directory of this script
    BASE_DIR = Path(__file__).resolve().parent
    DOTENV_PATH = BASE_DIR / '.env'
    
    log.info(f"Looking for .env file at: {DOTENV_PATH}")
    
    if DOTENV_PATH.exists():
        log.info("✅ .env file found")
        
        # Read and clean the .env file content
        with open(DOTENV_PATH, 'rb') as f:
            raw_content = f.read()
            # Remove UTF-8 BOM if present
            if raw_content.startswith(codecs.BOM_UTF8):
                cleaned_content = raw_content[len(codecs.BOM_UTF8):].decode('utf-8')
            else:
                cleaned_content = raw_content.decode('utf-8')
            
            # Mask the key for security
            masked_content = cleaned_content.replace('OPENAI_API_KEY=', 'OPENAI_API_KEY=******')
            log.info(f"🔒 .env content: {masked_content}")
            
        # Load environment variables from cleaned content
        with open(DOTENV_PATH, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        load_dotenv(DOTENV_PATH, encoding='utf-8')
    else:
        log.error("❌ .env file not found")
        sys.exit(1)
        
except Exception as e:
    log.error(f"❌ Error loading .env file: {str(e)}")
    log.error(traceback.format_exc())
    sys.exit(1)

# Validate API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    # Mask the key for security
    masked_key = OPENAI_API_KEY[:6] + '*' * (len(OPENAI_API_KEY) - 10) + OPENAI_API_KEY[-4:]
    log.info(f"✅ OPENAI_API_KEY found (length: {len(OPENAI_API_KEY)}, masked: {masked_key})")
else:
    log.error("❌ OPENAI_API_KEY not found in environment variables")
    sys.exit(1)

# Initialize components with detailed error handling
def initialize_components():
    """Initialize all system components with detailed error handling"""
    components = {}
    
    try:
        # 1. Initialize embeddings
        log.info("🔄 Initializing embeddings...")
        components['embeddings'] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./embedding_cache"
        )
        log.info("✅ Embeddings initialized")
        
        # 2. Initialize vectorstore
        log.info("🔄 Initializing vectorstore...")
        components['vectorstore'] = Chroma(
            persist_directory="./chroma_db",
            embedding_function=components['embeddings']
        )
        log.info("✅ Vectorstore initialized")
        
        # 3. Initialize OpenAI LLM
        log.info("🔄 Initializing OpenAI LLM...")
        components['llm'] = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_retries=3,
            openai_api_key=OPENAI_API_KEY
        )
        log.info("✅ OpenAI LLM initialized")
        
        # 4. Initialize QA chain
        log.info("🔄 Initializing QA chain...")
        components['qa_chain'] = RetrievalQA.from_chain_type(
            llm=components['llm'],
            chain_type="stuff",
            retriever=components['vectorstore'].as_retriever(),
            return_source_documents=True
        )
        log.info("✅ QA chain initialized")
        
        return components
        
    except Exception as e:
        log.error(f"❌ Initialization failed: {str(e)}")
        log.error(traceback.format_exc())
        return None

# Initialize system components
log.info("🚀 Starting system initialization...")
system_components = initialize_components()

if not system_components:
    log.critical("💥 Critical initialization failure. Exiting.")
    sys.exit(1)

# Get QA chain from components
qa_chain = system_components['qa_chain']

def process_question(question):
    """Process a question and return formatted response"""
    try:
        log.info(f"📩 Processing question: {question}")
        result = qa_chain.invoke({"query": question})
        
        if "result" not in result or not result["result"]:
            return {
                "answer": "I couldn't find an answer to your question in the documents.",
                "sources": []
            }
            
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page = doc.metadata.get('page', 'N/A')
                sources.append({"source": source, "page": page})
                
        return {
            "answer": result["result"],
            "sources": sources
        }
    except Exception as e:
        log.error(f"Error processing question: {str(e)}")
        log.error(traceback.format_exc())
        return {
            "answer": f"Error processing your question: {str(e)}",
            "sources": []
        }

# Test initialization when run directly
if __name__ == "__main__":
    print("Running self-test...")
    test_question = "What do you know about priortization?"
    print(f"Testing with question: '{test_question}'")
    response = process_question(test_question)
    print(f"Response: {response['answer']}")
    if response['sources']:
        print("Sources:")
        for src in response['sources']:
            print(f"- {src['source']} (page {src['page']})")