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

            # Mask sensitive info before logging
            masked_content = []
            for line in cleaned_content.splitlines():
                if any(key in line for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "SERPAPI_API_KEY"]): # Add any other sensitive keys
                    masked_content.append(f"{line.split('=')[0]} = *****")
                else:
                    masked_content.append(line)
            log.info("Loaded .env content:\n" + "\n".join(masked_content))
            
            # Load environment variables
            from io import StringIO
            load_dotenv(stream=StringIO(cleaned_content))
    else:
        log.warning("❌ .env file not found. Ensure API keys are set as environment variables.")
        # If .env is not found, attempt to load from system environment
        load_dotenv() # Tries to load from system environment variables
except Exception as e:
    log.error(f"Error loading environment variables: {e}")
    sys.exit(1) # Exit if environment variables cannot be loaded

# --- Component Initialization ---
# This part of your code needs to be executed once
# so that the QA chain is ready to process questions.
# We will move it into a function or run it once.
components = {}

def initialize_components():
    global components # Use global to modify the components dictionary outside this function
    log.info("🔄 Initializing components (embeddings, vectorstore, LLM, QA Chain)...")
    try:
        # Initialize Embeddings
        # You might want to specify a different model or device if needed
        # For a general-purpose model, 'sentence-transformers/all-MiniLM-L6-v2' is common
        log.info("Initializing HuggingFace Embeddings...")
        components['embeddings'] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        log.info("✅ Embeddings initialized.")

        # Initialize Vectorstore (Chroma)
        log.info("🔄 Initializing vectorstore...")
        # Placeholder for actual document loading and splitting if not already done
        # For simplicity, assuming your documents are already processed and stored in a 'chroma_db' directory
        # You'll need to replace this with your actual vectorstore loading logic
        # Example: if you have persistent Chroma DB
        persist_directory = "./chroma_db" # Ensure this directory exists and contains your Chroma DB
        # The deprecation warning suggests using from langchain_chroma import Chroma
        # For now, let's stick to langchain_community as in your original code
        components['vectorstore'] = Chroma(
            embedding_function=components['embeddings'],
            persist_directory=persist_directory
        )
        log.info(f"✅ Vectorstore initialized from {persist_directory}.")


        # Initialize LLM (ChatOpenAI)
        # Ensure OPENAI_API_KEY is set in your .env or environment variables
        log.info("🔄 Initializing ChatOpenAI LLM...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        components['llm'] = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)
        log.info("✅ LLM initialized.")

        # Initialize RetrievalQA Chain
        log.info("🔄 Initializing RetrievalQA Chain...")
        components['qa_chain'] = RetrievalQA.from_chain_type(
            llm=components['llm'],
            chain_type="stuff", # Other options: "map_reduce", "refine", "map_rerank"
            retriever=components['vectorstore'].as_retriever(),
            return_source_documents=True
        )
        log.info("✅ RetrievalQA Chain initialized.")

    except Exception as e:
        log.error(f"❌ Initialization failed: {e}")
        log.error(traceback.format_exc())
        log.critical("💥 Critical initialization failure. Exiting.")
        sys.exit(1) # Exit the application on critical failure

# --- Question Processing Function ---
def process_question(question: str) -> dict:
    try:
        # Ensure components are initialized before processing
        if not all(key in components for key in ['qa_chain']):
            log.error("QA Chain not initialized. Cannot process question.")
            return {
                "answer": "System not fully initialized. Please check logs for errors.",
                "sources": []
            }

        qa_chain = components['qa_chain']

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

# --- Main execution block for continuous input ---
if __name__ == "__main__":
    print("Welcome to the Document Q&A Bot!")
    print("Type your questions and press Enter. Type 'exit' to quit.")

    # Initialize components once when the script starts
    initialize_components()

    while True:
        user_input = input("\nYour question (or 'exit' to quit): ").strip()

        if user_input.lower() == 'exit':
            print("Exiting Q&A Bot. Goodbye!")
            break

        if not user_input:
            print("Please enter a question.")
            continue

        response = process_question(user_input)

        print(f"\nAI Response: {response['answer']}")
        if response['sources']:
            print("Sources:")
            for src in response['sources']:
                print(f"  - Document: {src['source']}, Page: {src['page']}")
        else:
            print("No specific sources found for this response.")