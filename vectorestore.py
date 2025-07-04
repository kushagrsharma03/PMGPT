import os
import sys
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ✅ Check internet connection
try:
    requests.get("https://www.google.com", timeout=5)
    print("🌐 Internet connection verified")
except:
    print("❌ No internet connection")
    sys.exit(1)

# ✅ Folder with your PDFs
DOCUMENTS_PATH = "./documents"
CHROMA_PATH = "./chroma_db"
EMBEDDING_CACHE = "./embedding_cache"

# Create required folders
os.makedirs(DOCUMENTS_PATH, exist_ok=True)
os.makedirs(EMBEDDING_CACHE, exist_ok=True)

# ✅ Create embedding function with local cache
try:
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=EMBEDDING_CACHE
    )
    print("✅ Embedding function created")
except Exception as e:
    print(f"❌ Error creating embeddings: {e}")
    sys.exit(1)

# ✅ Function to load & split PDFs
def load_and_split_pdfs(folder_path):
    all_chunks = []
    valid_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    if not valid_files:
        print(f"⚠️ No PDF files found in {folder_path}")
        return all_chunks
        
    for filename in valid_files:
        try:
            pdf_path = os.path.join(folder_path, filename)
            print(f"📄 Processing: {filename}")
            
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"✅ Created {len(chunks)} chunks from {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    return all_chunks

# ✅ Load and split
print("⏳ Loading and splitting PDFs...")
documents = load_and_split_pdfs(DOCUMENTS_PATH)

if not documents:
    print("❌ No documents processed. Exiting.")
    sys.exit(1)
 
# ✅ Save to Chroma vector store
try:
    print("💾 Saving to Chroma vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    # Chroma automatically persists after 0.4.x
    print(f"🎉 Success! Embeddings saved to '{CHROMA_PATH}'")
    print(f"📊 Total documents processed: {len(documents)}")
except Exception as e:
    print(f"❌ Error creating vector store: {e}")
    sys.exit(1)