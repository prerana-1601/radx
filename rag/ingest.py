# rag/ingest.py

import os
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

# ----------------------------
# 1. Load API Key from .env
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

# ----------------------------
# 2. Define source documents
# ----------------------------
radiopaedia_url = "https://radiopaedia.org/articles/pneumonia?utm_source=chatgpt.com"
who_pdf_url = "https://iris.who.int/bitstream/handle/10665/66956/WHO_V_and_B_01.35.pdf?utm_source=chatgpt.com"
who_pdf_path = "who_pneumonia.pdf"

docs = []

# Load Radiopaedia webpage
print("🔗 Loading Radiopaedia article...")
try:
    loader1 = WebBaseLoader(radiopaedia_url)
    docs.extend(loader1.load())
    print("✅ Radiopaedia article loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Radiopaedia article: {e}")

# Download WHO PDF if not present
if not os.path.exists(who_pdf_path):
    print(f"📥 Downloading WHO PDF from {who_pdf_url} ...")
    try:
        r = requests.get(who_pdf_url)
        r.raise_for_status()
        with open(who_pdf_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded WHO PDF to {who_pdf_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download WHO PDF: {e}")

# Load WHO PDF
print("📄 Loading WHO PDF...")
try:
    loader2 = PyPDFLoader(who_pdf_path)
    docs.extend(loader2.load())
    print(f"✅ WHO PDF loaded successfully")
except Exception as e:
    print(f"❌ Failed to load WHO PDF: {e}")

print(f"📚 Total documents loaded: {len(docs)}")

# ----------------------------
# 3. Split into chunks
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
splits = text_splitter.split_documents(docs)
print(f"✂️ Split into {len(splits)} chunks")

# ----------------------------
# 4. Create embeddings & persist DB
# ----------------------------
persist_directory = "rag/db"

try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    #vectorstore.persist()
    print(f"✅ Ingestion complete! Database saved to {persist_directory}")
except Exception as e:
    print(f"❌ Failed to create embeddings or persist database: {e}")

