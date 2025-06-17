# indexing.py
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

def index_pdf(pdf_path: Path, cfg: dict,
                qdrant_url="http://localhost:6333",
                collection_name="learning_vectors"):
    # Delete existing collection if present
    client = QdrantClient(url=qdrant_url)
    if collection_name in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection_name=collection_name)

    # Load and split
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)

    # Embed & store
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    QdrantVectorStore.from_documents(
        documents=chunks,
        url=qdrant_url,
        collection_name=collection_name,
        embedding=embedder,
    )
    return True
