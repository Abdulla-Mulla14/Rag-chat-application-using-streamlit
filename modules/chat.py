# chat.py
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def get_vector_store(qdrant_url="http://localhost:6333", collection="learning_vectors"):
    client = QdrantClient(url=qdrant_url)
    if collection not in [c.name for c in client.get_collections().collections]:
        raise ValueError("PDF not indexed yet.")
    # Embeddings model
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return QdrantVectorStore.from_existing_collection(
        url=qdrant_url,
        collection_name=collection,
        embedding=embedder,
    )

def query_pdf_chat(user_query: str, cfg: dict) -> str:
    # 1) Retrieval
    vector_db = get_vector_store()
    results = vector_db.similarity_search(query=user_query)

    # 2) Build context
    context = "\n\n".join(
        f"Page {r.metadata['page_label']}: {r.page_content}"
        for r in results
    )
    system = f"You are a helpful assistant. Answer only from context below:\n\n{context}"

    # 3) Initialize LLM with user settings
    chat = ChatOpenAI(
    model_name=cfg["model_name"],        
    temperature=cfg["temperature"],
    max_tokens=cfg["max_tokens"],
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)

    # 4) Generate answer
    resp = chat.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_query)
    ])
    return resp.content
