from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "chat_memory"

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11434"
)

client = QdrantClient(url=QDRANT_URL)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )
    print("✅ Qdrant collection created.")

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

def store_conversation(question, answer, context):
    document = Document(
        page_content=f"Q: {question}\nA: {answer}",
        metadata={"context": context}
    )
    vector_store.add_documents([document])
