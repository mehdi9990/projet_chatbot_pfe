from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

CSV_PATH = "realistic_restaurant_reviews.csv"
DB_LOCATION = "./chroma_langchain_db"
COLLECTION_NAME = "restaurant_reviews"

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11434"
)

if not os.path.exists(DB_LOCATION):
    print("📦 Creating MiniVerse DB...")

    df = pd.read_csv(CSV_PATH)
    documents = []
    ids = []

    for i, row in df.iterrows():
        content = f"{row['Title']} {row['Review']}"
        doc = Document(page_content=content)
        documents.append(doc)
        ids.append(str(i))

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )
    vector_store.add_documents(documents=documents, ids=ids)
else:
    print("📂 Loading MiniVerse DB...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

# def search_vector_store(query: str, k: int = 5, score_threshold: float = 0.3):
#     results = vector_store.similarity_search_with_score(query, k=k)
#     return [doc for doc, score in results if score < score_threshold]

def search_vector_store(query: str, k: int = 5):
    results = vector_store.similarity_search(query, k=k)
    return results
