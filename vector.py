from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

CSV_PATH = "realistic_restaurant_reviews.csv"
DB_LOCATION = "./chroma_langchain_db"
COLLECTION_NAME = "restaurant_reviews"
EMBEDDING_MODEL = "mxbai-embed-large"

# Embeddings
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url="http://127.0.0.1:11434"
)

# Créer ou charger la vector DB
if not os.path.exists(DB_LOCATION):
    print("📦 Creating new vector database...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    documents = []
    ids = []

    for i, row in df.iterrows():
        content = f"{row['Title']} {row['Review']}"
        doc = Document(
            page_content=content,
            metadata={
                "rating": row.get("Rating", None),
                "date": row.get("Date", None),
            }
        )
        documents.append(doc)
        ids.append(str(i))

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )
    vector_store.add_documents(documents=documents, ids=ids)

    print("✅ Vector DB created successfully.")
else:
    print("📂 Loading existing vector database...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

# Fonction de recherche intelligente
def search_vector_store(query: str, k: int = 5, score_threshold: float = 0.3):
    results = vector_store.similarity_search_with_score(query, k=k)
    filtered_docs = [doc for doc, score in results if score < score_threshold]
    return filtered_docs

# Pour compatibilité
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
