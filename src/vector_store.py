import os

from langchain_community.vectorstores import FAISS
from src.embeddings import create_embeddings


VECTOR_DB_PATH = "vector_db"


def create_vector_store(chunks):

    if not chunks:
        raise ValueError("No document chunks were provided.")

    embeddings = create_embeddings()

    texts = []
    metadatas = []

    for chunk in chunks:

        text = chunk.get("text", "").strip()
        page = chunk.get("page", "Unknown")

        if not text:
            continue

        texts.append(text)

        metadatas.append({
            "page": page,
            "source": chunk.get("source", "Uploaded PDF")
        })

    if not texts:
        raise ValueError("No valid text found in the document.")

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    vector_store.save_local(VECTOR_DB_PATH)

    return vector_store


def search_documents(vector_store, question, k=5):

    if not question.strip():
        return []

    results_with_scores = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    results = []

    for document, score in results_with_scores:

        document.metadata["similarity_score"] = float(score)

        results.append(document)

    return results