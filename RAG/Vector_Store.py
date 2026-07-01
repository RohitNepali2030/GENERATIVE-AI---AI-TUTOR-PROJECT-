import os
import warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

VECTOR_STORE_PATH = "rag/vector_store.pkl"

# Load embedding model — runs locally, no API needed
# print("Loading embedding model (first time may take a moment)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_store(chunks: list) -> dict:
    """Convert text chunks to vectors and store in FAISS index."""
    print("  Building vector store...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    store = {
        "index": index,
        "chunks": chunks,
        "embeddings": embeddings
    }

    # Save to disk
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(store, f)

    print(f"  Vector store saved with {len(chunks)} chunks.")
    return store


def load_vector_store() -> dict:
    """Load existing vector store from disk."""
    if not os.path.exists(VECTOR_STORE_PATH):
        return None
    with open(VECTOR_STORE_PATH, "rb") as f:
        return pickle.load(f)


def search_vector_store(store: dict, query: str, top_k: int = 3) -> list:
    """Find the most relevant chunks for a query."""
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = store["index"].search(query_embedding, top_k)
    results = [store["chunks"][i] for i in indices[0] if i < len(store["chunks"])]
    return results