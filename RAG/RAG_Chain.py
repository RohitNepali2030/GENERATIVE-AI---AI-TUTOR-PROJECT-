from RAG.Document_Loader import load_document, split_into_chunks
from RAG.Vector_Store import build_vector_store, load_vector_store, search_vector_store
from Config import invoke_with_fallback
from langchain_core.messages import HumanMessage

# Global store — loaded once per session
_store = None


def ingest_document(file_path: str):
    """Load a document, chunk it, and build the vector store."""
    global _store
    print("\nIngesting document...")
    text = load_document(file_path)
    chunks = split_into_chunks(text)
    _store = build_vector_store(chunks)
    print("Document ready! You can now ask questions about it.\n")


def load_existing_store():
    """Load previously saved vector store if it exists."""
    global _store
    _store = load_vector_store()
    return _store is not None


def ask_document(question: str) -> str:
    """Retrieve relevant chunks and answer the question using Gemini."""
    global _store

    if _store is None:
        return "No document loaded. Please upload a document first (option 17)."

    # Step 1: Find relevant chunks
    relevant_chunks = search_vector_store(_store, question, top_k=3)

    if not relevant_chunks:
        return "Could not find relevant information in the document."

    # Step 2: Build context from chunks
    context = "\n\n".join(relevant_chunks)

    # Step 3: Ask Gemini with the context
    prompt = f"""You are an academic tutor. Answer the student's question using 
ONLY the information provided in the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context from document:
\"\"\"{context}\"\"\"

Student's question: {question}

Answer clearly and helpfully:"""

    return invoke_with_fallback([HumanMessage(content=prompt)])