import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text_file(file_path: str) -> str:
    """Load content from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(file_path: str) -> str:
    """Load content from a .pdf file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def load_document(file_path: str) -> str:
    """Auto-detect file type and load accordingly."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        print(f"  Loading PDF: {file_path}")
        return load_pdf_file(file_path)
    elif ext == ".txt":
        print(f"  Loading text file: {file_path}")
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")


def split_into_chunks(text: str) -> list:
    """Split large text into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = 500 characters
        chunk_overlap=50,     # 50 character overlap between chunks
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    print(f"  Split into {len(chunks)} chunks.")
    return chunks