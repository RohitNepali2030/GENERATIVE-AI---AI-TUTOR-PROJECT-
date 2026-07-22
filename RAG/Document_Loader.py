import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Clean extracted text for better RAG performance."""
    # Remove page numbers (e.g. "Page 1 of 10", "- 1 -")
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'-\s*\d+\s*-', '', text)

    # Remove repeated special characters (common in PDFs)
    text = re.sub(r'[-_=*]{3,}', '', text)

    # Remove extra whitespace and spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix broken words (common in scanned PDFs)
    # e.g. "Rec onn aiss ance" → not fixable without ML but remove double spaces
    text = re.sub(r' {2,}', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def load_text_file(file_path: str) -> str:
    """Load content from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(file_path: str) -> str:
    """Load content from a .pdf file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def load_document(file_path: str) -> str:
    """Auto-detect file type, load and clean the content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        print(f"  Loading PDF: {os.path.basename(file_path)}")
        text = load_pdf_file(file_path)
    elif ext == ".txt":
        print(f"  Loading text file: {os.path.basename(file_path)}")
        text = load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")

    # Clean the text before returning
    print("  Cleaning text...")
    text = clean_text(text)
    print(f"  Cleaned — {len(text)} characters ready for processing.")

    return text


def split_into_chunks(text: str) -> list:
    """Split large text into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,    # increased overlap for better context
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_text(text)

    # Remove empty or very short chunks
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

    print(f"  Split into {len(chunks)} chunks.")
    return chunks