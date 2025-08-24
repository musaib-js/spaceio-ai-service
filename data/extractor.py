import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text(file_path: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    """
    Extract text from a document using LangChain loaders and split into chunks.
    
    Args:
        file_path (str): Path to the file (pdf, docx, txt, etc.)
        chunk_size (int): Max characters per chunk
        chunk_overlap (int): Overlap between chunks

    Returns:
        List[str]: List of text chunks
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        loader = UnstructuredFileLoader(file_path)

    docs = loader.load()

    # Split into smaller chunks for embeddings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)

    return [s.page_content for s in splits]


if __name__ == "__main__":
    test_file = "sample.pdf"
    chunks = extract_text(test_file)
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{c[:300]}...")
    print(f"\nTotal chunks: {len(chunks)}")
