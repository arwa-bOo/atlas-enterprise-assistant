"""Corpus ingestion and indexing. Owner: Person 1."""

import re
import shutil
import unicodedata
from pathlib import Path

import chromadb
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.embeddings import embed_documents
from src.schemas import Chunk
from src.settings import (
    AR_RATIO_ARABIC_MIN,
    AR_RATIO_ENGLISH_MAX,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCS_DIR,
    INDEX_DIR,
    SUPPORTED_EXTENSIONS,
)

ARABIC_LETTERS = re.compile(r"[\u0600-\u06FF]")
LATIN_LETTERS = re.compile(r"[A-Za-z]")
DIACRITICS_AND_TATWEEL = re.compile(r"[\u064B-\u0652\u0640]")


def clean_text(text: str) -> str:
    """Normalize extracted text without changing its meaning.

    Args:
        text: Raw extracted text.
    Returns:
        Text with unified Unicode forms, no diacritics/tatweel, tidy whitespace.
    """
    text = unicodedata.normalize("NFKC", text)  # fixes PDF presentation forms
    text = DIACRITICS_AND_TATWEEL.sub("", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # markdown headings
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_docx(path: Path) -> str:
    """Read paragraphs AND tables from a DOCX, in document order."""
    doc = Document(path)
    parts: list[str] = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            parts.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def load_document(path: Path) -> str:
    """Extract clean text from a .md, .pdf or .docx file.

    Args:
        path: File to read.
    Returns:
        Cleaned text (Arabic order preserved).
    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {path.name}")
    if ext == ".md":
        raw = path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        raw = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        raw = _read_docx(path)
    return clean_text(raw)


def detect_language(text: str) -> str:
    """Classify text as "en", "ar" or "bilingual" by its share of Arabic letters.

    Args:
        text: Document text.
    Returns:
        One of "en", "ar", "bilingual".
    """
    arabic = len(ARABIC_LETTERS.findall(text))
    latin = len(LATIN_LETTERS.findall(text))
    ratio = arabic / max(arabic + latin, 1)
    if ratio < AR_RATIO_ENGLISH_MAX:
        return "en"
    if ratio > AR_RATIO_ARABIC_MIN:
        return "ar"
    return "bilingual"


def split_into_chunks(text: str, domain: str, filename: str) -> list[Chunk]:
    """Split one document into overlapping chunks tagged with metadata.

    Args:
        text: Cleaned document text.
        domain: "hr", "it" or "finance".
        filename: Source filename (kept exactly, Arabic included).
    Returns:
        Chunks carrying domain, source_filename, language and a unique chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # paragraphs first, then lines, then sentences (Latin and Arabic), then words
        separators=["\n\n", "\n", ". ", "؟ ", "۔ ", " ", ""],
    )
    language = detect_language(text)  # one label per document
    return [
        Chunk(
            chunk_id=f"{filename}::{i}",
            text=piece,
            domain=domain,
            source_filename=filename,
            language=language,
        )
        for i, piece in enumerate(splitter.split_text(text))
    ]


def build_index(rebuild: bool = False) -> None:
    """Load all docs, embed them and persist the vector index to INDEX_DIR.

    Args:
        rebuild: Wipe and rebuild even if an index already exists.
    """
    if INDEX_DIR.exists() and not rebuild:
        print("Index already exists. Use scripts/reset.py to rebuild.")
        return
    shutil.rmtree(INDEX_DIR, ignore_errors=True)

    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            text = load_document(path)
            chunks += split_into_chunks(text, path.parent.name, path.name)

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed_documents([c.text for c in chunks]),
        metadatas=[
            {
                "domain": c.domain,
                "source_filename": c.source_filename,
                "language": c.language,
            }
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks into {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
