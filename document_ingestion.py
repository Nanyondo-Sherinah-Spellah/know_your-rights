
import hashlib
from pathlib import Path
from typing import Any

import pymupdf

from chunk_corpus import make_chunks
from rag import collection, embedding_model


MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_PDF_PAGES = 300
BATCH_SIZE = 32


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract selectable text from an uploaded PDF in memory."""

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF files must be 20 MB or smaller.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            if len(pdf) == 0:
                raise ValueError("The PDF contains no pages.")

            if len(pdf) > MAX_PDF_PAGES:
                raise ValueError(
                    f"PDFs may contain at most {MAX_PDF_PAGES} pages."
                )

            page_text = []

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    page_text.append(
                        f"[Page {page_number}]\n{text}"
                    )

            extracted_text = "\n\n".join(page_text).strip()

    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            "The file could not be read as a valid PDF."
        ) from error

    if len(extracted_text) < 100:
        raise ValueError(
            "Very little text could be extracted. "
            "The PDF may be scanned or image-based."
        )

    return extracted_text, len(page_text)


def build_document_chunks(
    pdf_bytes: bytes,
    filename: str,
) -> list[dict[str, Any]]:
    """Extract a PDF and prepare ChromaDB-compatible chunks."""

    text, pages_with_text = extract_pdf_text(pdf_bytes)

    safe_filename = Path(filename).name
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    source_id = f"uploaded_{document_hash[:20]}"

    pieces = make_chunks(text)

    if not pieces:
        raise ValueError("No usable text chunks were created.")

    chunks = []

    for index, piece in enumerate(pieces):
        chunks.append(
            {
                "id": f"{source_id}_chunk_{index + 1}",
                "text": piece,
                "source_document": safe_filename,
                "source_record_id": source_id,
                "chunk_index": index,
                "chunk_count": len(pieces),
                "section_type": "uploaded_document",
                "title": safe_filename,
                "chapter": "Uploaded documents",
                "document_sha256": document_hash,
                "pages_with_text": pages_with_text,
            }
        )

    return chunks


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> dict[str, Any]:
    """Index a PDF without deleting unrelated knowledge-base records."""

    chunks = build_document_chunks(pdf_bytes, filename)

    source_id = chunks[0]["source_record_id"]
    source_filter = {"source_record_id": source_id}

    existing = collection.get(
        where=source_filter,
        include=["metadatas"],
    )

    existing_ids = set(existing["ids"])
    incoming_ids = {chunk["id"] for chunk in chunks}

    if existing_ids == incoming_ids:
        return {
            "status": "already_indexed",
            "filename": Path(filename).name,
            "chunks": len(chunks),
            "message": "This exact PDF is already indexed.",
        }

    # Compute every embedding before writing to ChromaDB.
    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Track IDs that did not exist before this indexing attempt.
    new_ids = incoming_ids - existing_ids

    try:
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            batch_embeddings = embeddings[start : start + BATCH_SIZE]

            collection.upsert(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                metadatas=[
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"id", "text"}
                    }
                    for item in batch
                ],
                embeddings=batch_embeddings.tolist(),
            )

    except Exception:
        # Roll back only newly introduced chunks.
        # Do not delete the original Constitution or unrelated documents.
        if new_ids:
            collection.delete(ids=list(new_ids))

        raise

    return {
        "status": "indexed",
        "filename": Path(filename).name,
        "chunks": len(chunks),
        "message": "The PDF was added to the knowledge base.",
    }
