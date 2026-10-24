"""Document ingestion: plain text, Markdown, and PDF.

Only the text of a document is kept in memory. Nothing is sent to an
external service. PDF support is optional at runtime: if ``pypdf`` is
not installed, PDF ingestion raises a clear error instead of failing
silently with an empty document.
"""

from __future__ import annotations

from pathlib import Path


def read_text_file(path: str | Path) -> str:
    """Read a ``.txt`` or ``.md`` file as UTF-8 text."""
    return Path(path).read_text(encoding="utf-8")


def read_pdf_file(path: str | Path) -> str:
    """Extract text from a PDF with ``pypdf``."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PDF support needs pypdf. Install it with: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(path: str | Path) -> tuple[str, str]:
    """Load one document, returning ``(source_name, text)``.

    Raises:
        ValueError: For unsupported extensions.
        FileNotFoundError: If ``path`` does not exist (from ``Path``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.name, read_text_file(path)
    if suffix == ".pdf":
        return path.name, read_pdf_file(path)
    raise ValueError(f"Unsupported file type: {suffix!r} (use .txt, .md, or .pdf)")


def load_directory(directory: str | Path) -> list[tuple[str, str]]:
    """Load every supported document in ``directory``, sorted by name."""
    docs = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
            docs.append(load_document(path))
    return docs
