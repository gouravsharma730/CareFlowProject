"""Load the clinic's policy documents and split them into chunks.

Loaders (one per format, with a fallback):
  1. Markdown (preferred): keeps the headings, so chunks follow sections.
  2. PDF (fallback): used when a document's Markdown file is missing or
     empty. Text is extracted page by page.

Chunking is structure-aware: a document is split at its section headings
(# to ####). A section longer than CHUNK_MAX_CHARS is split between
paragraphs, never in the middle of a paragraph or table.

Every chunk carries metadata: a stable chunk_id, the document it came
from, the section heading, and which loader produced it.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from careflow import config

logger = logging.getLogger(__name__)

# A heading line such as "### 3.2 Orthopaedics (ORTH)".
HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
# In PDF text, headings lose their "#" marks but stay short numbered
# lines, e.g. "1.0 PURPOSE AND SCOPE" or "3.2 Orthopaedics (ORTH)".
PDF_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][^.;,]{2,60}$")


@dataclass
class Chunk:
    """One piece of a policy document, ready to be indexed."""

    chunk_id: str        # e.g. "preauth-matrix#007"
    doc_slug: str        # e.g. "preauth-matrix"
    doc_title: str       # e.g. "Pre-Authorisation Requirement Matrix"
    doc_category: str    # e.g. "insurance"
    section: str         # e.g. "3.2 Orthopaedics (ORTH)"
    text: str            # the chunk's own text
    source_path: str     # file it was read from, relative to the corpus
    loader: str          # "markdown" or "pdf"

    def to_dict(self) -> dict:
        return asdict(self)

    def search_text(self) -> str:
        """Text used for search: title and section help find the chunk."""
        return f"{self.doc_title} | {self.section}\n{self.text}"


# ---- loaders ----

def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported here: only needed for the fallback

    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # PDF extraction can join a heading onto the line before it. Put each
    # "### ..." marker back at the start of its own line.
    text = re.sub(r"[ \t]*(#{2,4} )", r"\n\1", text)
    # Turn short numbered lines back into headings so chunking still
    # follows the document's sections.
    lines = [
        f"### {line.strip()}" if PDF_HEADING.match(line.strip()) else line
        for line in text.split("\n")
    ]
    return "\n".join(lines)


def load_document_text(entry: dict, corpus_dir: Path) -> tuple[str, str, str]:
    """Return (text, loader, relative path) for one document.

    Tries Markdown first and falls back to the PDF.

    Raises:
        FileNotFoundError: if neither file gives any text.
    """
    for loader, key, read in (
        ("markdown", "markdown", _load_markdown),
        ("pdf", "pdf", _load_pdf),
    ):
        relative = entry.get(key)
        if not relative:
            continue
        # Paths in index.json start with "corpus/", so they are
        # relative to the folder that contains the corpus folder.
        path = corpus_dir.parent / relative
        if not path.exists():
            logger.warning("%s file missing for %s: %s", loader, entry["slug"], path)
            continue
        try:
            text = read(path)
        except Exception:
            logger.exception("Could not read %s", path)
            continue
        if text.strip():
            if loader != "markdown":
                logger.warning("Using %s fallback for %s", loader, entry["slug"])
            return text, loader, relative
    raise FileNotFoundError(f"No readable file for document {entry['slug']}")


# ---- chunking ----

def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split text at headings. Returns [(heading, body), ...].

    Headings with no body (title banners) are dropped; the body that
    follows gets the nearest heading above it.
    """
    sections: list[tuple[str, str]] = []
    heading = "Introduction"
    lines: list[str] = []

    def flush():
        body = "\n".join(lines).strip().strip("-").strip()
        if body:
            sections.append((heading, body))

    for line in text.split("\n"):
        match = HEADING.match(line.strip())
        if match:
            flush()
            heading = match.group(2).strip()
            lines = []
        else:
            lines.append(line)
    flush()
    return sections


def _split_long(body: str, max_chars: int) -> list[str]:
    """Group paragraphs into pieces of at most max_chars (where possible)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    entry: dict, text: str, loader: str, source_path: str, max_chars: int
) -> list[Chunk]:
    """Turn one document's text into chunks with metadata."""
    chunks = []
    for heading, body in _split_sections(text):
        for piece in _split_long(body, max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{entry['slug']}#{len(chunks):03d}",
                    doc_slug=entry["slug"],
                    doc_title=entry["title"],
                    doc_category=entry.get("section", ""),
                    section=heading,
                    text=piece,
                    source_path=source_path,
                    loader=loader,
                )
            )
    return chunks


def load_corpus(
    corpus_dir: Path | None = None, max_chars: int | None = None
) -> list[Chunk]:
    """Load every document listed in corpus/index.json and chunk it."""
    corpus_dir = Path(corpus_dir) if corpus_dir else config.CORPUS_DIR
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    entries = json.loads((corpus_dir / "index.json").read_text(encoding="utf-8"))

    chunks: list[Chunk] = []
    for entry in entries:
        text, loader, source_path = load_document_text(entry, corpus_dir)
        chunks.extend(chunk_document(entry, text, loader, source_path, max_chars))
    logger.info("Loaded %d chunks from %d documents", len(chunks), len(entries))
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    all_chunks = load_corpus()
    for chunk in all_chunks:
        print(f"{chunk.chunk_id:26} {len(chunk.text):5} chars  [{chunk.loader}]  {chunk.section}")
    print(f"\nTotal: {len(all_chunks)} chunks")