"""On-disk storage for uploaded PDFs.

Files are stored under ``data/uploads/<document_id>.pdf``. The original name is
kept in the database for display and citations; the on-disk name is an opaque id
so two uploads with the same filename cannot overwrite each other.
"""

from __future__ import annotations

import re
from pathlib import Path

from sourceiq.logging_config import get_logger

logger = get_logger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def sanitise_filename(filename: str, *, max_length: int = 120) -> str:
    """Strip directories and unsafe characters from a user supplied filename."""
    name = Path(filename or "document.pdf").name
    name = _SAFE_NAME.sub("_", name).strip(" ._") or "document.pdf"
    if len(name) > max_length:
        stem, dot, suffix = name.rpartition(".")
        name = f"{stem[: max_length - len(suffix) - 1]}{dot}{suffix}" if dot else name[:max_length]
    return name


class DocumentStorage:
    def __init__(self, upload_dir: str | Path) -> None:
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, content: bytes) -> Path:
        destination = self.upload_dir / f"{document_id}.pdf"
        destination.write_bytes(content)
        logger.info("Saved upload %s (%d bytes)", destination.name, len(content))
        return destination

    def path(self, document_id: str) -> Path | None:
        candidate = self.upload_dir / f"{document_id}.pdf"
        return candidate if candidate.exists() else None

    def delete(self, document_id: str) -> bool:
        path = self.path(document_id)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        return True


__all__ = ["DocumentStorage", "sanitise_filename"]
