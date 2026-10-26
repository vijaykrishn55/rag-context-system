"""Simple, dependency-light loaders for Markdown, text, HTML, and Python files."""
from __future__ import annotations
from html.parser import HTMLParser
from pathlib import Path
from .schemas import SourceDocument


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip += 1
        elif tag.lower() in {"p", "div", "br", "h1", "h2", "h3", "li", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        elif tag.lower() in {"p", "div", "h1", "h2", "h3", "li", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _read_path(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(text)
        return "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    return text


def load_documents(directory: str | Path, extensions: set[str] | None = None) -> list[SourceDocument]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    allowed = extensions or {".md", ".markdown", ".txt", ".html", ".htm", ".py"}
    docs = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed):
        text = _read_path(path).strip()
        if text:
            docs.append(SourceDocument(
                source=path.name,
                text=text,
                metadata={"path": str(path), "file_type": path.suffix.lower()},
            ))
    if not docs:
        raise ValueError(f"No readable documents found in {root}; supported: {sorted(allowed)}")
    return docs
