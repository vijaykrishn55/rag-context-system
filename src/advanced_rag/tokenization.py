"""Tokenizer-aware token counting with an offline fallback."""
from __future__ import annotations
import re
from typing import Iterable

_TOKEN_RE = re.compile(r"\S+", re.UNICODE)


class TokenCounter:
    """Uses cl100k_base when tiktoken is installed; fallback is approximate."""
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = None
        self.mode = "regex_approximation"
        try:
            import tiktoken  # type: ignore
            self._encoding = tiktoken.get_encoding(encoding_name)
            self.mode = f"tiktoken:{encoding_name}"
        except Exception:
            pass

    def encode(self, text: str) -> list:
        if self._encoding is not None:
            return self._encoding.encode(text)
        return [m.group(0) for m in _TOKEN_RE.finditer(text)]

    def decode(self, tokens: list) -> str:
        if self._encoding is not None:
            return self._encoding.decode(tokens)
        # Offline fallback uses whitespace-delimited units. It preserves content and
        # order, but may normalize whitespace when a long text must be split.
        return " ".join(str(token) for token in tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def split(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
        """Split text to a token cap, retaining overlap when requested."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return [text] if text.strip() else []
        step = max_tokens - overlap_tokens
        chunks = []
        for start in range(0, len(tokens), step):
            part = tokens[start:start + max_tokens]
            if not part:
                break
            decoded = self.decode(part).strip()
            if decoded:
                chunks.append(decoded)
            if start + max_tokens >= len(tokens):
                break
        return chunks
