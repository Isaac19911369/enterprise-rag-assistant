"""Markdown ingestion: front matter (title, owner, access groups) and section-level chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    text: str
    groups: tuple[str, ...] = field(default=("all",))

    @property
    def citation(self) -> str:
        return f"{self.title} › {self.section}"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _split_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    parts, step = [], max_words - overlap
    for start in range(0, len(words), step):
        parts.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return parts


def parse_document(path: Path, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    """One chunk per ``##`` section (long sections split with overlap).

    Every chunk repeats the document title and section heading, so it stays
    self-explanatory when shown to the model or the user out of context.
    """
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    m = _FRONT_MATTER.match(raw)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        raw = raw[m.end() :]
    doc = path.stem
    title = meta.get("title") or doc.replace("-", " ").title()
    groups = tuple(meta.get("groups") or ["all"])

    chunks: list[Chunk] = []
    sections = re.split(r"^## +", raw, flags=re.M)
    for block in sections[1:]:
        heading, _, body = block.partition("\n")
        heading, body = heading.strip(), body.strip()
        for i, part in enumerate(_split_words(body, max_words, overlap)):
            cid = f"{doc}#{slugify(heading)}" + (f"-{i}" if i else "")
            text = f"{title}. {heading}.\n{part}"
            chunks.append(Chunk(cid, doc, title, heading, text, groups))
    return chunks


def load_corpus(directory: str | Path) -> list[Chunk]:
    files = sorted(Path(directory).glob("**/*.md"))
    if not files:
        raise FileNotFoundError(f"no markdown documents in {directory}")
    return [c for f in files for c in parse_document(f)]
