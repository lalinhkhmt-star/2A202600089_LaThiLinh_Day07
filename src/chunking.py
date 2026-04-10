from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i:i + self.max_sentences_per_chunk])
            chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
            
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        if sep == "":
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        splits = current_text.split(sep)
        final_chunks = []
        current_doc = []
        current_length = 0

        for s in splits:
            if len(s) > self.chunk_size:
                if current_doc:
                    final_chunks.append(sep.join(current_doc))
                    current_doc = []
                    current_length = 0
                
                recursive_splits = self._split(s, next_seps)
                final_chunks.extend(recursive_splits)
            else:
                s_len = len(s) + (len(sep) if current_doc else 0)
                if current_length + s_len > self.chunk_size:
                    final_chunks.append(sep.join(current_doc))
                    current_doc = [s]
                    current_length = len(s)
                else:
                    current_doc.append(s)
                    current_length += s_len

        if current_doc:
            final_chunks.append(sep.join(current_doc))

        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        f_chunks = FixedSizeChunker(chunk_size=chunk_size).chunk(text)
        s_chunks = SentenceChunker().chunk(text)
        r_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)
        
        return {
            "fixed_size": {
                "count": len(f_chunks),
                "avg_length": sum(len(c) for c in f_chunks) / len(f_chunks) if f_chunks else 0,
                "chunks": f_chunks
            },
            "by_sentences": {
                "count": len(s_chunks),
                "avg_length": sum(len(c) for c in s_chunks) / len(s_chunks) if s_chunks else 0,
                "chunks": s_chunks
            },
            "recursive": {
                "count": len(r_chunks),
                "avg_length": sum(len(c) for c in r_chunks) / len(r_chunks) if r_chunks else 0,
                "chunks": r_chunks
            }
        }

# class SectionAwareChunker:
#     """Custom chunking strategy for internal technical documentation.
 
#     Domain: Markdown handbooks, engineering runbooks, support playbooks —
#     any structured document that uses headers (##, ###) to separate topics.
 
#     Design rationale:
#         Documents in this domain organise content into clearly labelled
#         sections.  Each ## header starts a new topic; splitting across that
#         boundary causes a chunk to mix two unrelated topics and hurts
#         retrieval precision.
 
#         Priority order:
#           1. Split on markdown header lines (## / ###).
#           2. If a section is still > max_size chars, split on blank lines
#              (paragraph boundaries).
#           3. If a single paragraph is still > max_size, fall back to
#              FixedSizeChunker to guarantee the size limit is respected.
 
#         This preserves "topic integrity": a retriever receives one coherent
#         section rather than an arbitrary window of characters.
#     """
 
#     def __init__(self, max_size: int = 600, overlap: int = 50) -> None:
#         self.max_size = max_size
#         self.overlap = overlap
#         self._fallback = FixedSizeChunker(chunk_size=max_size, overlap=overlap)
 
#     # ------------------------------------------------------------------
#     def chunk(self, text: str) -> list[str]:
#         if not text.strip():
#             return []
 
#         # Step 1: split wherever a markdown header starts a new line
#         sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
 
#         chunks: list[str] = []
#         for section in sections:
#             section = section.strip()
#             if not section:
#                 continue
 
#             if len(section) <= self.max_size:
#                 chunks.append(section)
#                 continue
 
#             # Step 2: section too long → split on blank lines (paragraphs)
#             paragraphs = [p.strip() for p in re.split(r'\n\s*\n', section)
#                           if p.strip()]
#             buffer = ""
#             for para in paragraphs:
#                 candidate = (buffer + "\n\n" + para).strip() if buffer else para
#                 if len(candidate) <= self.max_size:
#                     buffer = candidate
#                 else:
#                     if buffer:
#                         chunks.append(buffer)
#                     # Step 3: single paragraph still too long → fixed fallback
#                     if len(para) > self.max_size:
#                         chunks.extend(self._fallback.chunk(para))
#                         buffer = ""
#                     else:
#                         buffer = para
#             if buffer:
#                 chunks.append(buffer)
 
#         return [c for c in chunks if c.strip()]
 
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Tự động xác định thư mục gốc của dự án
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent.parent # Từ src/ quay lại root
    
    # Các đường dẫn có thể chứa file luật
    possible_paths = [
        root_dir / "data" / "law.md",
        root_dir / "output" / "law" / "law.md",
        root_dir / "output" / "law.md",
        Path("data/law.md")
    ]
    
    text = ""
    for p in possible_paths:
        if p.exists():
            text = p.read_text(encoding="utf-8")
            print(f"--- Đang chạy so sánh trên file: {p} ---\n")
            break
            
    if not text:
        text = "Đây là văn bản mẫu để test nếu không tìm thấy file luật. Khoản 1. Điểm a. Điều 1. CHƯƠNG I."
        print("--- Không tìm thấy file luật tại các đường dẫn mặc định, đang chạy trên văn bản mẫu ---\n")

    comparator = ChunkingStrategyComparator()
    results = comparator.compare(text, chunk_size=1000)
    
    for strategy, stats in results.items():
        print(f"Strategy: {strategy}")
        print(f"  - Count: {stats['count']}")
        print(f"  - Avg Length: {stats['avg_length']:.2f}")
        print("-" * 30)
