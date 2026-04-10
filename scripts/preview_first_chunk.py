from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.CustomerChunker import LegalArticleChunker, preprocess_law_markdown
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print first chunk preview for each chunking strategy."
    )
    parser.add_argument(
        "--file",
        default="data/law.md",
        help="Path to input text file (default: data/law.md)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for FixedSize/Recursive/custom strategies (default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap for FixedSizeChunker (default: 50)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=3,
        help="Max sentences per chunk for SentenceChunker (default: 3)",
    )
    parser.add_argument(
        "--preview-len",
        type=int,
        default=350,
        help="Preview character length to print (default: 350)",
    )
    return parser


def summarize(name: str, chunks: list[str], preview_len: int) -> None:
    count = len(chunks)
    avg_len = (sum(len(c) for c in chunks) / count) if count else 0.0
    first = chunks[0] if chunks else ""
    preview = first[:preview_len].replace("\n", "\\n")

    print(f"===== {name} =====")
    print(f"count={count} | avg_length={avg_len:.2f}")
    print(f"first_chunk_len={len(first)}")
    print(f"first_chunk_preview={preview}")
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    preprocessed_text = preprocess_law_markdown(text)

    strategies = [
        (
            "fixed_size",
            FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap).chunk(preprocessed_text),
        ),
        (
            "by_sentences",
            SentenceChunker(max_sentences_per_chunk=args.max_sentences).chunk(preprocessed_text),
        ),
        ("recursive", RecursiveChunker(chunk_size=args.chunk_size).chunk(preprocessed_text)),
        ("custom_legal", LegalArticleChunker(max_size=args.chunk_size, preprocess=True).chunk(text)),
    ]

    print(f"Input file: {file_path}")
    print(
        f"Config: chunk_size={args.chunk_size}, overlap={args.overlap}, "
        f"max_sentences={args.max_sentences}, preview_len={args.preview_len}"
    )
    print(f"Preprocessed length: {len(preprocessed_text)} (raw: {len(text)})")
    print()

    for name, chunks in strategies:
        summarize(name, chunks, args.preview_len)


if __name__ == "__main__":
    main()
