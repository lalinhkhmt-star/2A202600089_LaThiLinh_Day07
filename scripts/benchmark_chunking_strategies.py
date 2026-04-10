from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.CustomerChunker import LegalArticleChunker, preprocess_law_markdown
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore


@dataclass
class BenchmarkCase:
    query: str
    gold_keywords: list[str]


def _is_relevant(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def build_chunks(text: str, chunk_size: int = 1000) -> dict[str, list[str]]:
    return {
        "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=50).chunk(text),
        "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
        "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        "custom_legal": LegalArticleChunker(max_size=chunk_size, preprocess=True).chunk(text),
    }


def evaluate_strategy(strategy_name: str, chunks: list[str], cases: list[BenchmarkCase]) -> dict:
    docs = [
        Document(
            id=f"{strategy_name}_{idx}",
            content=chunk,
            metadata={"strategy": strategy_name, "chunk_index": idx},
        )
        for idx, chunk in enumerate(chunks)
    ]
    store = EmbeddingStore(collection_name=f"bench_{strategy_name}", embedding_fn=_mock_embed)
    store.add_documents(docs)

    hits = 0
    details: list[dict] = []
    for case in cases:
        results = store.search(case.query, top_k=3)
        hit = any(_is_relevant(r["content"], case.gold_keywords) for r in results)
        if hit:
            hits += 1
        details.append(
            {
                "query": case.query,
                "hit": hit,
                "top1_score": results[0]["score"] if results else 0.0,
                "top1_preview": (results[0]["content"][:140].replace("\n", " ") if results else ""),
            }
        )

    return {
        "strategy": strategy_name,
        "num_chunks": len(chunks),
        "hit_at_3": hits,
        "total": len(cases),
        "details": details,
    }


def main() -> None:
    file_path = Path("data/law.md")
    raw_text = file_path.read_text(encoding="utf-8")
    cleaned_text = preprocess_law_markdown(raw_text)

    benchmark_cases = [
        BenchmarkCase(
            query="Khái niệm pháp luật thi hành án hình sự là gì?",
            gold_keywords=["khái niệm", "thi hành án hình sự", "ngành luật độc lập"],
        ),
        BenchmarkCase(
            query="Nguyên tắc nhân đạo trong thi hành án hình sự thể hiện như thế nào?",
            gold_keywords=["nguyên tắc nhân đạo", "nhân đạo", "thi hành án hình sự"],
        ),
        BenchmarkCase(
            query="Tác dụng giáo dục cải tạo của hình phạt là gì?",
            gold_keywords=["giáo dục", "cải tạo", "hình phạt"],
        ),
        BenchmarkCase(
            query="Nhiệm vụ của pháp luật thi hành án hình sự là gì?",
            gold_keywords=["nhiệm vụ", "pháp luật thi hành án hình sự", "thi hành án"],
        ),
        BenchmarkCase(
            query="Các quyền lợi hợp pháp bị xâm phạm thì người bị kết án giải quyết thế nào?",
            gold_keywords=["quyền lợi hợp pháp", "xâm phạm", "khiếu nại", "tố cáo", "người bị kết án"],
        ),
    ]

    all_chunks = build_chunks(cleaned_text, chunk_size=1000)
    reports = [
        evaluate_strategy(name, chunks, benchmark_cases)
        for name, chunks in all_chunks.items()
    ]
    reports.sort(key=lambda r: (r["hit_at_3"], -r["num_chunks"]), reverse=True)

    print("=== BENCHMARK CHUNKING STRATEGIES ===")
    print(f"Input: {file_path}")
    print(f"Raw length: {len(raw_text)} | Preprocessed length: {len(cleaned_text)}")
    print()
    print("Summary:")
    for r in reports:
        print(
            f"- {r['strategy']:12s} | chunks={r['num_chunks']:4d} | "
            f"hit@3={r['hit_at_3']}/{r['total']}"
        )

    best = reports[0]
    print()
    print(f"Best strategy (by hit@3): {best['strategy']}")
    print()
    print("Per-query details (best strategy):")
    for i, d in enumerate(best["details"], start=1):
        print(
            f"{i}. hit={d['hit']} | top1_score={d['top1_score']:.3f} | "
            f"query={d['query']}"
        )
        print(f"   top1_preview={d['top1_preview']}")


if __name__ == "__main__":
    main()
