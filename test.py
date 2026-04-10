"""
Exercise 3.1 — Thiết Kế Retrieval Strategy
===========================================
Sử dụng các class trong src/chunking.py (đã có sẵn) cùng với
SectionAwareChunker (custom strategy, thêm vào cuối chunking.py).

Domain: Internal Technical Documentation
"""
from pathlib import Path
from src.chunking import ChunkingStrategyComparator, SectionAwareChunker

DATA_DIR = Path("D:/AI_thực chiến/Day_07/Day-07-Lab-Data-Foundations/data")
DOCS = {
    "rag_system_design":         (DATA_DIR / "rag_system_design.md").read_text(),

}

# ── Step 1: Baseline ─────────────────────────────────────────────────────────
def step1_baseline():
    print("=" * 68)
    print("STEP 1 — Baseline: ChunkingStrategyComparator().compare()")
    print("=" * 68)
    comparator = ChunkingStrategyComparator()
    results = {}
    for doc_name, text in DOCS.items():
        print(f"\n📄 {doc_name}  ({len(text)} chars)")
        r = comparator.compare(text, chunk_size=300)
        results[doc_name] = r
        for strategy, s in r.items():
            print(f"  [{strategy:10s}]  chunks={s['num_chunks']:3d} | "
                  f"avg={s['avg_chunk_len']:6.1f} | "
                  f"min={s['min_chunk_len']:4d} | max={s['max_chunk_len']:4d}")
    return results

# ── Step 2: Custom Strategy ───────────────────────────────────────────────────
def step2_custom():
    print("\n" + "=" * 68)
    print("STEP 2 — Custom Strategy: SectionAwareChunker(max_size=600)")
    print("=" * 68)
    chunker = SectionAwareChunker(max_size=600, overlap=50)
    results = {}
    for doc_name, text in DOCS.items():
        chunks  = chunker.chunk(text)
        lengths = [len(c) for c in chunks]
        stats = {
            "num_chunks":    len(chunks),
            "avg_chunk_len": round(sum(lengths)/len(lengths), 1) if lengths else 0,
            "min_chunk_len": min(lengths) if lengths else 0,
            "max_chunk_len": max(lengths) if lengths else 0,
            "chunks":        chunks,
        }
        results[doc_name] = stats
        print(f"\n📄 {doc_name}  ({len(text)} chars)")
        print(f"  [section_aware]  chunks={stats['num_chunks']:3d} | "
              f"avg={stats['avg_chunk_len']:6.1f} | "
              f"min={stats['min_chunk_len']:4d} | max={stats['max_chunk_len']:4d}")
        print("  Sample chunks (first 3):")
        for i, c in enumerate(chunks[:3]):
            print(f"    [{i}] {c[:100].replace(chr(10),' ')!r}")
    return results

# ── Step 3: Comparison ────────────────────────────────────────────────────────
def step3_compare(baseline, custom):
    print("\n" + "=" * 68)
    print("STEP 3 — So sánh Custom vs Baseline (rag_system_design)")
    print("=" * 68)
    doc, base, cust = "rag_system_design", baseline["rag_system_design"], custom["rag_system_design"]

    print(f"\n{'Strategy':<18} {'#Chunks':>8} {'Avg':>8} {'Min':>6} {'Max':>6}")
    print("-" * 50)
    for name in ("fixed", "sentence", "recursive"):
        s = base[name]
        print(f"{name:<18} {s['num_chunks']:>8} {s['avg_chunk_len']:>8.1f} "
              f"{s['min_chunk_len']:>6} {s['max_chunk_len']:>6}")
    print(f"{'section_aware':<18} {cust['num_chunks']:>8} {cust['avg_chunk_len']:>8.1f} "
          f"{cust['min_chunk_len']:>6} {cust['max_chunk_len']:>6}")

    def score(chunk, query):
        q = set(query.lower().split())
        return len(q & set(chunk.lower().split())) / len(q)

    queries = [
        "what metadata should be stored for each document",
        "what should happen when retrieval results are weak",
        "how to measure retrieval quality with benchmark queries",
    ]
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        for name in ("fixed", "sentence", "recursive"):
            best = max(base[name]["chunks"], key=lambda c: score(c, query))
            print(f"  [{name:12s}] score={score(best,query):.2f}  "
                  f"→ {best[:85].replace(chr(10),' ')!r}")
        best_c = max(cust["chunks"], key=lambda c: score(c, query))
        print(f"  [section_aware] score={score(best_c,query):.2f}  "
              f"→ {best_c[:85].replace(chr(10),' ')!r}")

    print("""
── Nhận xét ──────────────────────────────────────────────────────────────

Fixed   : cắt đều 300 chars, không quan tâm ranh giới logic → chunk
           thường chứa nửa ý của 2 topic khác nhau.

Sentence: 3 câu/chunk rất nhỏ → câu trả lời đúng trong 1 chunk nhưng
           thiếu surrounding context để LLM trả lời đầy đủ.

Recursive: tôn trọng \\n\\n nhưng bỏ qua headers → đôi khi merge 2 section
           khác nhau thành 1 chunk khi mỗi section ngắn.

SectionAware: mỗi chunk = 1 ## section = 1 topic duy nhất → tối ưu cho
           tài liệu markdown có cấu trúc rõ ràng.  Chunk lớn hơn một
           chút nhưng self-contained và dễ trace về nguồn gốc.
""")

if __name__ == "__main__":
    baseline = step1_baseline()
    custom   = step2_custom()
    step3_compare(baseline, custom)
    print("=" * 68)
    print("✅  Exercise 3.1 hoàn thành.")
    print("=" * 68)