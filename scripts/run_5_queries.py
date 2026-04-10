from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import SAMPLE_FILES, demo_llm, load_documents_from_files
from src.agent import KnowledgeBaseAgent
from src.embeddings import _mock_embed
from src.store import EmbeddingStore


QUERIES = [
    "Pháp luật về thi hành án hình sự được hiểu như thế nào?",
    "Nguyên tắc nhân đạo được thể hiện ra sao trong quá trình thi hành án hình sự?",
    "Hình phạt có vai trò giáo dục, cải tạo người phạm tội như thế nào?",
    "Pháp luật thi hành án hình sự có những nhiệm vụ gì?",
    "Khi quyền lợi hợp pháp bị xâm phạm, người bị kết án cần xử lý ra sao?",
]


def main() -> None:
    docs = load_documents_from_files(SAMPLE_FILES)
    store = EmbeddingStore(collection_name="manual_query_run", embedding_fn=_mock_embed)
    store.add_documents(docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    print("=== RUN 5 QUERIES (from code) ===")
    print(f"Loaded docs/chunks: {store.get_collection_size()}")
    print()

    for idx, q in enumerate(QUERIES, start=1):
        top = store.search(q, top_k=1)
        top_preview = top[0]["content"][:220].replace("\n", " ") if top else ""
        top_score = top[0]["score"] if top else 0.0
        answer = agent.answer(q, top_k=3)

        print(f"#{idx} Query: {q}")
        print(f"Top1 score: {top_score:.4f}")
        print(f"Top1 preview: {top_preview}")
        print(f"Agent answer: {answer}")
        print("-" * 90)


if __name__ == "__main__":
    main()
