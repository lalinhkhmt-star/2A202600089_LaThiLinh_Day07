# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lã Thị LinhLinh
**Nhóm:** B2B2
**Ngày:** 10/04/20262026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> _High cosine similarity nghĩa là hai vector có hướng gần như trùng nhau trong không gian embedding, thể hiện hai đoạn văn bản có ý nghĩa tương tự hoặc liên quan chặt chẽ về mặt ngữ nghĩa._

**Ví dụ HIGH similarity:**

- Sentence A: Người bị kết án phạt tù được hưởng án treo nếu đủ điều kiện theo quy định
- Sentence B: Án treo áp dụng cho người bị phạt tù khi có đủ các điều kiện pháp luật
- Tại sao tương đồng: Cả hai câu đều nói về điều kiện áp dụng án treo, chỉ khác nhau về cấu trúc câu nhưng cùng nội dung.

**Ví dụ LOW similarity:**

- Sentence A:Phạm nhân được tham gia bảo hiểm xã hội tự nguyện trong trại giam
- Sentence B:Công thức nấu phở bò Hà Nội gồm bánh phở, thịt bò, hành lá
- Tại sao khác: Một câu về quyền của phạm nhân trong thi hành án, câu kia về ẩm thực - hoàn toàn không liên quan.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> Cosine similarity chỉ quan tâm đến góc giữa hai vector (hướng) chứ không bị ảnh hưởng bởi độ dài vector, phù hợp với text embeddings vì ý nghĩa của câu phụ thuộc vào hướng trong không gian vector, không phải độ lớn. Euclidean distance lại bị ảnh hưởng bởi độ dài vector nên có thể đánh giá sai hai câu có cùng nghĩa nhưng khác độ dài.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> Công thức:

    bước = kích thước khối - chồng lấp = 500 - 50 = 450
    Số khối = ceil((tổng chiều dài - kích thước khối) / bước) + 1
    Số khối = ceil((10000 - 500) / 450) + 1 = ceil(9500/450) + 1 = ceil(21.11) + 1 = 22 + 1 = 23
    Đáp án: 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> Khi overlap tăng từ 50 lên 100, step giảm từ 450 xuống 400, số chunks tăng từ 23 lên 25 chunks. Overlap nhiều hơn giúp bảo toàn ngữ cảnh ở ranh giới giữa các chunk, đặc biệt quan trọng với văn bản pháp luật vì một điều luật có thể bị cắt ngang giữa chừng, làm mất tính liên tục của quy định.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật Thi Hành Án Hình SỰ

**Tại sao nhóm chọn domain này?**

> _Viết 2-3 câu:_

### Data Inventory

| #   | Tên tài liệu             | Nguồn                            | Số ký tự | Metadata đã gán |
| --- | ------------------------ | -------------------------------- | -------- | --------------- |
| 1   | Luật Thi Hành Án Hình SỰ | https://www.nguyenphuonglaw.com/ | 259332   |                 |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
| --------------- | ---- | ------------- | ------------------------------ |
|                 |      |               |                                |
|                 |      |               |                                |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy                         | Chunk Count | Avg Length | Preserves Context? |
| -------- | -------------------------------- | ----------- | ---------- | ------------------ |
| `data/law.md` | FixedSizeChunker (`fixed_size`)  | 586 | 499.30 | Trung bình (có overlap nhưng dễ cắt giữa điều/khoản) |
| `data/law.md` | SentenceChunker (`by_sentences`) | 452 | 579.64 | Khá (giữ câu tốt, nhưng không bám cấu trúc điều luật) |
| `data/law.md` | RecursiveChunker (`recursive`)   | 783 | 334.50 | 

### Strategy Của Tôi

**Loại:** custom strategy (`LegalArticleChunker`)

**Mô tả cách hoạt động:**

> Strategy tách văn bản theo mẫu `Điều [số].` để mỗi chunk tương ứng một điều luật thay vì một cửa sổ ký tự cố định. Nếu một điều quá dài, strategy tiếp tục tách theo `khoản` dạng đánh số `1.`, `2.`, `3.` để giảm độ dài chunk. Mục tiêu là giữ được tính toàn vẹn ngữ nghĩa theo cấu trúc pháp lý (điều -> khoản) và giảm nhiễu khi truy xuất. Strategy dựa trên dấu hiệu định dạng văn bản luật tiếng Việt (tiêu đề điều và khoản ở đầu dòng).

**Tại sao tôi chọn strategy này cho domain nhóm?**

> Domain luật hình sự có cấu trúc phân cấp khá rõ: chương, mục, điều, khoản, điểm. Việc chunk theo biên của điều/khoản giúp câu trả lời truy xuất sát đúng đơn vị pháp lý được hỏi (ví dụ hỏi Điều 5 thì trả về đúng Điều 5). Cách này thường phù hợp hơn fixed-size khi tài liệu giữ đúng format chuẩn.

**Code snippet (nếu custom):**

```python
class LegalArticleChunker:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size

    def chunk(self, text: str) -> list[str]:
        import re
        articles = re.split(r'\n(?=Điều \d+\.)', text)
        final_chunks = []
        for art in articles:
            art = art.strip()
            if not art:
                continue
            if len(art) > self.max_size:
                sub_parts = re.split(r'\n(?=\d+\.)', art)
                final_chunks.extend([p.strip() for p in sub_parts if p.strip()])
            else:
                final_chunks.append(art)
        return final_chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy      | Chunk Count | Avg Length | Retrieval Quality? |
| -------- | ------------- | ----------- | ---------- | ------------------ |
| `data/law.md` | best baseline (`recursive`) | 783 | 334.50 | Tốt, ổn định trên dữ liệu thực tế |
| `data/law.md` | **của tôi** (`LegalArticleChunker`) | 3 | 87777.67 | Kém trên file hiện tại do regex không match đầy đủ format |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
| ---------- | -------- | --------------------- | --------- | -------- |
| Tôi        | LegalArticleChunker (custom) | 6.5/10 (tạm đo nội bộ) | Bám cấu trúc pháp lý khi format chuẩn | Nhạy với format tài liệu, cần regex robust hơn |
| [Tên]      | RecursiveChunker | [chờ nhóm cập nhật] | Ổn định, tổng quát nhiều loại văn bản | Không tách đúng biên điều luật 100% |
| [Tên]      | SentenceChunker | [chờ nhóm cập nhật] | Dễ hiểu, giữ câu hoàn chỉnh | Có thể trộn nhiều điều vào cùng chunk |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> Với dữ liệu hiện tại của nhóm, `RecursiveChunker` đang cho kết quả cân bằng nhất giữa độ dài chunk và mức giữ ngữ cảnh, nên là lựa chọn an toàn để benchmark chung. Strategy custom theo điều luật có tiềm năng tốt hơn về độ chính xác pháp lý, nhưng cần cải tiến regex để chịu được biến thể format (không xuống dòng, in hoa/thường, đánh số khác chuẩn). Sau khi làm robust phần parse cấu trúc điều/khoản, custom strategy có thể vượt baseline trong domain luật.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> Tôi dùng regex `(?<=[.!?])\s+` để tách câu dựa trên dấu kết câu (`.`, `!`, `?`) và khoảng trắng theo sau. Sau khi split, tôi `strip()` và loại các phần rỗng để tránh sinh chunk rác khi văn bản có nhiều khoảng trắng/xuống dòng liên tiếp. Các câu sau đó được gộp theo batch `max_sentences_per_chunk` để đảm bảo mỗi chunk có số câu ổn định và dễ kiểm soát.

**`RecursiveChunker.chunk` / `_split`** — approach:

> Thuật toán recursive tách theo thứ tự ưu tiên separator: `\n\n` -> `\n` -> `. ` -> `" "` -> fallback cắt cứng. Ở mỗi mức, nếu một đoạn vẫn vượt `chunk_size` thì gọi đệ quy với separator tiếp theo; nếu đoạn đủ nhỏ thì gom vào `current_doc` đến khi chạm ngưỡng rồi flush thành một chunk. Base case gồm: chuỗi rỗng trả `[]`, chuỗi ngắn hơn hoặc bằng `chunk_size` trả `[text]`, và khi hết separator (hoặc separator rỗng) thì cắt theo cửa sổ ký tự cố định.

### EmbeddingStore

**`add_documents` + `search`** — approach:

> `add_documents` tạo embedding cho từng `Document.content`, gán id duy nhất dạng `doc_id_index`, rồi lưu vào ChromaDB (nếu khả dụng) hoặc fallback in-memory list. Ở bản in-memory, mỗi record giữ đủ `id`, `content`, `metadata`, `embedding` để phục vụ retrieval và filter. `search` embed câu query rồi tính cosine similarity (`compute_similarity`) với toàn bộ embeddings, sau đó sort giảm dần theo `score` và lấy `top_k`.

**`search_with_filter` + `delete_document`** — approach:

> `search_with_filter` thực hiện filter trước rồi mới search: với Chroma thì truyền `where=metadata_filter` ngay trong query; với in-memory thì lọc record theo metadata key/value trước, sau đó mới tính similarity trên tập đã lọc. `delete_document` xóa theo `doc_id`: ở Chroma thử xóa theo `where={"doc_id": doc_id}` (và fallback theo `ids=[doc_id]`), còn ở in-memory thì rebuild list, loại mọi record có `id == doc_id` hoặc `metadata.doc_id == doc_id`. Hàm trả về `True/False` tùy có thực sự xóa được chunk nào hay không.

### KnowledgeBaseAgent

**`answer`** — approach:

> Hàm `answer` áp dụng flow RAG tối giản: gọi `store.search(question, top_k)` để lấy các chunk liên quan nhất, rồi nối phần `content` của các chunk thành context bằng `\n\n`. Prompt được dựng theo mẫu: `Context: ... Question: ...`, trong đó context được inject trực tiếp phía trên câu hỏi để LLM trả lời dựa trên tri thức truy xuất. Cuối cùng hàm gọi `llm_fn(prompt)` và trả về chuỗi kết quả.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-9.0.2
rootdir: D:\AI thực chiến\Day_07\Day-07-Lab-Data-Foundations
collecting ... collected 42 items
...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.23s ==============================
```

**Số tests pass:** **42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự<br>đoán | Actual<br>Score | Đúng? |
| ---- | ---------- | ---------- | ---------- | ------------- | ----- |
| 1    | "Người chấp hành án có được liên lạc với gia đình không?" | "Phạm nhân được thăm gặp thân nhân theo quy định như thế nào?" | high | 0.8127 | Đúng |
| 2    | "Điều kiện để được giảm thời hạn chấp hành án là gì?" | "Khi nào người bị kết án có thể được xét giảm án?" | high | 0.8451 | Đúng |
| 3    | "Cơ quan nào giám sát hoạt động thi hành án hình sự?" | "Món bún bò Huế nấu đúng vị cần nguyên liệu gì?" | low | 0.0978 | Đúng |
| 4    | "Người bị kết án có quyền khiếu nại quyết định thi hành án không?" | "Pháp luật quy định thế nào về quyền tố cáo trong thi hành án hình sự?" | high | 0.7684 | Đúng |
| 5    | "Chế độ lao động của phạm nhân trong trại giam được tổ chức ra sao?" | "Thời tiết hôm nay ở Đà Lạt có mưa không?" | low | 0.0832 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> Kết quả bất ngờ nhất là cặp số 4 có điểm cao (trên 0.7684) dù một câu hỏi về khiếu nại và câu còn lại nói về tố cáo; hai ý này không trùng hoàn toàn nhưng vẫn gần nhau vì cùng nhóm quyền pháp lý trong thi hành án. Đồng thời, cặp số 3 và 5 có điểm rất thấp (0.0978 và 0.0832) khi ghép câu pháp lý với câu ngoài domain, cho thấy embeddings tách khá rõ các chủ đề không liên quan. Điều này cho thấy embedding ưu tiên ngữ cảnh và trường nghĩa chung hơn là khớp chính xác từng từ.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query | Gold Answer |
| --- | ----- | ----------- |
| 1   | Pháp luật về thi hành án hình sự được hiểu như thế nào? | Pháp luật thi hành án hình sự là hệ thống quy phạm điều chỉnh quan hệ phát sinh trong quá trình tổ chức thi hành bản án, quyết định hình sự của Tòa án nhằm bảo đảm bản án được thực hiện đúng pháp luật. |
| 2   | Nguyên tắc nhân đạo được thể hiện ra sao trong quá trình thi hành án hình sự? | Nguyên tắc nhân đạo thể hiện ở việc tôn trọng nhân phẩm, bảo đảm quyền con người của người chấp hành án, cấm đối xử tàn bạo và tạo điều kiện cho họ học tập, lao động, chữa bệnh, tái hòa nhập cộng đồng. |
| 3   | Hình phạt có vai trò giáo dục, cải tạo người phạm tội như thế nào? | Hình phạt không chỉ trừng trị mà còn giáo dục ý thức chấp hành pháp luật, cải tạo người phạm tội thông qua quản lý, lao động, học tập để họ sửa chữa sai lầm và trở thành công dân có ích. |
| 4   | Pháp luật thi hành án hình sự có những nhiệm vụ gì? | Nhiệm vụ của pháp luật thi hành án hình sự là bảo đảm thi hành nghiêm chỉnh bản án, bảo vệ pháp chế xã hội chủ nghĩa, quyền và lợi ích hợp pháp của cá nhân/tổ chức, đồng thời góp phần phòng ngừa tội phạm và giáo dục người phạm tội. |
| 5   | Khi quyền lợi hợp pháp bị xâm phạm, người bị kết án cần xử lý ra sao? | Người bị kết án có quyền khiếu nại, tố cáo đến cơ quan hoặc người có thẩm quyền theo thủ tục luật định; cơ quan có trách nhiệm tiếp nhận, giải quyết và bảo vệ quyền lợi hợp pháp bị xâm phạm. |

### Kết Quả Của Tôi

| #   | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
| --- | ----- | ------------------------------- | ----- | --------- | ---------------------- |
| 1   | Pháp luật về thi hành án hình sự được hiểu như thế nào? | Đoạn top-1 nói về tịch thu, sung quỹ nhà nước (không đi thẳng vào định nghĩa pháp luật thi hành án hình sự). | 0.1750 | Không | Demo LLM trả lời dựa trên context truy xuất, nhưng phần preview cho thấy chưa bám đúng trọng tâm câu hỏi định nghĩa. |
| 2   | Nguyên tắc nhân đạo được thể hiện ra sao trong quá trình thi hành án hình sự? | Đoạn top-1 thiên về tài liệu tham khảo/lịch sử, chưa thể hiện rõ nguyên tắc nhân đạo. | 0.0718 | Không | Demo LLM dùng context chứa mục tài liệu và lịch sử nên câu trả lời khó sát ý "nguyên tắc nhân đạo". |
| 3   | Hình phạt có vai trò giáo dục, cải tạo người phạm tội như thế nào? | Đoạn top-1 thuộc Chương 1 (khái niệm pháp luật thi hành án hình sự), có liên quan nền tảng. | 0.2260 | Có | Demo LLM lấy context pháp lý tổng quan; mức phù hợp trung bình với câu hỏi về vai trò giáo dục/cải tạo. |
| 4   | Pháp luật thi hành án hình sự có những nhiệm vụ gì? | Đoạn top-1 thuộc Chương 1 "Khái niệm, Nhiệm vụ và Nguồn", bám sát truy vấn. | 0.1487 | Có | Demo LLM dùng đúng cụm nội dung liên quan nhiệm vụ nên trả lời phù hợp hơn các query còn lại. |
| 5   | Khi quyền lợi hợp pháp bị xâm phạm, người bị kết án cần xử lý ra sao? | Đoạn top-1 lại nói về tịch thu, sung quỹ nhà nước; chưa nêu khiếu nại/tố cáo. | 0.0624 | Không | Demo LLM trả lời từ context chưa đúng chủ đề quyền khiếu nại/tố cáo nên độ phù hợp thấp. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **2 / 5**

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> _Viết 2-3 câu:_

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> _Viết 2-3 câu:_

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> _Viết 2-3 câu:_

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
| --------------------------- | ------- | ---------------- |
| Warm-up                     | Cá nhân | / 5              |
| Document selection          | Nhóm    | / 10             |
| Chunking strategy           | Nhóm    | / 15             |
| My approach                 | Cá nhân | / 10             |
| Similarity predictions      | Cá nhân | / 5              |
| Results                     | Cá nhân | / 10             |
| Core implementation (tests) | Cá nhân | / 30             |
| Demo                        | Nhóm    | / 5              |
| **Tổng**                    |         | **/ 100**        |
