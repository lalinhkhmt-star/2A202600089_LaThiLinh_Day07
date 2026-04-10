from __future__ import annotations

import re


def preprocess_law_markdown(text: str) -> str:
    """
    Clean markdown-heavy law/course material before chunking.

    This function removes noisy sections (cover/table-of-contents style),
    strips markdown table artifacts, and normalizes spacing so chunkers can
    split on meaningful content boundaries.
    """
    if not text or not text.strip():
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Drop a likely cover/TOC prefix if a first real chapter heading exists.
    chapter_match = re.search(r"^\s*#{2,6}\s*\*{0,2}\s*CHƯƠNG\b", cleaned, flags=re.MULTILINE)
    if chapter_match:
        cleaned = cleaned[chapter_match.start() :]

    lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()

        # Remove markdown table separators and empty table rows.
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            line = line.strip("|").replace("<br>", " ").strip()
            if not line:
                continue

        # Remove decorative bullets that contain no text.
        if line in {"-", "*", "_"}:
            continue

        # Normalize escaped markdown asterisks.
        line = line.replace("\\*", "*")
        # Collapse repeated spaces.
        line = re.sub(r"[ \t]+", " ", line)

        lines.append(line)

    cleaned = "\n".join(lines)
    # Normalize excessive blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class LegalArticleChunker:
    """
    Custom chunking strategy for Vietnamese Law (Thi Hành Án Hình Sự).
    
    Design rationale: Văn bản luật được cấu trúc theo các Điều. 
    Việc tách theo từ khóa "Điều [số]" đảm bảo mỗi chunk là một quy định hoàn chỉnh,
    tránh việc câu hỏi về Điều 5 nhưng kết quả trả về lại lẫn sang Điều 6.
    """

    def __init__(self, max_size: int = 1000, preprocess: bool = True):
        self.max_size = max_size
        self.preprocess = preprocess

    def chunk(self, text: str) -> list[str]:
        if self.preprocess:
            text = preprocess_law_markdown(text)
        if not text:
            return []

        # Bước 1: Tách dựa trên từ khóa "Điều " ở đầu dòng
        # Regex này tìm "Điều" kèm số và dấu chấm (ví dụ: Điều 1. ...)
        articles = re.split(r'\n(?=Điều \d+\.)', text)
        
        final_chunks = []
        for art in articles:
            art = art.strip()
            if not art:
                continue
                
            # Bước 2: Nếu một Điều quá dài (hiếm gặp trong luật này), 
            # ta chia nhỏ tiếp theo các Khoản (1., 2., 3.)
            if len(art) > self.max_size:
                sub_parts = re.split(r'\n(?=\d+\.)', art)
                final_chunks.extend([p.strip() for p in sub_parts if p.strip()])
            else:
                final_chunks.append(art)
                
        # Fallback for documents that are not "Điều"-structured.
        if len(final_chunks) <= 1 and len(text) > self.max_size:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            merged: list[str] = []
            buffer = ""
            for para in paragraphs:
                candidate = f"{buffer}\n\n{para}".strip() if buffer else para
                if len(candidate) <= self.max_size:
                    buffer = candidate
                else:
                    if buffer:
                        merged.append(buffer)
                    if len(para) <= self.max_size:
                        buffer = para
                    else:
                        merged.extend(
                            para[i : i + self.max_size]
                            for i in range(0, len(para), self.max_size)
                        )
                        buffer = ""
            if buffer:
                merged.append(buffer)
            return merged

        return final_chunks