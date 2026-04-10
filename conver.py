import pymupdf4llm
import os

# Đảm bảo thư mục data tồn tại
if not os.path.exists('data'):
    os.makedirs('data')

# Đường dẫn file của bạn (Dùng dấu / cho an toàn)
input_pdf = "Data.pdf" 
output_md = "data/document1.md"

try:
    # Chuyển đổi sang markdown
    md_text = pymupdf4llm.to_markdown(input_pdf)
    
    # Ghi file với bảng mã utf-8 để không lỗi tiếng Việt
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_text)
        
    print(f"Thành công! File đã được lưu tại: {output_md}")
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")