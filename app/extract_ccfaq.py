"""
FAQ HTML → Markdown 轉換器

使用方式：
1. 用瀏覽器開啟 FAQ 頁面，Ctrl+S 儲存成「完整網頁」
2. 把存下來的 .html 檔案路徑填入 INPUT_HTML
3. 執行 python faq_parser.py
4. 輸出會存在 OUTPUT_MD 指定的路徑
"""

from bs4 import BeautifulSoup
from pathlib import Path
import re

# ===== 設定區 =====
INPUT_HTML = r"C:\Users\ChienYiChen\Desktop\practice\build-moat-live-sessions\knowledge_base_qa_bot\scaffold\hybrid\app\ctbc_cc.html"       # 你存下來的 HTML 檔案
OUTPUT_MD  = r"C:\Users\ChienYiChen\Desktop\practice\build-moat-live-sessions\knowledge_base_qa_bot\docs\cc_faq.md"    # 輸出的 Markdown 檔案
# ==================


def clean_text(text: str) -> str:
    """清理多餘空白與換行"""
    return re.sub(r'\s+', ' ', text).strip()
 
 
def convert_content_to_markdown(content_div) -> str:
    """將答案 HTML 轉成 Markdown，保留連結格式"""
    lines = []
 
    for p in content_div.find_all("p"):
        parts = []
        for node in p.children:
            if node.name == "a":
                # <a href="...">文字</a> → [文字](網址)
                href = node.get("href", "")
                text = clean_text(node.get_text())
                parts.append(f"[{text}]({href})")
            else:
                # 純文字節點
                text = clean_text(node if isinstance(node, str) else node.get_text())
                if text:
                    parts.append(text)
 
        line = "".join(parts).strip()
        if line:
            lines.append(line)
 
    return "\n".join(lines)
 
 
def parse_ctbc_faq(html_path: str) -> list[dict]:
    """解析中信 FAQ 的特定 HTML 結構"""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
 
    faqs = []
 
    # 找所有 FAQ 區塊（twrbo-c-toggle--qna）
    blocks = soup.find_all("div", class_=re.compile(r"twrbo-c-toggle--qna"))
 
    for block in blocks:
        title_div = block.find("div", class_="twrbo-c-toggle__title")
        content_div = block.find("div", class_="twrbo-c-toggle__content")
 
        if title_div:
            question = clean_text(title_div.get_text())
            answer = convert_content_to_markdown(content_div) if content_div else ""
 
            if question:
                faqs.append({"q": question, "a": answer})
 
    return faqs
 
 
def save_as_markdown(faqs: list[dict], output_path: str):
    """將 FAQ 列表存成 Markdown 格式"""
    lines = [
        "# 中國信託信用卡 FAQ\n",
        f"> 共 {len(faqs)} 筆問答\n",
        "---\n"
    ]
 
    for i, faq in enumerate(faqs, 1):
        lines.append(f"## Q{i}: {faq['q']}\n")
        if faq['a']:
            lines.append(f"{faq['a']}\n")
        lines.append("")
 
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
 
    print(f"✅ 已存成：{output_path}（共 {len(faqs)} 筆）")
 
 
if __name__ == "__main__":
    html_file = Path(INPUT_HTML)
 
    if not html_file.exists():
        print(f"❌ 找不到檔案：{INPUT_HTML}")
    else:
        faqs = parse_ctbc_faq(INPUT_HTML)
        print(f"找到 {len(faqs)} 筆問答")
 
        for faq in faqs[:3]:
            print(f"\nQ: {faq['q']}")
            print(f"A: {faq['a'][:100]}")
 
        if faqs:
            save_as_markdown(faqs, OUTPUT_MD)