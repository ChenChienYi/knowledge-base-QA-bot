"""
中國信託 FAQ HTML → Markdown 轉換器

支援：
  - <p> 純文字段落
  - <p> 含 <a> 連結
  - <ol> 有序清單（1. 2. 3.）
  - <ul> 無序清單（-）
  - <li> 含 <a> 連結
"""

from bs4 import BeautifulSoup, NavigableString
from pathlib import Path
import re

# ===== 設定區 =====
INPUT_HTML = r"C:\Users\ChienYiChen\Desktop\practice\build-moat-live-sessions\knowledge_base_qa_bot\scaffold\hybrid\app\ctbc_loan.html"       # 你存下來的 HTML 檔案
OUTPUT_MD  = r"C:\Users\ChienYiChen\Desktop\practice\build-moat-live-sessions\knowledge_base_qa_bot\docs\loan_faq.md"    # 輸出的 Markdown 檔案
# ==================


def clean_text(text: str) -> str:
    """清理多餘空白、換行、&nbsp;"""
    text = text.replace("\xa0", " ")  # &nbsp;
    return re.sub(r'\s+', ' ', text).strip()


def convert_inline(node) -> str:
    """將一個節點內的文字和連結轉成 Markdown inline 格式"""
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = clean_text(str(child))
            if text:
                parts.append(text)
        elif child.name == "a":
            href = child.get("href", "")
            text = clean_text(child.get_text())
            parts.append(f"[{text}]({href})")
        elif child.name == "u":
            text = clean_text(child.get_text())
            parts.append(f"__{text}__")
        else:
            text = clean_text(child.get_text())
            if text:
                parts.append(text)
    return "".join(parts).strip()


def convert_content_to_markdown(content_div) -> str:
    """將答案 HTML 轉成 Markdown，支援 p、ol、ul、li"""
    lines = []

    for child in content_div.children:
        if isinstance(child, NavigableString):
            text = clean_text(str(child))
            if text:
                lines.append(text)

        elif child.name == "p":
            text = convert_inline(child)
            if text:
                lines.append(text)

        elif child.name == "ol":
            # 有序清單 → 1. 2. 3.
            for i, li in enumerate(child.find_all("li", recursive=False), 1):
                text = convert_inline(li)
                if text:
                    lines.append(f"{i}. {text}")

        elif child.name == "ul":
            # 無序清單 → -
            for li in child.find_all("li", recursive=False):
                text = convert_inline(li)
                if text:
                    lines.append(f"- {text}")

    return "\n".join(lines)


def parse_ctbc_faq(html_path: str) -> list[dict]:
    """解析中信 FAQ 的特定 HTML 結構"""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    faqs = []

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


def save_as_markdown(faqs: list[dict], output_path: str, title: str = "中國信託 FAQ"):
    """將 FAQ 列表存成 Markdown 格式"""
    lines = [
        f"# {title}\n",
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
            print(f"A:\n{faq['a'][:200]}")
            print()

        if faqs:
            save_as_markdown(faqs, OUTPUT_MD, title="中國信託信貸 FAQ")