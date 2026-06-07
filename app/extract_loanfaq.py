"""
中國信託信貸 FAQ HTML → Markdown 轉換器 (包含 Hash ID 生成 & LLM 自動打標籤)

使用方式：
1. 確保已設定環境變數 OPENAI_API_KEY
2. 執行 python extract_loanfaq.py
"""

from bs4 import BeautifulSoup, NavigableString
from pathlib import Path
import re
import hashlib
import json
import os
from openai import OpenAI
from tqdm import tqdm  # 建議安裝 tqdm 來看進度條: pip install tqdm

# 1. 初始化 OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ===== 設定區 =====
INPUT_HTML = r"C:\Users\ChienYiChen\Desktop\practice\build-moat-live-sessions\knowledge_base_qa_bot\scaffold\hybrid\app\ctbc_loan.html"
OUTPUT_MD  = r"C:\Users\ChienYiChen\Desktop\practice\build-moat-live-sessions\knowledge_base_qa_bot\docs\loan_faq.md"
DEFAULT_PRODUCT = "loan"  # 預設產品，如果 LLM 判斷錯誤可以當作 fallback
# ==================

# 定義 LLM 打標籤的 Prompt (針對信貸微調)
TAGGING_PROMPT = """
你是一個專業的資料庫標籤分類員。
請閱讀提供的【銀行 FAQ 內容】，並嚴格輸出一個 JSON 物件作為 Metadata。

⚠️ 必須包含以下三個欄位：
1. "product": 固定為 "loan"。
2. "source_type": 固定為 "faq"。
3. "tags": 根據內容萃取出 2-3 個精準的核心關鍵字標籤 (請用繁體中文，並以字串陣列格式呈現)。
4. "version": 固定為 "v1"。

輸出範例：
{
  "product": "loan",
  "source_type": "faq",
  "tags": ["信貸", "利率", "還款方式"],
  "version": "v1"
}
"""

def generate_chunk_id(prefix: str, content: str) -> str:
    """生成帶有前綴的 Hash ID，取前 8 碼"""
    hash_str = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    return f"{prefix}_{hash_str}"

def clean_text(text: str) -> str:
    """清理多餘空白、換行、&nbsp;"""
    text = text.replace("\xa0", " ")
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
            for i, li in enumerate(child.find_all("li", recursive=False), 1):
                text = convert_inline(li)
                if text:
                    lines.append(f"{i}. {text}")
        elif child.name == "ul":
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
                chunk_id = generate_chunk_id("loan_faq", question + answer)
                faqs.append({"id": chunk_id, "q": question, "a": answer})

    return faqs

def generate_metadata_with_llm(question: str, answer: str) -> dict:
    """呼叫 LLM 針對單一 FAQ 生成 Metadata"""
    user_prompt = f"【問題】: {question}\n【答案】: {answer}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TAGGING_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.3
        )
        
        result_str = response.choices[0].message.content
        return json.loads(result_str)
        
    except Exception as e:
        print(f"  ⚠️ 打標籤失敗 ({question[:10]}...): {e}")
        return {
            "product": DEFAULT_PRODUCT,
            "source_type": "faq",
            "tags": []
        }

def save_as_markdown(faqs: list[dict], output_path: str, title: str = "中國信託信貸 FAQ"):
    """將 FAQ 列表存成 Markdown 格式，並嵌入 JSON Metadata"""
    lines = [
        f"# {title}\n",
        f"> 共 {len(faqs)} 筆問答\n",
        "---\n"
    ]

    print("🤖 正在呼叫 LLM 進行自動打標籤...")
    
    for i, faq in enumerate(tqdm(faqs), 1):
        lines.append(f"## [{faq['id']}] Q{i}: {faq['q']}\n")
        
        # 🌟 呼叫 LLM 生成這題的 Metadata
        # meta_dict = generate_metadata_with_llm(faq['q'], faq['a'])
        
        # # 🌟 將 Metadata 以 HTML 隱藏註解的方式寫入下一行
        # lines.append(f"\n")

        # 🌟 呼叫 LLM 生成這題的 Metadata
        meta_dict = generate_metadata_with_llm(faq['q'], faq['a'])
        
        # 🌟 將 Metadata 以 HTML 隱藏註解的方式寫入下一行
        lines.append(f"<!-- {json.dumps(meta_dict, ensure_ascii=False)} -->\n")
        
        if faq['a']:
            lines.append(f"{faq['a']}\n")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ 已成功存成包含 Metadata 的檔案：{output_path}（共 {len(faqs)} 筆）")

if __name__ == "__main__":
    html_file = Path(INPUT_HTML)

    if not html_file.exists():
        print(f"❌ 找不到檔案：{INPUT_HTML}")
    else:
        faqs = parse_ctbc_faq(INPUT_HTML)
        print(f"找到 {len(faqs)} 筆問答")
        
        # 執行並儲存
        if faqs:
            save_as_markdown(faqs, OUTPUT_MD)