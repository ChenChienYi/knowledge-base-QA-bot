"""
FAQ HTML → Markdown 轉換器 (包含 Hash ID 生成 & LLM 自動打標籤)

使用方式：
1. 確保已設定環境變數 OPENAI_API_KEY
2. 執行 python extract_ccfaq.py
"""

from bs4 import BeautifulSoup
from pathlib import Path
import re
import hashlib
import json
import os
from openai import OpenAI
from tqdm import tqdm # 建議安裝 tqdm 來看進度條: pip install tqdm

from dotenv import load_dotenv
load_dotenv()

# 1. 初始化 OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ===== 設定區 =====
INPUT_HTML = r"C:\Users\ChienYiChen\Desktop\practice\build-moat-live-sessions\knowledge_base_qa_bot\scaffold\hybrid\app\ctbc_cc.html"
OUTPUT_MD  = r"C:\Users\ChienYiChen\Desktop\practice\build-moat-live-sessions\knowledge_base_qa_bot\docs\cc_faq.md"
DEFAULT_PRODUCT = "credit card"  # 預設產品，如果 LLM 判斷錯誤可以當作 fallback
# ==================

# 定義 LLM 打標籤的 Prompt
TAGGING_PROMPT = """
你是一個專業的資料庫標籤分類員。
請閱讀提供的【銀行 FAQ 內容】，並嚴格輸出一個 JSON 物件作為 Metadata。

⚠️ 必須包含以下三個欄位：
1. "product": 固定為 "credit card"。
2. "source_type": 固定為 "faq"。
3. "tags": 根據內容萃取出 2-3 個精準的核心關鍵字標籤 (請用繁體中文，並以字串陣列格式呈現)。
4. "version": 固定為 "v1"。

輸出範例：
{
  "product": "credit card",
  "source_type": "faq",
  "tags": ["掛失", "手續費", "海外消費"],
  "version": "v1"
}
"""

def generate_chunk_id(prefix: str, content: str) -> str:
    """生成帶有前綴的 Hash ID，取前 8 碼"""
    hash_str = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    return f"{prefix}_{hash_str}"

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
                href = node.get("href", "")
                text = clean_text(node.get_text())
                parts.append(f"[{text}]({href})")
            else:
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
    blocks = soup.find_all("div", class_=re.compile(r"twrbo-c-toggle--qna"))
 
    for block in blocks:
        title_div = block.find("div", class_="twrbo-c-toggle__title")
        content_div = block.find("div", class_="twrbo-c-toggle__content")
 
        if title_div:
            question = clean_text(title_div.get_text())
            answer = convert_content_to_markdown(content_div) if content_div else ""
 
            if question:
                chunk_id = generate_chunk_id("cc_faq", question + answer)
                faqs.append({"id": chunk_id, "q": question, "a": answer})
 
    return faqs

def generate_metadata_with_llm(question: str, answer: str) -> dict:
    """呼叫 LLM 針對單一 FAQ 生成 Metadata"""
    user_prompt = f"【問題】: {question}\n【答案】: {answer}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 打標籤任務相對簡單，用 4o-mini 足夠且速度快
            messages=[
                {"role": "system", "content": TAGGING_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.3 # 降低隨機性，確保標籤格式穩定
        )
        
        result_str = response.choices[0].message.content
        return json.loads(result_str)
        
    except Exception as e:
        print(f"  ⚠️ 打標籤失敗 ({question[:10]}...): {e}")
        # 如果 LLM 失敗，回傳一個基本的預設值，避免整個程式崩潰
        return {
            "product": DEFAULT_PRODUCT,
            "source_type": "faq",
            "tags": []
        }
 
def save_as_markdown(faqs: list[dict], output_path: str):
    """將 FAQ 列表存成 Markdown 格式，並嵌入 JSON Metadata"""
    lines = [
        "# 中國信託信用卡 FAQ\n",
        f"> 共 {len(faqs)} 筆問答\n",
        "---\n"
    ]
 
    print("🤖 正在呼叫 LLM 進行自動打標籤...")
    
    # 這裡建議引入 tqdm 來顯示進度條，因為打 API 需要一點時間
    for i, faq in enumerate(tqdm(faqs), 1):
        lines.append(f"## [{faq['id']}] Q{i}: {faq['q']}\n")
        
        # 🌟 呼叫 LLM 生成這題的 Metadata
        meta_dict = generate_metadata_with_llm(faq['q'], faq['a'])
        
        # ✨ 新增：補上日期欄位（預設生效日，到期日為 None = 永久有效）
        meta_dict["effective_date"] = "2024-01-01"
        meta_dict["expiry_date"] = None
        
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
        
        # 為了測試，我們先只跑前 5 筆，確認沒問題再跑全部
        test_faqs = faqs
        #print("⚠️ 目前為測試模式，僅處理前 5 筆資料。請確認結果後再移除 [:5] 限制。")
        
        if test_faqs:
            save_as_markdown(test_faqs, OUTPUT_MD)