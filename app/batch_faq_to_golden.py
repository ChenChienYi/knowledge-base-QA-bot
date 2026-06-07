import os
import json
import re
import random
from openai import OpenAI
from dotenv import load_dotenv  # 👈 引入 dotenv
from tqdm import tqdm  # 用來顯示進度條


# 1. 🌟 必須在初始化 OpenAI 之前先載入 .env 檔案！
load_dotenv()

# 2. 初始化設定 (這時候 os.environ 裡面才會有 .env 的資料)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ===== 路徑設定區 =====
INPUT_MD_FILE = r"C:/Users/ChienYiChen/Desktop/practice/build-moat-live-sessions/knowledge_base_qa_bot/docs/cc_faq.md"
OUTPUT_JSONL = r"C:/Users/ChienYiChen/Desktop/practice/build-moat-live-sessions/knowledge_base_qa_bot/docs/golden_dataset.jsonl"
PRODUCT_NAME = "credit card" 
# ======================

# 2. 定義單題改寫的 System Prompt
SYSTEM_PROMPT = """
你是一個專門測試銀行 RAG 系統的「紅隊提示詞工程師」。
我會提供給你一筆【官方 FAQ 資料】以及指定的【題目類型】。
請你扮演真實消費者，將這個 FAQ 改寫成一句自然的口語提問。

【題目類型說明】：
- 事實題：針對該 FAQ 內容，用日常口語直接詢問具體規定。
- 推論題：設計一個具體的個人生活情境（例如人在國外、剛買電腦等），需要稍微推理才能用該 FAQ 回答。
- 陷阱題：完全無視提供的 FAQ！隨便問一個與該產品毫無關係的問題（例如房貸、股票，甚至是食譜或天氣）。

⚠️ 請嚴格輸出 JSON 格式，且只包含 query 欄位：
{
  "query": "改寫後的口語問題"
}
"""

def parse_markdown_faq(md_path: str) -> list[dict]:
    """讀取 markdown 檔案，自動切分出 Hash ID 與內容"""
    if not os.path.exists(md_path):
        print(f"❌ 找不到檔案：{md_path}")
        return []
        
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    faqs = []
    # 捕捉 ## [ID] Q1: 標題 的結構
    pattern = re.compile(r"##\s+\[(.*?)\]\s+Q\d+:\s+(.*?)\n(.*?)(?=\n##\s+\[|\Z)", re.DOTALL)
    matches = pattern.findall(content)

    for match in matches:
        chunk_id = match[0].strip()
        question = match[1].strip()
        answer = match[2].strip()
        
        if len(answer) > 10:
            faqs.append({
                "chunk_id": chunk_id,
                "official_q": question,
                "official_a": answer
            })
            
    print(f"✅ 成功從 Markdown 萃取出 {len(faqs)} 筆官方 FAQ")
    return faqs

def generate_single_variation(faq: dict, question_type: str, product_name: str) -> str:
    """呼叫 LLM 針對單一 FAQ 進行特定題型的改寫"""
    user_prompt = f"""
    【目標產品】: {product_name}
    【指定的題目類型】: {question_type}
    
    【原始官方 FAQ】:
    標題：{faq['official_q']}
    解答：{faq['official_a']}
    
    請依照【指定的題目類型】產出一個對應的 JSON 物件。
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 逐題改寫相對簡單，使用 4o-mini 速度極快且成本低
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.7 
        )
        
        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)
        return result_json.get("query", "這題改寫失敗了，請問如何處理？")
        
    except Exception as e:
        print(f"\n  ⚠️ 處理題型 {question_type} 時發生錯誤: {e}")
        return "這題改寫失敗了，請問如何處理？"

def main():
    faqs = parse_markdown_faq(INPUT_MD_FILE)
    if not faqs:
        return
    
    # 1. 隨機抽出 50 題，不可重複 (如果總數不到 50 則全取)
    sample_size = min(50, len(faqs))
    sampled_faqs = random.sample(faqs, sample_size)
    print(f"🎲 已隨機抽出 {sample_size} 題 FAQ 準備進行改寫...")
        
    final_golden_dataset = []
    
    # 2. 逐題設定機率並呼叫 LLM
    # 使用 tqdm 顯示進度條，因為 50 題 API 呼叫需要一點時間
    for i, faq in enumerate(tqdm(sampled_faqs, desc="✨ 正在生成黃金測試集"), 1):
        
        # 依據機率 40%, 40%, 20% 決定這題的類型
        # random.choices 回傳的是一個 List，所以要加上 [0] 取出字串
        q_type = random.choices(
            population=["事實題", "推論題", "陷阱題"], 
            weights=[0.4, 0.4, 0.2], 
            k=1
        )[0]
        
        # 呼叫 LLM 進行改寫
        generated_query = generate_single_variation(faq, q_type, PRODUCT_NAME)
        
        # 3. 陷阱題特殊處理：因為陷阱題與 FAQ 無關，所以預期的正確 ID 應該是空的
        if q_type == "陷阱題":
            relevance_judgments = {}
        else:
            relevance_judgments = {"Chunk ID": faq['chunk_id']}
        
        # 組裝成最後的 JSON 格式
        dataset_item = {
            "query": generated_query,
            "relevance_judgments": relevance_judgments,
            "metadata": {
                "product": PRODUCT_NAME,
                "question_type": q_type
            },
            "query_id": f"q_{str(i).zfill(2)}"  # 產生 q_01 到 q_50 的 ID
        }
        
        final_golden_dataset.append(dataset_item)
        
    # 4. 儲存成 JSONL 格式
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for data in final_golden_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print(f"\n🎉 大功告成！成功生成 {len(final_golden_dataset)} 筆測試題！")
    
    # 統計一下各題型的實際產出數量
    type_counts = {"事實題": 0, "推論題": 0, "陷阱題": 0}
    for data in final_golden_dataset:
        type_counts[data["metadata"]["question_type"]] += 1
        
    print("📊 本次生成的題型分佈：")
    for k, v in type_counts.items():
        print(f"  - {k}: {v} 題")
    print(f"📁 檔案已儲存至: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()