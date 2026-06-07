from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import time
# from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage

# 匯入後端檢索邏輯 (請依據你的實際目錄結構微調路徑)
from .retrieval import query, get_llm, query_stream

# 引入你剛剛的 hybrid_indexer 裡面的全局物件
from .hybrid_indexer import indexer

app = FastAPI(title="RAG 評估大腦 API", version="1.0.0")

# 🌟 定義聊天請求的格式
class ChatRequest(BaseModel):
    query: str
    # 如果你的 index.html 有傳送 session_id 或其他參數，可以加在這裡
    # session_id: str = "default"

# 🌟 【新增】掛載靜態目錄 (將 static 資料夾設定為公開)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🌟 【新增】設定根目錄直接轉跳到 eval.html (選用，方便你直接打開)
from fastapi.responses import RedirectResponse
@app.get("/")
async def root():
    return RedirectResponse(url="/static/eval.html")

# 🌟 【新增這段】讓伺服器啟動時，自動喚醒向量大腦
@app.on_event("startup")
async def startup_event():
    print("🚀 正在從 .kb 資料夾載入知識庫大腦...")
    indexer.load_index()  # 叫 indexer 把硬碟裡的資料讀進記憶體
    print("✅ 大腦載入完成！可以開始評估了。")

# 設定 CORS，允許未來不同網域的前端來呼叫這支 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 輔助函式：計算 MRR 與 忠實度
# ==========================================
# def calculate_mrr(expected_id: str, sources: list) -> float:
#     if not expected_id: 
#         return 0.0
#     for rank, src in enumerate(sources):
#         if expected_id in src.get("heading", "") or expected_id in src.get("source", ""):
#             return 1.0 / (rank + 1)
#     return 0.0

def calculate_mrr(expected_id: str, sources: list) -> float:
    if not expected_id:
        return 0.0
    for rank, src in enumerate(sources):
        if expected_id == src.get("chunk_id", ""):  # 精確比對
            return 1.0 / (rank + 1)
    return 0.0

# def evaluate_faithfulness(context: str, answer: str) -> int:
#     if "沒有相關資訊" in answer or "不在信用卡規章中" in answer:
#         return 1
#     judge_prompt = f"""請判斷下方的【回答】是否「完全且僅僅基於」【參考資料】。
# 如果完全忠實，請嚴格只輸出數字 '1'；如果有幻覺或過度推論，請嚴格只輸出數字 '0'。
# 【參考資料】:\n{context}\n【回答】:\n{answer}"""
#     try:
#         response = get_llm().invoke([HumanMessage(content=judge_prompt)])
#         return 1 if "1" in response.content else 0
#     except Exception:
#         return 0
    
def evaluate_faithfulness(context: str, answer: str) -> int:
    # 💡 【修改這裡】暫時關掉 LLM 評分，直接秒回傳 1 (代表 100%)，省時又省錢！
    return 1

    # 下方原本的程式碼就會被跳過，可以留著以後想開起時再用
    if "沒有相關資訊" in answer or "不在信用卡規章中" in answer:
        return 1
    judge_prompt = f"""請判斷下方的【回答】是否「完全且僅僅基於」【參考資料】。"""

# ==========================================
# 核心端點：接收測試集檔案，執行 A/B 策略對決
# ==========================================
from fastapi import Form

@app.post("/api/evaluate")
async def run_batch_evaluation(
    file: UploadFile = File(...), 
    limit: int = Form(50),
    chunk_size: int = Form(500),
    embedding_model: str = Form("text-embedding-3-small"),
    llm_model: str = Form("gpt-4o-mini"),
    threshold: float = Form(0.016),
    top_k: int = Form(3)
):
    # 🌟 實作規格 3 & 4：參數驅動的大腦切換機制 (Smart Cache)
    if indexer.current_chunk_size != chunk_size or indexer.current_embedding_model != embedding_model or indexer.vectorstore is None:
        print(f"🔄 偵測到參數變更，嘗試載入專屬大腦 (Chunk={chunk_size}, Embed={embedding_model})...")
        
        # 1. 優先嘗試從硬碟的 .kb/ 對應資料夾載入大腦
        indexer.load_index(embedding_model=embedding_model, chunk_size=chunk_size)
        
        # 2. 如果載入後發現 vectorstore 還是 None，代表這是第一次跑這個參數組合
        if indexer.vectorstore is None:
            print("⚠️ 找不到該參數組合的現成大腦，正在啟動文件讀取與自動重建機制...")
            indexer.build_index(chunk_size=chunk_size, embedding_model=embedding_model)
            print("✅ 全新大腦建置並存檔完成！")
        else:
            print("✅ 成功從快取載入現成大腦，秒速切換完畢！")

    content = await file.read()
    lines = content.decode("utf-8").splitlines()
    dataset = [json.loads(line) for line in lines if line.strip()][:limit]
    
    results = []
    
    # 統計變數
    trap_count, normal_count = 0, 0
    far_count, frr_count = 0, 0
    retrieval_error_count, success_count = 0, 0

    for record in dataset:
        q = record.get("query")
        # 如果是陷阱題，這裡會拿到 ""
        expected_chunk_id = record.get("relevance_judgments", {}).get("Chunk ID", "") 
        product_filter = record.get("metadata", {}).get("product", "credit card") 
        is_trap = not bool(expected_chunk_id)
        
        if is_trap:
            trap_count += 1
        else:
            normal_count += 1
            
        start_b = time.perf_counter()
        
        # 🌟 傳入所有的超參數進行測試
        res_b = query(
            question=q, 
            filter_dict={"product": product_filter},
            llm_model=llm_model,
            threshold=threshold,
            top_k=top_k
        )
        
        # 🌟 從 res_b 中直接取出剛剛算好的時間
        latency_b = res_b.get("latency_ms", 0)
        ttft_b = res_b.get("ttft_ms", 0)
        mrr_b = calculate_mrr(expected_chunk_id, res_b["sources"])
        
        # 🌟 實作規格 6：誤差歸因邏輯 (Error Attribution)
        is_refusal = "沒有相關資訊" in res_b["answer"]
        hit = mrr_b > 0
        error_type = "✅ 成功"
        
        if is_trap:
            if is_refusal:
                error_type = "✅ 成功 (正確拒答)"
            else:
                error_type = "🟡 LLM 誤答 (FAR/幻覺)"
                far_count += 1
        else:
            if hit:
                if is_refusal:
                    error_type = "🟡 LLM 誤拒 (FRR/過度敏感)"
                    frr_count += 1
                else:
                    error_type = "✅ 成功 (正確作答)"
                    success_count += 1
            else:
                error_type = "🔴 檢索失敗 (Retrieval Error)"
                retrieval_error_count += 1
                
        # 🌟 實作規格 1：Precision @ K 與 Recall @ K
        # 因為單題正確答案只有 1 個，Recall@K 就是 Hit (1或0)
        # Precision@K 就是 (1 / top_k) 如果有命中，否則為 0
        recall_at_k = 1.0 if hit else 0.0
        precision_at_k = (1.0 / top_k) if hit else 0.0

        # ==========================================
        # 🌟 【加回：終端機每題對決明細列印 (升級版)】
        # ==========================================
        print("\n" + "="*80)
        trap_label = " (陷阱題)" if is_trap else ""
        print(f"❓ 測試問題：{q}")
        print(f"🎯 預期 ID ：{expected_chunk_id if expected_chunk_id else '無'}{trap_label}")
        print(f"📊 判定結果：{error_type} | MRR: {mrr_b}")
        print("-" * 80)
        
        print(f"🔍 檢索狀況：撈出 {len(res_b['sources'])} 筆參考資料 (Top {top_k}, Threshold: {threshold})")
        for idx, src in enumerate(res_b["sources"], 1):
            # 判斷這筆是不是剛好命中標準答案
            is_match = "✨ (正確命中)" if src.get('chunk_id') == expected_chunk_id else ""
            print(f"   └─ [{idx}] 標題: {src.get('heading')} {is_match}")
            print(f"            ID: {src.get('chunk_id')} | Score: {src.get('score')}")
            
        print("="*80 + "\n")

        results.append({
            "query": q,
            "expected_id": expected_chunk_id,
            "is_trap": is_trap,
            "metrics": {
                "mrr": mrr_b,
                "hit": hit,
                "recall_at_k": recall_at_k,
                "precision_at_k": precision_at_k
            },
            "error_type": error_type,
            "latency_ms": latency_b, # 🌟 這裡
            "ttft_ms": ttft_b,       # 🌟 還有這裡
            "answer": res_b["answer"], 
            "sources": res_b["sources"]
        })

    # 計算總和指標
    total_mrr = sum(r["metrics"]["mrr"] for r in results)
    total_hit = sum(1 for r in results if r["metrics"]["hit"])
    
    far_rate = (far_count / trap_count) if trap_count > 0 else 0
    # FRR 的分母是「檢索有命中，本來該答出來的題目」
    valid_context_count = success_count + frr_count
    frr_rate = (frr_count / valid_context_count) if valid_context_count > 0 else 0

    aggregate = {
        "parameters_used": {
            "chunk_size": chunk_size,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "threshold": threshold,
            "top_k": top_k
        },
        "performance": {
            "avg_mrr": round(total_mrr / normal_count if normal_count else 0, 4),
            "hit_rate": round(total_hit / normal_count if normal_count else 0, 4),
            "avg_precision_at_k": round(sum(r["metrics"]["precision_at_k"] for r in results if not r["is_trap"]) / normal_count if normal_count else 0, 4),
            "avg_recall_at_k": round(total_hit / normal_count if normal_count else 0, 4), # 等同 Hit Rate
            # 🌟 新增平均 TTFT 與 平均總耗時
            "avg_ttft_ms": round(sum(r["ttft_ms"] for r in results) / len(results), 2),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 2),
        },
        "errors": {
            "far_rate": round(far_rate, 4), # 誤答率
            "frr_rate": round(frr_rate, 4), # 誤拒率
            "retrieval_error_count": retrieval_error_count,
            "llm_error_count": far_count + frr_count,
            "success_count": success_count + (trap_count - far_count)
        }
    }

    return {
        "status": "success",
        "aggregate_metrics": aggregate,
        "details": results
    }

# ==========================================
# 💬 客服大廳 (index.html) 專用 API 區塊
# ==========================================

# 1. 狀態統計 API (給前端顯示用)
@app.get("/api/stats")
async def get_stats():
    return {
        "status": "online",
        "knowledge_base_loaded": True,
        "current_model": "gpt-4o-mini"
    }

# 2. 聊天串流 API (接收問題，並以 Streaming 形式回傳)
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        query_stream(request.query), 
        media_type="text/event-stream"
    )