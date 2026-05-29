import os
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import json

# 1. 調整引入：改從我們新寫的 hybrid_indexer 引入核心大腦與中文斷詞器
from .hybrid_indexer import indexer, tokenize, Document

import asyncio # 實現非同步串流

SYSTEM_PROMPT = """你是一個專業的銀行客服 AI 助手。
你的首要任務是**嚴格根據**下方提供的參考資料（CONTEXT）來回答使用者的問題。

**規則：**
1. 你只能使用參考資料中出現的資訊，絕對不能憑空捏造或使用外部知識。
2. 如果參考資料中找不到答案，請明確回答：「很抱歉，知識庫中目前沒有相關資訊。」
3. 每次回答時，必須引用資料來源。請完全照抄參考資料上方的 `[Source: filename#heading]` 格式。
4. 請務必使用流暢的**台灣繁體中文**進行回答。
5. 回答時請使用條列格式，每個步驟單獨一行，不要把所有內容擠在同一行。
6. 來源引用請放在回答的最末行，格式為：📚 來源：[Source: filename#heading]"""

# 🌟 新增：問題重寫專用的 SYSTEM PROMPT (警衛防線：只負責重寫，絕不代答)
REWRITE_SYSTEM_PROMPT = """你是一個 RAG 系統的「前置問題重寫專家」。
你的任務是根據使用者過去的【對話歷史紀錄】，將使用者【當下的最新口語發問】，重寫成一個「完全獨立、意思明確、且包含完整金融專有名詞」的繁體中文搜尋關鍵字句子。

**規範：**
1. 修正口語中的代名詞。例如將「它」、「這個」、「那怎麼辦」根據上下文替換成明確的主體（如「分期靈活金」、「提前還款違約金」）。
2. 如果當下的發問本來就是一個獨立且資訊完整的句子，請直接原樣輸出，不要做無謂的修改。
3. ⚠️ 鐵律：你只能輸出重寫後的那一個句子，絕對不能回答使用者的問題，也不要包含任何解釋或前言！"""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm

def rewrite_query(current_query: str, history_list: list) -> str:
    """🌟 新增：利用 LLM 結合歷史記憶，將口語問題重寫為獨立搜尋語句"""
    if not history_list:
        return current_query  # 如果沒有歷史紀錄，不需要重寫

    # 組裝歷史對話文字，提供給重寫大腦看
    formatted_history = []
    for msg in history_list[-6:]:  # 最多唯讀最近 3 輪 (6 條訊息)，避免 context 爆炸
        role_label = "使用者" if msg.role == "user" else "AI 客服"
        formatted_history.append(f"{role_label}: {msg.content}")
    history_context = "\n".join(formatted_history)

    prompt = f"【對話歷史紀錄】:\n{history_context}\n\n【當下最新口語發問】:\n{current_query}\n\n請輸出重寫後的獨立搜尋句子："

    try:
        response = get_llm().invoke([
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        rewritten = response.content.strip()
        print(f"[Query Rewriter] 原問題: '{current_query}' ➔ 重寫後: '{rewritten}'", flush=True)
        return rewritten
    except Exception as e:
        print(f"[Query Rewriter] 發生錯誤，退回原問題: {e}", flush=True)
        return current_query

def build_prompt(query: str, ranked_chunks: list) -> str:
    # 組合所有檢索出來的筆記片段 (Chunks)
    context_blocks = []
    for doc, score in ranked_chunks:
        source_name = doc.metadata.get("source", "unknown")
        heading_name = doc.metadata.get("heading", "unknown")
        source_id = f"{source_name}#{heading_name}"
        
        # 將來源標籤與實際筆記內容組裝（加入 Hybrid 融合分數方便除錯）
        block = f"[Source: {source_id}] (Hybrid Score: {round(score, 4)})\n{doc.page_content}\n"
        context_blocks.append(block)

    context_text = "\n".join(context_blocks)
    final_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"
    return final_prompt


def hybrid_search(question: str, k: int = 3) -> list:
    """混合檢索核心：結合 FAISS 向量名次與 BM25 關鍵字名次 (RRF 演算法)，並追蹤原始分數"""
    if indexer.vectorstore is None or not indexer.bm25_sections:
        return []

    # 1. 獲取向量檢索排名與原始分數
    vector_results = indexer.vectorstore.similarity_search_with_score(question, k=10)
    # FAISS 的 L2 距離越小名次越前面。建立一個對照表紀錄原始向量距離
    vector_raw_scores = {doc.page_content: dist for doc, dist in vector_results}
    vector_ranked = [doc for doc, _ in sorted(vector_results, key=lambda x: x[1])]

    # 2. 獲取 BM25 關鍵字檢索排名與原始分數
    query_tokens = tokenize(question)
    bm25_scored_sections = [
        (sec, indexer.bm25_score(query_tokens, sec))
        for sec in indexer.bm25_sections
    ]
    bm25_ranked_sections = [sec for sec, score in sorted(bm25_scored_sections, key=lambda x: x[1], reverse=True) if score > 0][:10]
    # 建立一個對照表紀錄原始 BM25 分數
    bm25_raw_scores = {sec.content: score for sec, score in bm25_scored_sections}

    # 3. RRF 排名融合
    rrf_constant = 60
    doc_rrf_registry = {}

    # 累加向量軌道
    for rank, doc in enumerate(vector_ranked):
        doc_id = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("heading"))
        doc_rrf_registry[doc_id] = {
            "doc": doc, 
            "score": 1 / (rrf_constant + rank + 1),
            "v_rank": rank + 1,
            "b_rank": "-"  # 先預設沒命中
        }

    # 融合關鍵字軌道
    for rank, sec in enumerate(bm25_ranked_sections):
        doc_id = (sec.content, sec.file, sec.heading)
        doc_obj = Document(
            page_content=sec.content, 
            metadata={"source": sec.file, "heading": sec.heading, "heading_path": sec.heading_path}
        )
        
        if doc_id in doc_rrf_registry:
            doc_rrf_registry[doc_id]["score"] += 1 / (rrf_constant + rank + 1)
            doc_rrf_registry[doc_id]["b_rank"] = rank + 1
        else:
            doc_rrf_registry[doc_id] = {
                "doc": doc_obj, 
                "score": 1 / (rrf_constant + rank + 1),
                "v_rank": "-",  # 代表向量沒進前 10
                "b_rank": rank + 1
            }

    # 4. 依照 RRF 總分排序，取出前 k 名
    final_hybrid_ranked = sorted(doc_rrf_registry.values(), key=lambda x: x["score"], reverse=True)[:k]
    
    # 5. 把原始的分數塞進 Document 裡面，帶出去給 query 函式
    results = []
    for item in final_hybrid_ranked:
        doc = item["doc"]
        # 將豐富的追蹤數據通通灌進臨時 metadata
        doc.metadata["_debug_rrf_score"] = item["score"]
        doc.metadata["_debug_v_score"] = vector_raw_scores.get(doc.page_content, "未進入前10")
        doc.metadata["_debug_b_score"] = bm25_raw_scores.get(doc.page_content, 0.0)
        doc.metadata["_debug_v_rank"] = item["v_rank"]
        doc.metadata["_debug_b_rank"] = item["b_rank"]
        results.append((doc, item["score"]))
        
    return results


# 設置 RRF 混合分數的及格防線
# 雙軌同時命中時分數會高於 0.025，單軌精準命中時約在 0.016。低於 0.016 通常代表不相關。
HYBRID_THRESHOLD = 0.016

def query(question: str, history: list = None) -> dict:
    # 1. 檢查雙軌大腦建置了沒
    if indexer.vectorstore is None or not indexer.bm25_sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }
    
    # 🌟 核心進化：執行問題重寫。如果使用者追問「那怎麼辦」，search_query 會被洗成「信用卡遺失該怎麼辦」
    search_query = rewrite_query(question, history if history else [])

    # 2. 啟動全新的雙軌混合搜尋 (Hybrid Search)
    # 🌟 使用重寫後的高純度問題去跑混合搜尋
    ranked_chunks = hybrid_search(search_query, k=3)
    
    # 3. 新增的門檻過濾邏輯：注意！RRF 分數是「越高越好」，所以用 >= 符飾
    valid_chunks = [(doc, score) for doc, score in ranked_chunks if score >= HYBRID_THRESHOLD]

    # 4. Fallback 機制：過濾完之後如果全軍覆沒，直接宣布找不到
    if not valid_chunks:
        return {
            "answer": "很抱歉，知識庫中目前沒有相關資訊。",
            "sources": [],
        }

    # 5. 把過濾後的優質筆記組合進 Prompt 丟給 OpenAI
    # 🌟 注意：餵給大模型回答時，我們放 search_query (讓它根據精準意圖回答)，也可以放原本的 question。這裡建議放 search_query 效果最穩健。
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(search_query, valid_chunks)),
    ])

    # 6. 整理來源資訊（補齊 content 欄位，確保通過 Pydantic 驗證）
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 4),  # RRF 融合分
            
            # 原始雙軌分數儀表板欄位
            "vector_score": f"{round(float(doc.metadata['_debug_v_score']), 4)}" if isinstance(doc.metadata["_debug_v_score"], (int, float, type(doc.metadata["_debug_v_score"]))) and "未進入" not in str(doc.metadata["_debug_v_score"]) else str(doc.metadata["_debug_v_score"]),
            "bm25_score": round(float(doc.metadata["_debug_b_score"]), 4),
            "vector_rank": doc.metadata["_debug_v_rank"],
            "bm25_rank": doc.metadata["_debug_b_rank"],
            
            # 💡 核心修正：將原本漏掉的 content 補回來！
            "content": doc.page_content[:240]
        }
        for doc, score in valid_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }

# 🌟 核心完全體：非同步串流生成器
async def query_stream(question: str, history: list = None):
    """【加分任務】嚴格依照 SSE 規範設計的 RAG 串流生成器 (source -> token -> done)"""
    loop = asyncio.get_running_loop()

    if indexer.vectorstore is None or not indexer.bm25_sections:
        yield "event: error\ndata: 知識庫尚未建立索引\n\n"
        return

    try:
        # 1. 使用 run_in_executor 保護同步函式，避免阻塞 ASGI 事件循環
        search_query = await loop.run_in_executor(
            None, rewrite_query, question, history if history else []
        )

        # 2. 混合搜尋
        ranked_chunks = await loop.run_in_executor(
            None, hybrid_search, search_query, 3
        )
        valid_chunks = [(doc, score) for doc, score in ranked_chunks if score >= HYBRID_THRESHOLD]

    except Exception as e:
        print(f"[Stream Error] 檢索階段發生錯誤: {e}", flush=True)
        yield f"event: error\ndata: 檢索失敗: {str(e)}\n\n"
        return

    # Fallback 阻斷防線
    if not valid_chunks:
        yield "event: token\ndata: 很抱歉，知識庫中目前沒有相關資訊。\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    try:
        # 3. 整理來源資訊（使用 float() 進行強制造型，防止 numpy/faiss 的 float32 破壞 json.dumps）
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "heading": doc.metadata.get("heading", "unknown"),
                "score": float(round(float(score), 4)),
                "vector_score": f"{round(float(doc.metadata['_debug_v_score']), 4)}" if isinstance(doc.metadata["_debug_v_score"], (int, float)) else str(doc.metadata["_debug_v_score"]),
                "bm25_score": float(round(float(doc.metadata["_debug_b_score"]), 0)) if isinstance(doc.metadata["_debug_b_score"], (int, float)) else 0.0,
                "vector_rank": doc.metadata["_debug_v_rank"],
                "bm25_rank": doc.metadata["_debug_b_rank"],
                "content": doc.page_content[:240]
            }
            for doc, score in valid_chunks
        ]

        # 🎵 串流第一步：發送 source 事件
        source_payload = {"sources": sources, "rewritten_query": search_query}
        yield f"event: source\ndata: {json.dumps(source_payload, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.05)
        
    except Exception as e:
        print(f"[Stream Error] 來源序列化失敗: {e}", flush=True)
        yield f"event: error\ndata: 資料格式化失敗: {str(e)}\n\n"
        return

    # 4. 組裝 RAG 提示詞
    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(search_query, valid_chunks)),
    ]

    # 🎵 串流第二步：調用 astream() 逐字生成答案
    full_answer = ""
    try:
        async for chunk in get_llm().astream(prompt_messages):
            token = getattr(chunk, "content", str(chunk))
            if token:
                full_answer += token
                yield f"event: token\ndata: {token}\n\n"
    except Exception as e:
        print(f"[Stream Error] LLM 串流階段發生錯誤: {e}", flush=True)
        yield f"event: error\ndata: 串流異常中斷: {str(e)}\n\n"
        return

    # 🎵 串流第三步：發送 done 事件
    final_payload = {"full_answer": full_answer}
    yield f"event: done\ndata: {json.dumps(final_payload, ensure_ascii=False)}\n\n"