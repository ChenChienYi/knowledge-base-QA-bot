"""
retrieval.py（Langfuse SDK v4 相容版本）

核心改動：
  用 get_client().start_as_current_observation() context manager
  取代舊版的 trace() / span() / generation()
"""

import os
import time
import logging
import json
import asyncio

# from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse import get_client

from .hybrid_indexer import indexer, tokenize, Document

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一個專業的銀行客服 AI 助手。
你的首要任務是**嚴格根據**下方提供的參考資料（CONTEXT）來回答使用者的問題。

**規則：**
1. 你只能使用參考資料中出現的資訊，絕對不能憑空捏造或使用外部知識。
2. 如果參考資料中找不到答案，請明確回答：「很抱歉，知識庫中目前沒有相關資訊。」
3. 每次回答時，必須引用資料來源。請完全照抄參考資料上方的 `[Source: filename#heading]` 格式。
4. 請務必使用流暢的**台灣繁體中文**進行回答。
5. 回答時請使用條列格式，每個步驟單獨一行，不要把所有內容擠在同一行。
6. 來源引用請放在回答的最末行，格式為：📚 來源：[Source: filename#heading]"""

REWRITE_SYSTEM_PROMPT = """你是一個 RAG 系統的「前置問題重寫專家」。
你的任務是根據使用者過去的【對話歷史紀錄】，將使用者【當下的最新口語發問】，重寫成一個「完全獨立、意思明確、且包含完整金融專有名詞」的繁體中文搜尋關鍵字句子。

**規範：**
1. 修正口語中的代名詞。例如將「它」、「這個」、「那怎麼辦」根據上下文替換成明確的主體。
2. 如果當下的發問本來就是一個獨立且資訊完整的句子，請直接原樣輸出，不要做無謂的修改。
3. ⚠️ 鐵律：你只能輸出重寫後的那一個句子，絕對不能回答使用者的問題，也不要包含任何解釋或前言！"""

# ──────────────────────────────────────────────────────────────────
# LLM 單例
# ──────────────────────────────────────────────────────────────────

_llm = None

# 🌟 實作規格 5：動態抽換 LLM Model
def get_llm(model_name: str = "gpt-4o-mini"):
    # 每次都根據傳入的模型名稱回傳對應的 ChatOpenAI 實體
    return ChatOpenAI(
        model=model_name,
        request_timeout=20,
        max_retries=1,
    )

_EMBEDDING_CACHE = {}

def get_embedding_model(model_name: str = "text-embedding-3-small"):
    """
    根據名稱獲取 Embedding 模型的單例工廠函式
    """
    global _EMBEDDING_CACHE
    
    if model_name in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[model_name]
        
    if "granite-embedding" in model_name:
        print(f"📦 正在本地載入 HuggingFace 模型: {model_name}...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        print(f"🌐 正在初始化 OpenAI 雲端模型: {model_name}...")
        embeddings = OpenAIEmbeddings(model=model_name)
        
    _EMBEDDING_CACHE[model_name] = embeddings
    return embeddings
# ──────────────────────────────────────────────────────────────────
# rewrite_query
# ──────────────────────────────────────────────────────────────────

def rewrite_query(current_query: str, history_list: list) -> str:
    if not history_list:
        return current_query

    formatted_history = []
    for msg in history_list[-6:]:
        role_label = "使用者" if msg.role == "user" else "AI 客服"
        formatted_history.append(f"{role_label}: {msg.content}")
    history_context = "\n".join(formatted_history)

    prompt = (
        f"【對話歷史紀錄】:\n{history_context}\n\n"
        f"【當下最新口語發問】:\n{current_query}\n\n"
        f"請輸出重寫後的獨立搜尋句子："
    )

    try:
        response = get_llm().invoke([
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        rewritten = response.content.strip()
        logger.info(f"[Query Rewriter] 原問題: '{current_query}' ➔ 重寫後: '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"[Query Rewriter] 發生錯誤，退回原問題: {e}")
        return current_query


# ──────────────────────────────────────────────────────────────────
# build_prompt / hybrid_search（原邏輯不變）
# ──────────────────────────────────────────────────────────────────

def build_prompt(query: str, ranked_chunks: list) -> str:
    context_blocks = []
    for doc, score in ranked_chunks:
        source_name = doc.metadata.get("source", "unknown")
        heading_name = doc.metadata.get("heading", "unknown")
        source_id = f"{source_name}#{heading_name}"
        block = f"[Source: {source_id}] (Hybrid Score: {round(score, 4)})\n{doc.page_content}\n"
        context_blocks.append(block)
    context_text = "\n".join(context_blocks)
    return f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"


def hybrid_search(question: str, k: int = 3, filter_dict: dict = None) -> list:
    if indexer.vectorstore is None or not indexer.bm25_sections:
        return []

    search_kwargs = {"k": 10}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    vector_results = indexer.vectorstore.similarity_search_with_score(question, **search_kwargs)
    vector_raw_scores = {doc.page_content: dist for doc, dist in vector_results}
    vector_ranked = [doc for doc, _ in sorted(vector_results, key=lambda x: x[1])]

    query_tokens = tokenize(question)
    target_sections = indexer.bm25_sections
    if filter_dict:
        target_sections = [
            sec for sec in target_sections
            if sec.metadata and all(sec.metadata.get(key) == val for key, val in filter_dict.items())
        ]

    bm25_scored_sections = [
        (sec, indexer.bm25_score(query_tokens, sec)) for sec in target_sections
    ]
    bm25_ranked_sections = [
        sec for sec, score in sorted(bm25_scored_sections, key=lambda x: x[1], reverse=True)
        if score > 0
    ][:10]
    bm25_raw_scores = {sec.content: score for sec, score in bm25_scored_sections}

    rrf_constant = 60
    doc_rrf_registry = {}

    for rank, doc in enumerate(vector_ranked):
        doc_id = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("heading"))
        doc_rrf_registry[doc_id] = {
            "doc": doc, "score": 1 / (rrf_constant + rank + 1),
            "v_rank": rank + 1, "b_rank": "-",
        }

    for rank, sec in enumerate(bm25_ranked_sections):
        doc_id = (sec.content, sec.file, sec.heading)
        doc_obj = Document(
            page_content=sec.content,
            metadata={"source": sec.file, "heading": sec.heading,
                      "heading_path": sec.heading_path, **(sec.metadata or {})},
        )
        if doc_id in doc_rrf_registry:
            doc_rrf_registry[doc_id]["score"] += 1 / (rrf_constant + rank + 1)
            doc_rrf_registry[doc_id]["b_rank"] = rank + 1
        else:
            doc_rrf_registry[doc_id] = {
                "doc": doc_obj, "score": 1 / (rrf_constant + rank + 1),
                "v_rank": "-", "b_rank": rank + 1,
            }

    final_hybrid_ranked = sorted(
        doc_rrf_registry.values(), key=lambda x: x["score"], reverse=True
    )[:k]

    results = []
    for item in final_hybrid_ranked:
        doc = item["doc"]
        doc.metadata["_debug_rrf_score"] = item["score"]
        doc.metadata["_debug_v_score"] = vector_raw_scores.get(doc.page_content, "未進入前10")
        doc.metadata["_debug_b_score"] = bm25_raw_scores.get(doc.page_content, 0.0)
        doc.metadata["_debug_v_rank"] = item["v_rank"]
        doc.metadata["_debug_b_rank"] = item["b_rank"]
        results.append((doc, item["score"]))

    return results


HYBRID_THRESHOLD = 0.016


# ──────────────────────────────────────────────────────────────────
# query()  ← 同步版，Langfuse v4 插樁
# ──────────────────────────────────────────────────────────────────

# 🌟 修改：接收 threshold, llm_model, top_k
def query(question: str, history: list = None, filter_dict: dict = None, 
          llm_model: str = "gpt-4o-mini", threshold: float = 0.016, top_k: int = 3) -> dict:
          
    if indexer.vectorstore is None or not indexer.bm25_sections:
        return {"answer": "The knowledge base has not been indexed yet.", "sources": []}

    langfuse = get_client()

    with langfuse.start_as_current_observation(as_type="span", name="rag-query", input={"question": question}) as trace_span:

        # Step 1: Query Rewrite
        search_query = rewrite_query(question, history or [])

        # Step 2: Hybrid Retrieval (🌟 實作規格 2：動態門檻)
        ranked_chunks = hybrid_search(search_query, k=top_k, filter_dict=filter_dict)
        
        # 🌟 透過外部傳入的 threshold 來決定哪些 Chunk 及格
        valid_chunks = [(doc, score) for doc, score in ranked_chunks if score >= threshold]
        no_context = len(valid_chunks) == 0

        if no_context:
            return {"answer": "很抱歉，知識庫中目前沒有相關資訊。", "sources": []}
        
        # Step 3: LLM Generation (🌟 呼叫指定的 LLM)
        prompt_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(search_query, valid_chunks)),
        ]

        t0 = time.perf_counter()
        first_token_time = None
        full_answer = ""
        
        # 🌟 改用 stream() 逐字接收，精準捕捉 TTFT (首字時間)
        for chunk in get_llm(llm_model).stream(prompt_messages):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            full_answer += chunk.content
            
        # 計算時間 (包含前面的檢索時間 + LLM 時間)
        ttft_ms = int((first_token_time - t0) * 1000) if first_token_time else 0
        latency_ms = int((time.perf_counter() - t0) * 1000)

        sources = [
            {
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "source": doc.metadata.get("source", "unknown"),
                "heading": doc.metadata.get("heading", "unknown"),
                "score": round(float(score), 4),
                "content": doc.page_content[:240],
            }
            for doc, score in valid_chunks
        ]

        # 🌟 將 TTFT 與 Latency 包進回傳結果中
        return {
            "answer": full_answer, 
            "sources": sources,
            "ttft_ms": ttft_ms,
            "latency_ms": latency_ms
        }


# ──────────────────────────────────────────────────────────────────
# query_stream()  ← 非同步串流版
# ──────────────────────────────────────────────────────────────────

async def query_stream(question: str, history: list = None, filter_dict: dict = None):
    loop = asyncio.get_running_loop()

    if indexer.vectorstore is None or not indexer.bm25_sections:
        yield "event: error\ndata: 知識庫尚未建立索引\n\n"
        return

    langfuse = get_client()

    try:
        search_query = await loop.run_in_executor(
            None, rewrite_query, question, history or []
        )

        ranked_chunks = await loop.run_in_executor(
            None, hybrid_search, search_query, 3, filter_dict
        )
        valid_chunks = [(doc, score) for doc, score in ranked_chunks if score >= HYBRID_THRESHOLD]

    except Exception as e:
        logger.error(f"[Stream] 檢索階段錯誤: {e}")
        yield f"event: error\ndata: 檢索失敗: {str(e)}\n\n"
        return

    if not valid_chunks:
        yield "event: token\ndata: 很抱歉，知識庫中目前沒有相關資訊。\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    try:
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "heading": doc.metadata.get("heading", "unknown"),
                "score": float(round(float(score), 4)),
                "vector_score": (
                    f"{round(float(doc.metadata['_debug_v_score']), 4)}"
                    if isinstance(doc.metadata["_debug_v_score"], (int, float))
                    else str(doc.metadata["_debug_v_score"])
                ),
                "bm25_score": float(round(float(doc.metadata["_debug_b_score"]), 0))
                if isinstance(doc.metadata["_debug_b_score"], (int, float)) else 0.0,
                "vector_rank": doc.metadata["_debug_v_rank"],
                "bm25_rank": doc.metadata["_debug_b_rank"],
                "content": doc.page_content[:240],
            }
            for doc, score in valid_chunks
        ]
        source_payload = {"sources": sources, "rewritten_query": search_query}
        yield f"event: source\ndata: {json.dumps(source_payload, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.05)

    except Exception as e:
        logger.error(f"[Stream] 來源序列化失敗: {e}")
        yield f"event: error\ndata: 資料格式化失敗: {str(e)}\n\n"
        return

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(search_query, valid_chunks)),
    ]

    full_answer = ""
    stream_start = time.perf_counter()
    first_token_time = None

    try:
        async for chunk in get_llm().astream(prompt_messages):
            token = getattr(chunk, "content", str(chunk))
            if token:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    ttft_ms = int((first_token_time - stream_start) * 1000)
                    logger.info(f"[Stream] TTFT: {ttft_ms}ms")
                full_answer += token
                yield f"event: token\ndata: {token}\n\n"

    except Exception as e:
        logger.error(f"[Stream] LLM 串流錯誤: {e}")
        yield f"event: error\ndata: 串流異常中斷: {str(e)}\n\n"
        return

    # Langfuse 記錄串流結果
    total_ms = int((time.perf_counter() - stream_start) * 1000)
    ttft_ms = int((first_token_time - stream_start) * 1000) if first_token_time else 0

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="answer-generation-stream",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=[{"role": "system", "content": SYSTEM_PROMPT}],
        output=full_answer,
        metadata={"ttft_ms": ttft_ms, "total_ms": total_ms, "stream": True},
    ):
        pass

    yield f"event: done\ndata: {json.dumps({'full_answer': full_answer}, ensure_ascii=False)}\n\n"