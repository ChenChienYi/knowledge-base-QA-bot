from fastapi import APIRouter

# 👈 核心關鍵：將原本引入的舊模組，全部改為引入你剛剛改好的 hybrid 完全體！
from .hybrid_indexer import build_index
from .retrieval import query  # 這會引入已經改好 RRF 融合與 HYBRID_THRESHOLD 的新 query
from .schemas import ChatRequest, ChatResponse, IndexResponse

# 🌟 核心修正：安全補上 StreamingResponse 的引入，消滅 NameError
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs():
    # 這裡會真正呼叫到 hybrid_indexer 的 build_index 函式
    files_count, sections_count = build_index()
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 這裡會真正執行到新 retrieval.py 裡面處理 Hybrid 搜尋的 query 函式
    return query(req.query, req.history) # 將請求中的 history 歷史紀錄一同傳入 query 處理中心

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    # 引入我們即時改造的非同步串流大腦
    from .retrieval import query_stream

    # 使用 FastAPI 內建的 StreamingResponse 封裝非同步生成器
    return StreamingResponse(
        query_stream(req.query, req.history),
        media_type="text/event-stream"
    )

@router.get("/api/stats")
def get_stats():
    # 引入全局單例 indexer
    from .hybrid_indexer import indexer
    return {
        "files_indexed": indexer.files_indexed,
        "sections_indexed": indexer.sections_indexed
    }