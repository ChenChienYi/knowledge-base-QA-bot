"""
main.py（加入 Langfuse 可觀測性版本）

與原版差異：
  1. 加入 request_id middleware — 每筆請求都有唯一 ID 可追蹤
  2. 加入 access log middleware — 記錄每筆請求的路徑、狀態碼、延遲
  3. shutdown 事件 flush Langfuse，確保資料不遺漏
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import Response, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .hybrid_indexer import load_index_json
from .routes import router
from .new_routes import new_router
from .observability import flush

# ──────────────────────────────────────────
# Logging 基礎設定
# 輸出結構化 JSON-like 格式，方便後續接 CloudWatch / Datadog
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rag.access")

# ──────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────
app = FastAPI(title="Vector RAG Knowledge Base Q&A Bot")

app.mount("/static", StaticFiles(directory="static"), name="static")  # ✨ 新增

app.include_router(new_router)
app.include_router(router)

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")  # ✨ 新增


# ──────────────────────────────────────────
# Middleware 1：Request ID
# 每筆請求注入唯一 ID，方便跨 log 追蹤同一筆請求
# ──────────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ──────────────────────────────────────────
# Middleware 2：Access Log
# 記錄每筆請求的路徑、method、狀態碼、延遲
# （/health 跳過，避免 log 爆量）
# ──────────────────────────────────────────
@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    # 健康檢查不記錄
    if request.url.path == "/health":
        return await call_next(request)

    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "access",
        extra={
            "request_id": getattr(request.state, "request_id", "-"),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


# ──────────────────────────────────────────
# Startup：載入索引
# ──────────────────────────────────────────
@app.on_event("startup")
def load_persisted_index():
    # 初始化本地 SQLite DB
    from .database import init_db
    init_db()
    logger.info("[DB] qa_feedback table ready.")

    try:
        files_count, sections_count = load_index_json()
        logger.info(
            f"[Hybrid RAG] Index loaded. files={files_count} chunks={sections_count}"
        )
    except Exception as exc:
        logger.warning(f"[Hybrid RAG] Index load skipped: {exc}")


# ──────────────────────────────────────────
# Shutdown：確保 Langfuse 資料全部送出
# ──────────────────────────────────────────
@app.on_event("shutdown")
def shutdown_flush():
    logger.info("[Langfuse] Flushing events before shutdown...")
    flush()