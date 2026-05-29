from fastapi import FastAPI

# 👈 將原本引用的舊 indexer 改為我們新寫好的 hybrid_indexer，
# 並引入同時能喚醒 FAISS 向量與 BM25 統計大腦的 load_index_json 函式
from .hybrid_indexer import load_index_json
from .routes import router
from .new_routes import new_router   # 👈 1. 引入你新寫的網頁路由

app = FastAPI(title="Vector RAG Knowledge Base Q&A Bot")
# 註冊路由
app.include_router(new_router)  # 👈 2. 將網頁路由掛載進去（建議放在 router 前面，讓首頁優先導向網頁）
app.include_router(router)


@app.on_event("startup")
def load_persisted_index():
    """當伺服器重新啟動時，從硬碟同時喚醒 FAISS 與 BM25 雙軌大腦"""
    try:
        files_count, sections_count = load_index_json()
        print(f"[Hybrid RAG] Successfully loaded persisted index. Files: {files_count}, Chunks: {sections_count}", flush=True)
    except Exception as exc:
        print(f"[Hybrid RAG] Skipping persisted index load due to error: {exc}", flush=True)