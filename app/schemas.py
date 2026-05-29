from pydantic import BaseModel
from typing import Union  # 👈 新增引入 Union，因為名次可能是整數或 "-" 字串


# 🌟 新增：定義單條歷史訊息的結構
class MessageHistory(BaseModel):
    role: str      # "user" 或 "assistant"
    content: str   # 對話內容

class IndexResponse(BaseModel):
    files_indexed: int
    sections_indexed: int

class ChatRequest(BaseModel):
    query: str
    history: list[MessageHistory] = []  # 🌟 新增：允許前端傳入對話歷史紀錄，預設為空串列

class SourceInfo(BaseModel):
    source: str
    heading: str
    score: float
    content: str
    # 💡 核心關鍵：在這裡補上這四個 Debug 變數的型態登記，FastAPI 才不會把它們抹除！
    vector_score: str
    bm25_score: float
    vector_rank: Union[int, str]
    bm25_rank: Union[int, str]

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]

# 🌟 新增：專門用於串流初始階段傳輸來源資訊的模型 for streaming interface
class StreamSourceResponse(BaseModel):
    sources: list[SourceInfo]
    rewritten_query: str