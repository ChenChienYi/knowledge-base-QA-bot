"""
observability.py（Langfuse SDK v4 相容版本）

環境變數（加到 .env）：
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com
"""

import os
import logging
from langfuse import get_client

logger = logging.getLogger(__name__)

def get_langfuse():
    return get_client()

def flush():
    """確保所有事件送出，在 FastAPI shutdown 事件呼叫"""
    try:
        get_client().flush()
    except Exception as e:
        logger.warning(f"[Langfuse] flush 失敗: {e}")
