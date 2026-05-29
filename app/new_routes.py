from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

# 這裡不引入原本的業務邏輯，因為這個檔案「只專注在提供網頁畫面」
new_router = APIRouter()

# 動態推算 index.html 的位置（跟 new_routes.py 在同一個 app 資料夾下）
HTML_FILE_PATH = Path(__file__).resolve().parent / "index.html"

@new_router.get("/", response_class=HTMLResponse)
def read_browser_ui():
    """當使用者用瀏覽器打開首頁時，返回簡易 Browser UI 網頁"""
    if not HTML_FILE_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html 檔案不存在，請確保它被放在 app/ 資料夾下")
    
    with open(HTML_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()