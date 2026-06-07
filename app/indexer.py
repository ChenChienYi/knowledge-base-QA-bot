# 💡 總結這個處理中心的流水線：
# 讀取檔案 (load_markdown_sections) ➔ 建置大腦 (build_index) ➔ 存入硬碟 (save_vector_index) ➔ 有人發問時，進行比對 (search)

import json
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / ".kb" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# TODO: Configure chunking parameters for traditional RAG.
#
# Design decision: Balance semantic recall against context noise.
#
# Hints:
# 1. chunk_size around 500 chars is a reasonable prototype default.
# 2. chunk_overlap helps avoid cutting facts at boundaries.
# 3. separators should prefer Markdown structure before individual words.

# Chunking Parameters
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " "], # 加入全形標點符號
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0

# 標籤產生器: 用它轉換成乾淨的標籤（"refund-policy-2024"），方便系統記錄這段知識的來源位置。
# def slugify(text: str) -> str:
#     slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
#     return slug or "section"

def slugify(text: str) -> str:
    # 最簡單的改法：把空白替換成橫線，其他保留原樣
    slug = text.strip().replace(" ", "-")
    return slug or "section"

# 向量翻譯機: 翻譯成「數學向量」
def get_embeddings():
    global _embeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            request_timeout=20,
            max_retries=1,
        )
    return _embeddings

# 文章拆解員: 把.md檔案打開，並且根據 Markdown 的大標題（#），把長篇文章切成一段一段的獨立區塊。
def load_markdown_sections(path: Path) -> list[Document]:
    # TODO: Load Markdown into source-citable Document records.
    #
    # Design decision: Preserve filename#heading metadata before chunking.
    #
    # Hints:
    # 1. Use HEADING_RE to split by Markdown headings.
    # 2. Put heading_path and content into page_content.
    # 3. Store source metadata like "refund_policy.md#refund-timeline".
    content = path.read_text(encoding="utf-8")
    docs = []
    
    current_heading = "Introduction" # 預設標題（如果檔案開頭沒有標題）
    current_content = []

    for line in content.splitlines():
        # 用內建的正則表達式檢查這行是不是標題 (例如: ## 標題)
        match = HEADING_RE.match(line) 
        if match:
            # 遇到新標題了！先把「上一段」的內容存起來
            text = "\n".join(current_content).strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    # metadata 是為了讓 AI 回答時知道這段話來自哪裡
                    metadata={"source": path.name, "heading": slugify(current_heading)}
                ))
            # 更新目前的新標題，並重新開始收集內文
            current_heading = match.group(2)
            current_content = [line] 
        else:
            current_content.append(line)

    # 檔案讀完後，把最後一段也存起來
    text = "\n".join(current_content).strip()
    if text:
        docs.append(Document(
            page_content=text,
            metadata={"source": path.name, "heading": slugify(current_heading)}
        ))
        
    return docs

# 建置向量大腦: 這是 indexer.py 裡最重要的函式（也就是你剛才實作的那個）。它會指揮所有人：
#1. 叫「文章拆解員」(load_markdown_sections) 去讀取所有的 .md 檔案。
#2. 用 splitter 把段落切得更細碎。
#3. 叫「向量翻譯機」(get_embeddings) 把文字變成數學向量。
#4. 最後把這些向量統整成一個強大的 FAISS 搜尋資料庫（向量大腦）。
def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    # TODO: Build a FAISS vector index from docs/*.md.
    #
    # Hints:
    # 1. Load all Markdown files from docs_dir.
    # 2. Convert each heading section to a Document.
    # 3. Split documents into chunks with splitter.split_documents().
    # 4. Create FAISS.from_documents(chunks, get_embeddings()).
    # 5. Save the FAISS index to .kb/faiss_index/.
    # 6. Return (files_indexed, chunks_indexed).
    
    all_docs = []
    files_indexed = 0

    # 1. 讀取 docs_dir 裡面所有的 Markdown 檔案
    # 使用 .glob("*.md") 可以快速抓出資料夾底下所有的 md 檔
    for md_file in docs_dir.glob("*.md"):
        sections = load_markdown_sections(md_file)
        all_docs.extend(sections)
        files_indexed += 1

    sections_indexed = len(all_docs)

    # 如果資料夾裡完全沒有檔案，就直接提早結束，避免報錯
    if not all_docs:
        return 0, 0

    # 2. 把所有段落丟進切塊器，切成更小的 chunks
    chunks = splitter.split_documents(all_docs)

    # 3. 呼叫 OpenAI 把文字轉成向量，並建立 FAISS 索引
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    
    # 4. 把建好的索引存到硬碟裡，下次伺服器重啟就不用再算一次
    save_vector_index()

    return files_indexed, sections_indexed


# 將向量大腦存檔: 把大總管（build_index）辛苦建好的 FAISS 向量大腦，打包存進電腦硬碟（.kb/faiss_index/）
def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    # TODO: Persist the FAISS index so restart does not require re-embedding.
    #
    # Hints:
    # 1. Return early if vectorstore is None.
    # 2. Clear stale persisted files with shutil.rmtree(...) if the new index is empty.
    # 3. Use vectorstore.save_local(str(index_dir)).
    # 4. Write metadata.json with embedding_model, files_indexed, and sections_indexed.
    # 5. json.dumps(..., indent=2) makes the metadata easy to inspect.
    global vectorstore, files_indexed, sections_indexed

    # 如果目前還沒有建立任何向量索引，就直接提早結束
    if vectorstore is None:
        return

    # 如果舊的存檔資料夾已經存在，先把它整個刪除清空，避免新舊檔案混雜
    if index_dir.exists():
        shutil.rmtree(index_dir)
    
    # 建立全新的存檔資料夾
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1. 儲存 FAISS 向量索引（這會產生 index.faiss 和 index.pkl）
    vectorstore.save_local(str(index_dir))

    # 2. 準備並寫入 metadata.json，方便我們除錯以及下次啟動時讀取狀態
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed
    }
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

# 讀檔員: 當你的伺服器重開機（main.py 啟動）時，它會直接去硬碟把大腦載入記憶體。
def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    # TODO: Load .kb/faiss_index/ on server startup if it exists.
    #
    # Hints:
    # 1. Check for index.faiss and index.pkl.
    # 2. Read metadata.json and verify embedding_model still matches.
    # 3. Use FAISS.load_local(..., allow_dangerous_deserialization=True).
    # 4. Only use dangerous deserialization for indexes created by this local app.
    global vectorstore, files_indexed, sections_indexed

    # 檢查硬碟裡有沒有之前存過的索引檔案，如果沒有就直接回傳 0, 0
    if not (index_dir / "index.faiss").exists():
        return 0, 0

    # 1. 載入本機的 FAISS 向量資料庫
    # (注意：allow_dangerous_deserialization=True 是因為讀取我們自己產生的 .pkl 檔案是安全的，套件預設會阻擋)
    vectorstore = FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True
    )

    # 2. 嘗試讀取 metadata.json 來恢復檔案與段落的數量記錄
    try:
        with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
            # 確認存檔使用的模型與目前設定的模型是一致的
            if metadata.get("embedding_model") == EMBEDDING_MODEL:
                files_indexed = metadata.get("files_indexed", 0)
                sections_indexed = metadata.get("sections_indexed", 0)
    except Exception as e:
        print(f"Warning: Could not load metadata.json: {e}")

    return files_indexed, sections_indexed

# 檢索員: 它會把問題也轉成向量，然後去 FAISS 資料庫裡面找，撈出語意最接近的前 k 個筆記段落。
def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
