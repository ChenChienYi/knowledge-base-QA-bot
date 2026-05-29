import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 自動動態推算專案根目錄，避免路徑地獄
_current = Path(__file__).resolve()
while _current.name != "scaffold" and _current.parent != _current:
    _current = _current.parent
ROOT_DIR = _current.parent

DOCS_DIR = ROOT_DIR / "docs"
FAISS_DIR = ROOT_DIR / ".kb" / "faiss_index"
BM25_INDEX_PATH = ROOT_DIR / ".kb" / "index.json"  # 👈 規格要求的 inspectable JSON

EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# 繁體中文 + 英文數字混合斷詞器
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
ENG_WORD_RE = re.compile(r"[a-z0-9]+")

STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "from", 
    "how", "i", "is", "it", "my", "of", "the", "to", "what", "when", "which",
    "的", "了", "在", "是", "我", "你", "他", "它", "們", "與", "及", "和"
}

@dataclass
class BM25Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }

def slugify(text: str) -> str:
    return text.strip().replace(" ", "-")

def tokenize(text: str) -> list[str]:
    """專為中文與英數混合設計的斷詞器，並過濾停用字"""
    text_lower = text.lower()
    # 抓取所有中文字元
    tokens = CHINESE_CHAR_RE.findall(text_lower)
    # 抓取所有連續的英文單字或數字
    tokens.extend(ENG_WORD_RE.findall(text_lower))
    # 過濾掉停用字與空白
    return [t for t in tokens if t not in STOP_WORDS and t.strip()]

def load_markdown_sections(path: Path) -> list[Document]:
    """文章拆解員：將 Markdown 檔案切成 Section-level 紀錄"""
    content = path.read_text(encoding="utf-8")
    docs = []
    
    current_heading = "Introduction"
    heading_hierarchy = ["Introduction"]
    current_content = []

    for line in content.splitlines():
        match = HEADING_RE.match(line) 
        if match:
            text = "\n".join(current_content).strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": path.name, 
                        "heading": slugify(current_heading),
                        "heading_path": list(heading_hierarchy)
                    }
                ))
            
            # 更新標題與層級結構
            level = len(match.group(1)) # # 數量代表層級
            current_heading = match.group(2)
            
            # 動態調整 heading_path 樹狀結構
            if level == 1:
                heading_hierarchy = [current_heading]
            else:
                # 確保子標題能接在父標題後面
                heading_hierarchy = heading_hierarchy[:level-1]
                while len(heading_hierarchy) < level - 1:
                    heading_hierarchy.append("Unknown")
                heading_hierarchy.append(current_heading)
                
            current_content = [line] 
        else:
            current_content.append(line)

    text = "\n".join(current_content).strip()
    if text:
        docs.append(Document(
            page_content=text,
            metadata={
                "source": path.name, 
                "heading": slugify(current_heading),
                "heading_path": list(heading_hierarchy)
            }
        ))
        
    return docs

class HybridIndexer:
    def __init__(self):
        # 軌道一：Vector 變數
        self.vectorstore = None
        self._embeddings = None
        self.files_indexed = 0
        self.sections_indexed = 0
        
        # 軌道二：BM25 變數 (補齊 markdown_kb 框架所需的變數)
        self.bm25_sections: list[BM25Section] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len = 0.0

    def get_embeddings(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                request_timeout=20,
                max_retries=1,
            )
        return self._embeddings

    def rebuild_stats(self) -> None:
        """【補齊框架邏輯】計算 BM25 核心統計資料"""
        self.doc_freq = Counter()
        if not self.bm25_sections:
            self.avg_doc_len = 0.0
            return

        total_tokens = 0
        unique_files = set()

        for sec in self.bm25_sections:
            unique_files.add(sec.file)
            total_tokens += len(sec.tokens)
            # 每一個 token 在此區塊中不論出現幾次，對 doc_freq 而言都只算包含一次
            for token in set(sec.tokens):
                self.doc_freq[token] += 1

        self.files_indexed = len(unique_files)
        self.sections_indexed = len(self.bm25_sections)
        self.avg_doc_len = total_tokens / self.sections_indexed

    def write_index_json(self) -> None:
        """【補齊框架邏輯】將統計大綱存入 inspectable JSON"""
        os.makedirs(BM25_INDEX_PATH.parent, exist_ok=True)
        payload = {
            "stats": {
                "files_indexed": self.files_indexed,
                "sections_indexed": self.sections_indexed,
                "avg_doc_len": round(self.avg_doc_len, 2)
            },
            "sections": [sec.to_dict() for sec in self.bm25_sections]
        }
        with open(BM25_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

# Wiki Index
    def generate_wiki_index(self) -> None:
        """從記憶體結構自動導出人類與 Agent 可讀的維基總目錄"""
        if not self.bm25_sections:
            return

        WIKI_INDEX_PATH = ROOT_DIR / "wiki" / "index.md"
        os.makedirs(WIKI_INDEX_PATH.parent, exist_ok=True)

        # 1. 初始化 Markdown 標題與基礎宏觀統計
        md_lines = [
            "# 📚 銀行內部知識庫總目錄 (Wiki Index)\n",
            "*(本目錄由系統自動生成，請勿手動修改)*\n",
            f"目前架構內共包含 **{self.files_indexed}** 個核心範疇，共 **{self.sections_indexed}** 個精確主題章節。\n",
            "---\n"
        ]

        # 2. 依據檔案名稱 (file) 將所有的知識區塊進行分門別類 (Grouping)
        grouped_sections = {}
        for sec in self.bm25_sections:
            if sec.file not in grouped_sections:
                grouped_sections[sec.file] = []
            grouped_sections[sec.file].append(sec)

        # 3. 開始建立有層級的樹狀 Markdown 結構
        for file_name, sec_list in grouped_sections.items():
            # 為每個文件建立一個獨立的次級大標題
            icon = "💳" if "cc" in file_name.lower() else "💰"
            md_lines.append(f"## {icon} {file_name.replace('_', ' ').title()} 範疇 (`{file_name}`)\n")
            
            for sec in sec_list:
                # 排除單調的 Introduction 標題，只針對有意義的 FAQ 章節建立跳轉錨點
                if sec.heading.lower() == "introduction":
                    md_lines.append(f"* 📝 **文件導言 (Introduction)**\n")
                    continue
                
                # 計算階層深度：根據 heading_path 的長度來決定縮排的空白數
                # 這樣就能在 Markdown 中優雅呈現樹狀目錄
                indent = "    " * (len(sec.heading_path) - 1) if len(sec.heading_path) > 1 else ""
                
                # 👈 核心命名公式：[呈現給人類看的標題文字](檔案路徑#精確的章節定位針)
                md_lines.append(f"{indent}* 🔹 [{sec.heading}]({file_name}#{sec.heading})\n")
            
            md_lines.append("\n") # 每個檔案區塊間隔空一行

        # 4. 將組裝好的完整 Markdown 內容強行寫入 wiki/index.md
        with open(WIKI_INDEX_PATH, "w", encoding="utf-8") as f:
            f.writelines(md_lines)
        print(f"[Wiki Generator] Successfully auto-generated {WIKI_INDEX_PATH}", flush=True)

    def build_index(self) -> tuple[int, int]:
        """【核心融合】一鍵啟動雙軌索引建立流水線"""
        if not DOCS_DIR.exists():
            return 0, 0

        all_docs = []
        self.files_indexed = 0

        # 1. 讀取所有 Markdown 文件
        for md_file in DOCS_DIR.glob("*.md"):
            sections = load_markdown_sections(md_file)
            all_docs.extend(sections)
            self.files_indexed += 1

        if not all_docs:
            return 0, 0

        # 2. 針對「向量大腦」進行 Recursive 細切塊
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，", " "]
        )
        chunks = splitter.split_documents(all_docs)

        # 3. 建立 【軌道一】：FAISS 向量大腦
        self.vectorstore = FAISS.from_documents(chunks, self.get_embeddings())
        if FAISS_DIR.exists():
            shutil.rmtree(FAISS_DIR)
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(FAISS_DIR))

        # 4. 建立 【軌道二】：將切塊好的 Chunks 轉換成 BM25 關鍵字統計物件
        self.bm25_sections = []
        for idx, chunk in enumerate(chunks):
            src_file = chunk.metadata.get("source", "unknown")
            heading = chunk.metadata.get("heading", "unknown")
            heading_path = chunk.metadata.get("heading_path", [heading])
            
            # 建立可以用來配對的 token 清單 (包含標題路徑與內文)
            token_text = " ".join(heading_path) + " " + chunk.page_content
            tokens = tokenize(token_text)
            
            sec_id = f"{src_file}#{heading}__chunk{idx}"
            self.bm25_sections.append(BM25Section(
                id=sec_id,
                file=src_file,
                heading=heading,
                heading_path=heading_path,
                content=chunk.page_content,
                tokens=tokens
            ))

        # 5. 計算 BM25 統計指標並導出 JSON
        self.rebuild_stats()
        self.write_index_json()
        
        # 6. 把總目錄製作出來
        self.generate_wiki_index()

        # 寫入向量專屬的 metadata.json 以相容舊規格
        vector_metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "files_indexed": self.files_indexed,
            "sections_indexed": self.sections_indexed
        }
        with open(FAISS_DIR / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(vector_metadata, f, indent=2)

        return self.files_indexed, self.sections_indexed

    def load_index(self) -> tuple[int, int]:
        """當伺服器重新啟動時，從硬碟同時喚醒雙軌大腦"""
        # 1. 喚醒向量大腦
        if (FAISS_DIR / "index.faiss").exists():
            self.vectorstore = FAISS.load_local(
                folder_path=str(FAISS_DIR),
                embeddings=self.get_embeddings(),
                allow_dangerous_deserialization=True
            )
        
        # 2. 喚醒 BM25 統計大腦
        if BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.bm25_sections = [
                BM25Section(**sec) for sec in payload["sections"]
            ]
            self.rebuild_stats()

        return self.files_indexed, self.sections_indexed

    def bm25_score(self, query_tokens: list[str], section: BM25Section, k1: float = 1.5, b: float = 0.75) -> float:
        """【補齊框架邏輯】實作精準的 BM25 數學計算公式"""
        score = 0.0
        doc_len = len(section.tokens)
        if doc_len == 0 or self.avg_doc_len == 0:
            return 0.0

        # 計算當前區塊的詞頻統計
        tf_counter = Counter(section.tokens)

        for token in query_tokens:
            if token not in section.tokens:
                continue

            tf = tf_counter[token]
            df = self.doc_freq.get(token, 0)
            
            # 計算 逆向文件頻率 (IDF)
            # 使用經典的 BM25 IDF 公式，加上 0.5 避免負數
            idf = math.log((self.sections_indexed - df + 0.5) / (df + 0.5) + 1.0)
            
            # BM25 核心公式（結合字數正規化）
            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len)))
            term_score = idf * tf_component
            
            # 加分項修正 (Stretch Hint)：如果關鍵字出現在標題路徑中，額外給予 20% 權重加成
            if any(token in hp.lower() for hp in section.heading_path):
                term_score *= 1.2

            score += term_score

        return score

# 宣告全局單例，保持與原有程式架構的相容性
indexer = HybridIndexer()

def build_index() -> tuple[int, int]:
    return indexer.build_index()

def load_index_json() -> tuple[int, int]:
    return indexer.load_index()