import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# from langchain.schema import Document
from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 自動動態推算專案根目錄，避免路徑地獄
_current = Path(__file__).resolve()
while _current.name != "scaffold" and _current.parent != _current:
    _current = _current.parent
ROOT_DIR = _current.parent

DOCS_DIR = ROOT_DIR / "docs"
# FAISS_DIR = ROOT_DIR / ".kb" / "faiss_index"
# BM25_INDEX_PATH = ROOT_DIR / ".kb" / "index.json"  # 👈 規格要求的 inspectable JSON

# 2. 🌟 將寫死路徑改為「動態路徑工廠」
def get_kb_paths(embedding_model_name: str, chunk_size: int):
    """
    依據使用的模型與切塊大小，動態建立並回傳專屬的儲存路徑
    """
    safe_model_name = embedding_model_name.replace("/", "_")
    # 資料夾名稱範例: .kb/text-embedding-3-small_c500 或 .kb/ibm-granite_granite-embedding-311m-multilingual-r2_c300
    kb_folder = ROOT_DIR / ".kb" / f"{safe_model_name}_c{chunk_size}"
    kb_folder.mkdir(parents=True, exist_ok=True)
    
    return {
        "faiss_dir": kb_folder / "faiss_index",
        "bm25_path": kb_folder / "index.json"
    }

EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# 👇 新增這行：專門捕捉 
METADATA_RE = re.compile(r"^<!--\s*(\{.*?\})\s*-->$")

# 👇 新增這行：萃取 heading 中的 [cc_faq_xxxxxxxx] 格式 ID
CHUNK_ID_RE = re.compile(r"^\[([a-zA-Z0-9_]+)\]\s*")

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
    metadata: dict = None # 👈 新增這行：讓 BM25 也能儲存產品與標籤

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
            "metadata": self.metadata or {} # 👈 匯出成 JSON 時一併匯出
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

# 🌟 【新增】根據 source 檔名與 heading 產生穩定的 hash ID
def generate_chunk_id(src_file: str, heading: str) -> str:
    """以 source#heading 為基礎產生 8 碼 MD5 hash ID，格式：{filename}_{hash}"""
    raw = f"{src_file}#{heading}"
    hash_suffix = hashlib.md5(raw.encode()).hexdigest()[:8]
    base_name = src_file.replace(".md", "")
    return f"{base_name}_{hash_suffix}"

def load_markdown_sections(path: Path) -> list[Document]:
    """文章拆解員：將 Markdown 檔案切成 Section-level 紀錄，並支援讀取隱藏 Metadata"""
    content = path.read_text(encoding="utf-8")
    docs = []
    
    # current_heading = "Introduction"
    current_chunk_id = generate_chunk_id(path.name, "Introduction")  # 預設值
    heading_hierarchy = ["Introduction"]
    current_content = []
    current_metadata = {}  # 👈 新增：暫存目前段落的標籤

    for line in content.splitlines():
        heading_match = HEADING_RE.match(line) 
        meta_match = METADATA_RE.match(line) # 👈 偵測是不是隱藏標籤
        
        if heading_match:
            text = "\n".join(current_content).strip()
            if text:
                base_meta = {
                    "source": path.name, 
                    "heading": slugify(current_heading),
                    "heading_path": list(heading_hierarchy),
                    "chunk_id": current_chunk_id,  # 🌟 加入萃取到的 ID
                    "effective_date": None,   # ✨ 預設 None，可由 MD 隱藏註解覆蓋
                    "expiry_date": None,      # ✨ 預設 None（永久有效）
                }
                base_meta.update(current_metadata) # 👈 將抓到的 Metadata 合併進去
                docs.append(Document(page_content=text, metadata=base_meta))

            # 更新標題與層級結構
            level = len(heading_match.group(1))
            raw_heading = heading_match.group(2)

            # 🌟 嘗試從標題萃取 ID，萃取後的 heading 去掉 [id] 前綴
            id_match = CHUNK_ID_RE.match(raw_heading)
            
            if id_match:
                current_chunk_id = id_match.group(1)           # cc_faq_739305e6
                current_heading = raw_heading[id_match.end():]  # Q1: 個人資料權利行使...
            else:
                current_chunk_id = generate_chunk_id(path.name, slugify(raw_heading))  # fallback
                current_heading = raw_heading
            
            if level == 1:
                heading_hierarchy = [current_heading]
            else:
                heading_hierarchy = heading_hierarchy[:level-1]
                while len(heading_hierarchy) < level - 1:
                    heading_hierarchy.append("Unknown")
                heading_hierarchy.append(current_heading)
                
            current_content = [] 
            current_metadata = {} # 👈 清空，準備迎接下個段落的標籤
            
        elif meta_match:
            # 👈 如果抓到隱藏註解，就把它 Parse 成 JSON 字典
            try:
                current_metadata = json.loads(meta_match.group(1))
            except json.JSONDecodeError:
                print(f"⚠️ [Warning] 無法解析 {path.name} 中的 Metadata: {line}")
        else:
            current_content.append(line)

    # 處理檔案最後一個段落
    text = "\n".join(current_content).strip()
    if text:
        base_meta = {
            "source": path.name, 
            "heading": slugify(current_heading),
            "heading_path": list(heading_hierarchy)
        }
        base_meta.update(current_metadata)
        docs.append(Document(page_content=text, metadata=base_meta))
        
    return docs

class HybridIndexer:
    def __init__(self):
        self.vectorstore = None
        self._embeddings = None
        self.files_indexed = 0
        self.sections_indexed = 0

        # 🌟 狀態追蹤器：紀錄現在腦袋裡裝的是哪一個模型與大小
        self.current_chunk_size = 500
        self.current_embedding_model = "text-embedding-3-small"
        
        self.bm25_sections: list[BM25Section] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len = 0.0

    def rebuild_stats(self) -> None:
        self.doc_freq = Counter()
        if not self.bm25_sections:
            self.avg_doc_len = 0.0
            return

        total_tokens = 0
        unique_files = set()

        for sec in self.bm25_sections:
            unique_files.add(sec.file)
            total_tokens += len(sec.tokens)
            for token in set(sec.tokens):
                self.doc_freq[token] += 1

        self.files_indexed = len(unique_files)
        self.sections_indexed = len(self.bm25_sections)
        self.avg_doc_len = total_tokens / self.sections_indexed

    def write_index_json(self, bm25_path: Path) -> None:
        """存入 JSON，現在需要吃動態路徑參數"""
        payload = {
            "stats": {
                "files_indexed": self.files_indexed,
                "sections_indexed": self.sections_indexed,
                "avg_doc_len": round(self.avg_doc_len, 2)
            },
            "sections": [sec.to_dict() for sec in self.bm25_sections]
        }
        with open(bm25_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_index(self, embedding_model: str = "text-embedding-3-small", chunk_size: int = 500) -> tuple[int, int]:
        """
        🌟 實驗核心：根據參數，喚醒對應目錄下的雙軌大腦
        """
        from .retrieval import get_embedding_model
        self.current_embedding_model = embedding_model
        self.current_chunk_size = chunk_size
        
        # 1. 取得當下參數專屬的存放目錄
        paths = get_kb_paths(embedding_model, chunk_size)
        
        # 2. 準備對應的 Embedding 實體 (HuggingFace 或是 OpenAI)
        self._embeddings = get_embedding_model(embedding_model)

        # 3. 喚醒 FAISS 向量大腦
        faiss_path = paths["faiss_dir"]
        if (faiss_path / "index.faiss").exists():
            self.vectorstore = FAISS.load_local(
                folder_path=str(faiss_path),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = None # 找不到就設為 None，讓 eval_server 知道要觸發 rebuild

        # 4. 喚醒 BM25 統計大腦
        bm25_path = paths["bm25_path"]
        if bm25_path.exists():
            with open(bm25_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.bm25_sections = [BM25Section(**sec) for sec in payload["sections"]]
            self.rebuild_stats()
        else:
            self.bm25_sections = []

        return self.files_indexed, self.sections_indexed

    def build_index(self, chunk_size: int = 500, chunk_overlap: int = 50, embedding_model: str = "text-embedding-3-small") -> tuple[int, int]:
        """
        🌟 實驗核心：根據參數，重新切塊、算向量，並存入專屬目錄
        """
        from .retrieval import get_embedding_model
        self.current_chunk_size = chunk_size
        self.current_embedding_model = embedding_model
        
        # 取得專屬存放目錄
        paths = get_kb_paths(embedding_model, chunk_size)

        if not DOCS_DIR.exists():
            return 0, 0

        all_docs = []
        self.files_indexed = 0
        for md_file in DOCS_DIR.glob("*.md"):
            sections = load_markdown_sections(md_file)
            all_docs.extend(sections)
            self.files_indexed += 1

        if not all_docs:
            return 0, 0

        # 動態調整切塊大小
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " "]
        )
        chunks = splitter.split_documents(all_docs)

        self.bm25_sections = []
        for idx, chunk in enumerate(chunks):
            src_file = chunk.metadata.get("source", "unknown")
            heading = chunk.metadata.get("heading", "unknown")
            heading_path = chunk.metadata.get("heading_path", [heading])

            sec_id = chunk.metadata.get("chunk_id") or generate_chunk_id(src_file, heading)
            chunk.metadata["chunk_id"] = sec_id
            
            token_text = " ".join(heading_path) + " " + chunk.page_content
            tokens = tokenize(token_text)

            self.bm25_sections.append(BM25Section(
                id=sec_id, file=src_file, heading=heading, heading_path=heading_path,
                content=chunk.page_content, tokens=tokens, metadata=chunk.metadata
            ))

        # 呼叫 retrieval 的工廠取得正確的 Embedding 模型
        self._embeddings = get_embedding_model(embedding_model)
        
        # 建立 FAISS 向量大腦並存檔
        self.vectorstore = FAISS.from_documents(chunks, self._embeddings)
        
        if paths["faiss_dir"].exists():
            shutil.rmtree(paths["faiss_dir"])
        paths["faiss_dir"].mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(paths["faiss_dir"]))

        # 儲存 BM25 統計大腦
        self.rebuild_stats()
        self.write_index_json(paths["bm25_path"])

        # 寫入 metadata 紀錄
        vector_metadata = {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "files_indexed": self.files_indexed,
            "sections_indexed": self.sections_indexed
        }
        with open(paths["faiss_dir"] / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(vector_metadata, f, indent=2)

        return self.files_indexed, self.sections_indexed

    def bm25_score(self, query_tokens: list[str], section: BM25Section, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        doc_len = len(section.tokens)
        if doc_len == 0 or self.avg_doc_len == 0:
            return 0.0

        tf_counter = Counter(section.tokens)
        for token in query_tokens:
            if token not in section.tokens:
                continue

            tf = tf_counter[token]
            df = self.doc_freq.get(token, 0)
            idf = math.log((self.sections_indexed - df + 0.5) / (df + 0.5) + 1.0)
            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len)))
            term_score = idf * tf_component
            
            if any(token in hp.lower() for hp in section.heading_path):
                term_score *= 1.2

            score += term_score
        return score

# 宣告全局單例
indexer = HybridIndexer()

# 為了讓外部可以直接呼叫 (預設參數)
def build_index() -> tuple[int, int]:
    return indexer.build_index()

def load_index_json() -> tuple[int, int]:
    return indexer.load_index()

if __name__ == "__main__":
    files, sections = build_index()
    print(f"✅ 建索引完成：{files} 個檔案，{sections} 個 sections")