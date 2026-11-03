"""SQLite knowledge storage and explainable Chinese lexical BM25 retrieval.

No embeddings, vector database, model training or remote requests are used here.
Every connection is short lived; this database is separate from OCR history.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sqlite3
import uuid
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARACTERS = 200_000
MAX_PDF_PAGES = 40
MAX_DOCUMENTS = 100
RETRIEVAL_METHOD = "BM25 中文词法检索（非向量）"
_STOP = set("的了是在和与及为也有这那把被就都而或呢吗请")
_STOP_PAIRS = {"什么", "如何", "怎么", "一个", "这个", "那个", "我们", "请问", "一下"}


class KnowledgeError(ValueError):
    def __init__(self, message, code="INVALID_INPUT", status=400):
        super().__init__(message)
        self.code, self.status = code, status


def tokenize(text):
    terms = []
    for part in re.findall(r"[a-zA-Z][a-zA-Z0-9_+.-]*|[\u3400-\u9fff]+|\d+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            terms.extend("c1:" + char for char in part if char not in _STOP)
            terms.extend("c2:" + part[i:i + 2] for i in range(len(part) - 1)
                         if part[i:i + 2] not in _STOP_PAIRS)
        else:
            terms.append("w:" + part)
    return terms


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeError("知识内容不能为空。", "EMPTY_DOCUMENT")
    if "\x00" in value:
        raise KnowledgeError("文档包含无法处理的空字符，请上传 UTF-8 纯文本。", "INVALID_DOCUMENT")
    if len(value) > MAX_TEXT_CHARACTERS:
        raise KnowledgeError("文档正文不能超过 200000 个字符，请拆分后导入。", "DOCUMENT_TOO_LARGE", 413)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _title(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 120:
        raise KnowledgeError("文档标题应为 1 至 120 个字符。", "INVALID_TITLE")
    return value.strip()


def _chunks(text, maximum=700, overlap=80):
    # Fixed character windows prefer paragraph/sentence endings. Overlap retains
    # local context without treating neighbouring snippets as independent facts.
    start = 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            candidates = [text.rfind(mark, start + maximum // 2, end) for mark in ("\n", "。", "！", "？", ";")]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def read_uploaded_document(filename, raw):
    if not isinstance(filename, str) or not filename.strip():
        raise KnowledgeError("请选择有文件名的 txt、md 或 PDF 文件。", "INVALID_FILENAME")
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or len(filename) > 180:
        raise KnowledgeError("文件名过长或无效。", "INVALID_FILENAME")
    if not isinstance(raw, bytes) or not raw:
        raise KnowledgeError("上传文件为空。", "EMPTY_DOCUMENT")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise KnowledgeError("知识文件不能超过 2 MB。", "UPLOAD_TOO_LARGE", 413)
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise KnowledgeError("txt 和 md 文件必须使用 UTF-8 编码。", "INVALID_TEXT_ENCODING") from None
    elif suffix == ".pdf":
        if not raw.lstrip().startswith(b"%PDF-"):
            raise KnowledgeError("文件不是有效的 PDF。", "INVALID_PDF")
        try:
            from pypdf import PdfReader
        except ImportError:
            raise KnowledgeError("PDF 文本解析依赖尚未安装，请安装 pypdf，或改用 txt/md。", "PDF_DEPENDENCY_MISSING", 503) from None
        try:
            reader = PdfReader(io.BytesIO(raw), strict=False)
            if reader.is_encrypted:
                raise KnowledgeError("不支持加密 PDF，请先另存为未加密的文本 PDF。", "ENCRYPTED_PDF")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise KnowledgeError("PDF 最多支持 40 页，请拆分后上传。", "PDF_TOO_MANY_PAGES", 413)
            pages, decoded_bytes, characters = [], 0, 0
            for page in reader.pages:
                contents = page.get_contents()
                if contents is not None:
                    decoded_bytes += len(contents.get_data())
                    if decoded_bytes > 20 * 1024 * 1024:
                        raise KnowledgeError("PDF 解压后的内容过大，请拆分或转为 txt。", "PDF_CONTENT_TOO_LARGE", 413)
                page_text = page.extract_text() or ""
                characters += len(page_text)
                if characters > MAX_TEXT_CHARACTERS:
                    raise KnowledgeError("PDF 提取文字超过 200000 字符，请拆分。", "DOCUMENT_TOO_LARGE", 413)
                pages.append(page_text)
            text = "\n\n".join(pages)
        except KnowledgeError:
            raise
        except Exception:
            raise KnowledgeError("PDF 无法解析，请使用完整、有效的文本 PDF。", "INVALID_PDF") from None
        if not text.strip():
            raise KnowledgeError("PDF 没有可提取文字；扫描件请先用文字识别并人工校对，再确认导入。", "PDF_HAS_NO_TEXT", 422)
    else:
        raise KnowledgeError("知识文件仅支持 txt、md 和含文本的 PDF。", "UNSUPPORTED_DOCUMENT")
    return filename, _text(text)


class KnowledgeStore:
    def __init__(self, database, builtin_dir=None, seed_builtin=True):
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        if seed_builtin and builtin_dir is not None:
            self._seed(Path(builtin_dir))

    def _connect(self):
        db = sqlite3.connect(str(self.database), timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _initialize(self):
        with closing(self._connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
                    source_type TEXT NOT NULL, source_record_id INTEGER,
                    text TEXT NOT NULL, characters INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL, chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL, content TEXT NOT NULL, token_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_terms (
                    chunk_id TEXT NOT NULL REFERENCES knowledge_chunks(id) ON DELETE CASCADE,
                    term TEXT NOT NULL, tf INTEGER NOT NULL,
                    PRIMARY KEY(chunk_id,term)
                );
                CREATE INDEX IF NOT EXISTS ix_knowledge_term ON knowledge_terms(term);
                CREATE INDEX IF NOT EXISTS ix_knowledge_doc_chunks ON knowledge_chunks(document_id,position);
                CREATE INDEX IF NOT EXISTS ix_knowledge_content_hash ON knowledge_documents(content_sha256);
                CREATE TABLE IF NOT EXISTS knowledge_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            db.commit()

    @staticmethod
    def _metadata(row):
        return {key: row[key] for key in ("id", "title", "filename", "source_type", "source_record_id",
                                         "chunk_count", "characters", "created_at")}

    def _seed(self, folder):
        with closing(self._connect()) as db:
            seeded = db.execute("SELECT value FROM knowledge_settings WHERE key='builtins_v1'").fetchone()
        if seeded:
            return
        for path in sorted(folder.glob("*.md")):
            text = path.read_text(encoding="utf-8-sig")
            title = text.splitlines()[0].lstrip("# ").strip() if text.strip() else path.stem
            self.add_document(title, text, path.name, "builtin")
        with closing(self._connect()) as db:
            db.execute("INSERT OR REPLACE INTO knowledge_settings VALUES ('builtins_v1','done')")
            db.commit()

    def counts(self):
        with closing(self._connect()) as db:
            return {"document_count": db.execute("SELECT COUNT(*) FROM knowledge_documents").fetchone()[0],
                    "chunk_count": db.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]}

    def list_documents(self):
        with closing(self._connect()) as db:
            return [self._metadata(row) for row in db.execute("SELECT * FROM knowledge_documents ORDER BY created_at DESC,id DESC")]

    def add_document(self, title, text, filename="note.md", source_type="note", source_record_id=None):
        title, text = _title(title), _text(text)
        if source_type not in {"note", "upload", "builtin", "ocr"}:
            raise KnowledgeError("知识来源无效。", "INVALID_SOURCE")
        if not isinstance(filename, str) or len(filename) > 180:
            raise KnowledgeError("文件名无效。", "INVALID_FILENAME")
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if source_record_id is not None and (type(source_record_id) is not int or source_record_id < 1):
            raise KnowledgeError("OCR 记录编号无效。", "INVALID_RECORD_ID")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with closing(self._connect()) as db:
            # Serialize quota and deduplication checks with the following write.
            db.execute("BEGIN IMMEDIATE")
            same = db.execute("SELECT * FROM knowledge_documents WHERE content_sha256=?", (digest,)).fetchone()
            if same:
                return {**self._metadata(same), "duplicate": True}
            if db.execute("SELECT COUNT(*) FROM knowledge_documents").fetchone()[0] >= MAX_DOCUMENTS:
                raise KnowledgeError("知识库最多保存 100 份文档，请删除不需要的资料后再导入。", "KNOWLEDGE_QUOTA", 413)
            identifier = uuid.uuid4().hex
            stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            chunks = list(_chunks(text))
            db.execute("INSERT INTO knowledge_documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (identifier, title, filename, source_type, source_record_id, text, len(text), digest, len(chunks), stamp))
            for position, chunk in enumerate(chunks):
                chunk_id = uuid.uuid4().hex
                counts = Counter(tokenize(title + " " + title + " " + chunk))
                db.execute("INSERT INTO knowledge_chunks VALUES (?,?,?,?,?)", (chunk_id, identifier, position, chunk, sum(counts.values())))
                db.executemany("INSERT INTO knowledge_terms VALUES (?,?,?)", [(chunk_id, term, tf) for term, tf in counts.items()])
            db.commit()
            row = db.execute("SELECT * FROM knowledge_documents WHERE id=?", (identifier,)).fetchone()
            return {**self._metadata(row), "duplicate": False}

    def get_document(self, document_id, max_characters=12000):
        if not isinstance(document_id, str) or not re.fullmatch(r"[a-f0-9]{32}", document_id):
            raise KnowledgeError("文档不存在。", "DOCUMENT_NOT_FOUND", 404)
        if type(max_characters) is not int or not 1 <= max_characters <= 20000:
            raise KnowledgeError("读取长度应为 1 至 20000。", "INVALID_LIMIT")
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM knowledge_documents WHERE id=?", (document_id,)).fetchone()
            if row is None:
                raise KnowledgeError("文档不存在，可能已被删除。", "DOCUMENT_NOT_FOUND", 404)
            return {**self._metadata(row), "text": row["text"][:max_characters],
                    "truncated": len(row["text"]) > max_characters, "content_sha256": row["content_sha256"]}

    def delete_document(self, document_id):
        self.get_document(document_id, 1)
        with closing(self._connect()) as db:
            deleted = db.execute("DELETE FROM knowledge_documents WHERE id=?", (document_id,)).rowcount
            db.commit()
        return bool(deleted)

    def search(self, query, limit=4):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise KnowledgeError("检索问题应为 1 至 4000 个字符。", "INVALID_QUERY")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise KnowledgeError("检索条数应为 1 至 5。", "INVALID_LIMIT")
        terms = list(dict.fromkeys(tokenize(query)))[:64]
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        with closing(self._connect()) as db:
            total, average = db.execute("SELECT COUNT(*),AVG(token_count) FROM knowledge_chunks").fetchone()
            if not total:
                return []
            rows = db.execute(f"SELECT t.chunk_id,t.term,t.tf,c.token_count FROM knowledge_terms t JOIN knowledge_chunks c ON c.id=t.chunk_id WHERE t.term IN ({placeholders})", terms).fetchall()
            frequency = Counter(row["term"] for row in rows)
            scores = Counter()
            for row in rows:
                tf, length, term = row["tf"], row["token_count"], row["term"]
                idf = math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5))
                weight = 0.22 if term.startswith("c1:") else 1.0
                scores[row["chunk_id"]] += weight * idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / max(average, 1)))
            selected = sorted(scores, key=lambda key: (-scores[key], key))[:limit * 4]
            if not selected:
                return []
            marks = ",".join("?" for _ in selected)
            details = {row["chunk_id"]: row for row in db.execute(f"""SELECT c.id AS chunk_id,c.document_id,c.position,c.content,
                d.title,d.filename,d.source_type,d.source_record_id
                FROM knowledge_chunks c JOIN knowledge_documents d ON d.id=c.document_id WHERE c.id IN ({marks})""", selected)}
            results, per_document = [], Counter()
            for chunk_id in selected:
                row = details[chunk_id]
                if per_document[row["document_id"]] >= 2:
                    continue
                per_document[row["document_id"]] += 1
                results.append({**{key: row[key] for key in ("chunk_id", "document_id", "title", "filename", "position", "source_type", "source_record_id")},
                                "excerpt": row["content"], "score": round(scores[chunk_id], 6)})
                if len(results) == limit:
                    break
            return results
