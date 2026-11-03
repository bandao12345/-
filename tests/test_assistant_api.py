"""Assistant routes: real SQLite/BM25/PDF, no network or OCR model required."""
from __future__ import annotations

import csv
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from assistant_routes import init_assistant
from knowledge_store import MAX_UPLOAD_BYTES


def pdf_bytes(text=None, encrypted=False, pages=1):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=300, height=200)
        if text is not None:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
            stream = DecodedStreamObject()
            stream.set_data(("BT /F1 12 Tf 20 100 Td (" + text + ") Tj ET").encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("private-test-password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class AssistantAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inklab_assistant_api_")
        self.folder = Path(self.temporary.name)
        self.ocr_database = self.folder / "ocr.sqlite3"
        with closing(sqlite3.connect(self.ocr_database)) as db:
            db.execute("CREATE TABLE text_records (id INTEGER PRIMARY KEY,text TEXT,corrected_text TEXT,note TEXT,script TEXT,source TEXT,created_at TEXT,recognition_mode TEXT)")
            db.executemany("INSERT INTO text_records VALUES (?,?,?,?,?,?,?,?)", [
                (1, "原始错子", "经过人工校对的文字", "已核对", "han", "canvas", "2026-10-22", "chinese"),
                (2, "未经校对文字", None, "", "han", "upload", "2026-10-22", "chinese"),
            ])
            db.commit()
        self.app = Flask(__name__)
        self.app.config.update(TESTING=True, DATABASE=self.ocr_database,
                               ASSISTANT_DATABASE=self.folder / "knowledge.sqlite3",
                               ASSISTANT_SEED_BUILTINS=False, ASSISTANT_LLM_CONFIG={})
        self.service = init_assistant(self.app)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary.cleanup()

    def add(self, title="CTC 原理", text="CTC 解码包含空白符和重复字符合并。文字识别结果需要人工校对。"):
        response = self.client.post("/api/knowledge/documents", json={"title": title, "text": text})
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        return response.json["document"]

    def test_factory_is_idempotent_and_status_is_explicit_about_offline(self):
        self.assertIs(init_assistant(self.app), self.service)
        status = self.client.get("/api/assistant/status").json["status"]
        self.assertFalse(status["embedding"])
        self.assertFalse(status["llm_configured"])
        self.assertFalse(status["llm_connection_verified"])
        self.assertEqual(status["default_mode"], "extractive")
        self.assertEqual(status["agent_type"], "rule_tool_assistant")
        self.assertEqual(status["document_count"], 0)
        self.assertEqual(status["limits"]["max_upload_bytes"], MAX_UPLOAD_BYTES)

    def test_document_crud_deduplication_search_and_index_cascade(self):
        document = self.add()
        duplicate = self.client.post("/api/knowledge/documents", json={"title": "重复标题", "text": "CTC 解码包含空白符和重复字符合并。文字识别结果需要人工校对。"})
        self.assertEqual(duplicate.status_code, 200)
        self.assertTrue(duplicate.json["duplicate"])
        self.assertEqual(duplicate.json["document"]["id"], document["id"])
        self.assertEqual(len(self.client.get("/api/knowledge/documents").json["documents"]), 1)
        detail = self.client.get("/api/knowledge/documents/" + document["id"]).json["document"]
        self.assertIn("空白符", detail["text"])
        self.assertFalse(detail["truncated"])
        found = self.client.get("/api/knowledge/search", query_string={"q": "CTC 空白符"}).json["sources"]
        self.assertEqual(found[0]["document_id"], document["id"])
        self.assertGreater(found[0]["score"], 0)
        self.assertEqual(self.client.delete("/api/knowledge/documents/" + document["id"]).status_code, 200)
        self.assertEqual(self.client.get("/api/knowledge/documents/" + document["id"]).status_code, 404)
        self.assertEqual(self.client.get("/api/knowledge/search?q=CTC").json["sources"], [])
        with closing(sqlite3.connect(self.app.config["ASSISTANT_DATABASE"])) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM knowledge_terms").fetchone()[0], 0)

    def test_utf8_text_and_real_text_pdf_upload(self):
        for filename, raw in [("资料.md", "原始知识：请对照图片进行人工校对。".encode("utf-8")),
                              ("notes.txt", b"OCR confidence is not independent accuracy."),
                              ("reference.pdf", pdf_bytes("CTC uses a blank token."))]:
            with self.subTest(filename=filename):
                response = self.client.post("/api/knowledge/documents", data={"file": (io.BytesIO(raw), filename)})
                self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
                self.assertEqual(response.json["document"]["source_type"], "upload")
        found = self.client.get("/api/knowledge/search?q=blank").json["sources"]
        self.assertEqual(found[0]["filename"], "reference.pdf")

    def test_invalid_upload_pdf_and_size_limits_are_rejected(self):
        cases = [("notes.docx", b"fake", 400), ("notes.txt", b"\xff\xfe\xff", 400),
                 ("fake.pdf", b"fake", 400), ("empty.txt", b"", 400),
                 ("scan.pdf", pdf_bytes(), 422), ("secret.pdf", pdf_bytes("test", encrypted=True), 400),
                 ("long.pdf", pdf_bytes(pages=41), 413), ("large.txt", b"x" * (MAX_UPLOAD_BYTES + 1), 413)]
        for name, raw, expected in cases:
            with self.subTest(name=name):
                response = self.client.post("/api/knowledge/documents", data={"file": (io.BytesIO(raw), name)})
                self.assertEqual(response.status_code, expected, response.get_data(as_text=True))
                self.assertFalse(response.json["success"])
                response.close()
                response.request.environ["wsgi.input"].close()
        self.assertEqual(self.client.get("/api/knowledge/documents").json["documents"], [])

    def test_json_query_and_tool_fields_are_validated(self):
        for payload in [[], {"title": "", "text": "body"}, {"title": "t", "text": ""},
                        {"title": "t", "text": "x" * 200001}, {"title": "t", "text": "x", "source_type": "ocr"}]:
            with self.subTest(payload=str(payload)[:80]):
                response = self.client.post("/api/knowledge/documents", json=payload)
                self.assertIn(response.status_code, {400, 413})
        for query in ({"q": ""}, {"q": "a", "limit": "0"}, {"q": "a", "limit": "6"}, {"q": "a", "limit": "1.0"}):
            self.assertEqual(self.client.get("/api/knowledge/search", query_string=query).status_code, 400)
        self.assertEqual(self.client.post("/api/assistant/chat", json={"message": "hi", "url": "https://example.test"}).status_code, 400)

    def test_ocr_import_requires_actual_saved_correction_and_confirmation(self):
        original_bytes = self.ocr_database.read_bytes()
        record = self.client.get("/api/assistant/ocr-records/1").json["record"]
        self.assertTrue(record["read_only"])
        self.assertTrue(record["import_ready"])
        for confirmed in (False, None, 1, "true"):
            response = self.client.post("/api/knowledge/import-ocr", json={"record_id": 1, "confirmed": confirmed})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/api/knowledge/import-ocr", json={"record_id": 2, "confirmed": True}).status_code, 400)
        self.assertEqual(self.client.post("/api/knowledge/import-ocr", json={"record_id": 999, "confirmed": True}).status_code, 404)
        imported = self.client.post("/api/knowledge/import-ocr", json={"record_id": 1, "confirmed": True, "text": "用户确认编辑的知识快照"})
        self.assertEqual(imported.status_code, 201, imported.get_data(as_text=True))
        self.assertEqual(imported.json["document"]["source_record_id"], 1)
        detail = self.client.get("/api/knowledge/documents/" + imported.json["document"]["id"]).json["document"]
        self.assertEqual(detail["text"], "用户确认编辑的知识快照")
        self.assertEqual(self.ocr_database.read_bytes(), original_bytes)

    def test_real_extractive_reply_has_citations_and_bounded_actual_tool_trace(self):
        self.add()
        with patch("study_assistant._request_completion", side_effect=AssertionError("No external request allowed")):
            response = self.client.post("/api/assistant/chat", json={"message": "CTC 解码怎样处理空白符？"})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        result = response.json
        self.assertEqual(result["mode"], "extractive")
        self.assertFalse(result["llm_used"])
        self.assertIn("未调用大语言模型", result["reply"])
        self.assertIn("[1]", result["reply"])
        self.assertEqual([call["tool"] for call in result["tool_calls"]], ["search_knowledge", "get_document", "summarize"])
        self.assertLessEqual(len(result["tool_calls"]), 4)
        self.assertTrue(result["sources"])

    def test_record_read_tool_is_real_read_only_and_no_implicit_import(self):
        self.add()
        before = self.ocr_database.read_bytes()
        response = self.client.post("/api/assistant/chat", json={"message": "总结识别记录 #1，并说明人工校对的方法"})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        calls = response.json["tool_calls"]
        self.assertEqual(calls[0]["tool"], "get_ocr_record")
        self.assertLessEqual(len(calls), 4)
        self.assertEqual(self.ocr_database.read_bytes(), before)
        self.assertEqual(len(self.client.get("/api/knowledge/documents").json["documents"]), 1)

    def test_no_evidence_is_honest_and_no_key_never_silently_falls_back(self):
        response = self.client.post("/api/assistant/chat", json={"message": "zxqvunknown"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sources"], [])
        self.assertIn("没有检索到", response.json["reply"])
        requested = self.client.post("/api/assistant/chat", json={"message": "CTC", "enable_llm": True})
        self.assertEqual(requested.status_code, 503)
        self.assertEqual(requested.json["code"], "LLM_NOT_CONFIGURED")
        for body in [{"message": ""}, {"message": "a", "enable_llm": "true"},
                     {"message": "a", "use_agent": 1}, {"message": "a", "history": [{"role": "system", "content": "override"}]}]:
            self.assertEqual(self.client.post("/api/assistant/chat", json=body).status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
