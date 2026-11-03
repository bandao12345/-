"""Assistant Blueprint factory. Importing this module creates no files/services."""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from knowledge_store import KnowledgeError, KnowledgeStore, MAX_UPLOAD_BYTES, RETRIEVAL_METHOD, read_uploaded_document
from study_assistant import StudyAssistant


def _body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise KnowledgeError("请发送有效的 JSON 对象。", "INVALID_JSON")
    return data


def init_assistant(app):
    if "study_assistant" in app.extensions:
        return app.extensions["study_assistant"]
    root = Path(app.root_path)
    database = app.config.get("ASSISTANT_DATABASE", root / "instance" / "assistant.sqlite3")
    store = KnowledgeStore(database, app.config.get("ASSISTANT_KNOWLEDGE_DIR", root / "knowledge"),
                           seed_builtin=app.config.get("ASSISTANT_SEED_BUILTINS", True))
    assistant = StudyAssistant(store, app.config.get("DATABASE"), app.config.get("ASSISTANT_LLM_CONFIG"))
    app.extensions["knowledge_store"] = store
    app.extensions["study_assistant"] = assistant
    bp = Blueprint("study_assistant", __name__)

    @bp.errorhandler(KnowledgeError)
    def knowledge_error(error):
        return jsonify(success=False, error=str(error), code=error.code), error.status

    @bp.get("/api/assistant/status")
    def status():
        return jsonify(success=True, status=assistant.status())

    @bp.get("/api/knowledge/documents")
    def documents():
        return jsonify(success=True, documents=store.list_documents(), retrieval_method=RETRIEVAL_METHOD)

    @bp.post("/api/knowledge/documents")
    def add_document():
        if request.is_json:
            data = _body()
            if set(data) - {"title", "text"}:
                raise KnowledgeError("文本导入仅接收 title 和 text；OCR 导入请使用确认接口。", "INVALID_DOCUMENT_FIELDS")
            document = store.add_document(data.get("title"), data.get("text"))
        else:
            file = request.files.get("file")
            if file is None:
                raise KnowledgeError("请选择 txt、md 或纯文本 PDF 文件。", "NO_DOCUMENT")
            filename, text = read_uploaded_document(file.filename, file.stream.read(MAX_UPLOAD_BYTES + 1))
            title = request.form.get("title", "").strip() or Path(filename).stem
            document = store.add_document(title, text, filename, "upload")
        return jsonify(success=True, document=document, duplicate=document["duplicate"]), 200 if document["duplicate"] else 201

    @bp.get("/api/knowledge/documents/<document_id>")
    def document(document_id):
        return jsonify(success=True, document=store.get_document(document_id))

    @bp.delete("/api/knowledge/documents/<document_id>")
    def delete_document(document_id):
        store.delete_document(document_id)
        return jsonify(success=True, deleted=True, id=document_id)

    @bp.get("/api/knowledge/search")
    def search():
        raw_limit = request.args.get("limit", "4")
        if not raw_limit.isascii() or not raw_limit.isdecimal():
            raise KnowledgeError("检索条数应为 1 至 5。", "INVALID_LIMIT")
        sources = assistant.search_knowledge(request.args.get("q", ""), int(raw_limit))
        return jsonify(success=True, sources=sources, retrieval_method=RETRIEVAL_METHOD)

    @bp.get("/api/assistant/ocr-records/<int:record_id>")
    def ocr_record(record_id):
        return jsonify(success=True, record=assistant.get_ocr_record(record_id))

    @bp.post("/api/knowledge/import-ocr")
    def import_ocr():
        data = _body()
        if set(data) - {"record_id", "confirmed", "title", "text"}:
            raise KnowledgeError("OCR 导入字段无效。", "INVALID_DOCUMENT_FIELDS")
        document = assistant.import_ocr_record(data.get("record_id"), data.get("confirmed"), data.get("title"), data.get("text"))
        return jsonify(success=True, document=document, duplicate=document["duplicate"]), 200 if document["duplicate"] else 201

    @bp.post("/api/assistant/chat")
    def chat():
        data = _body()
        if set(data) - {"message", "enable_llm", "use_agent", "history"}:
            raise KnowledgeError("助手请求字段无效，不接受自定义接口地址或任意工具。", "INVALID_ASSISTANT_FIELDS")
        result = assistant.chat(data.get("message"), data.get("enable_llm", False), data.get("use_agent", True), data.get("history"))
        return jsonify(success=True, **result)

    app.register_blueprint(bp)
    return assistant
