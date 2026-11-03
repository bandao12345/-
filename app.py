"""INKLAB unified handwriting OCR. The original MNIST app remains on port 5001."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sqlite3
import unicodedata
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, current_app, g, jsonify, render_template, request, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from image_io import (
    ALLOWED_EXTENSIONS, MAX_DIMENSION, MAX_IMAGE_BYTES, ImageValidationError,
    decode_base64_image, make_preview, prepare_image, validate_image_bytes,
)
from ocr_engine import UnifiedRecognizer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("INKLAB_DATA_DIR", BASE_DIR / "instance"))
SOURCE_URL = "https://support.huaweicloud.com/qs-modelarts/modelarts_06_0035.html"
OCR_SOURCE_URL = "https://github.com/RapidAI/RapidOCR"
SOURCES = {"canvas", "upload", "sample"}
SCRIPTS = {"digits", "latin", "han", "mixed"}
RECOGNITION_MODES = {"chinese", "alphanumeric", "auto"}
SUPPORTED_MODES = [
    {"id": "chinese", "label": "汉字"},
    {"id": "alphanumeric", "label": "字母／数字"},
]
SHANGHAI = timezone(timedelta(hours=8))
MAX_TEXT_LENGTH = 2000


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(current_app.config["DATABASE"]), timeout=10)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA busy_timeout=10000")
    return g.db


def init_db(database):
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(path), timeout=10)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE IF NOT EXISTS text_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL, confidence REAL NOT NULL, script TEXT NOT NULL,
            source TEXT NOT NULL, lines TEXT NOT NULL, thumbnail TEXT NOT NULL,
            image_width INTEGER NOT NULL, image_height INTEGER NOT NULL,
            engine TEXT NOT NULL, warnings TEXT NOT NULL,
            created_at TEXT NOT NULL, elapsed_ms REAL NOT NULL,
            corrected_text TEXT, note TEXT NOT NULL DEFAULT '',
            recognition_mode TEXT NOT NULL DEFAULT 'auto'
        )""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(text_records)")}
        if "recognition_mode" not in columns:
            # Preserve every existing record and label its historical generic route.
            db.execute("ALTER TABLE text_records ADD COLUMN recognition_mode TEXT NOT NULL DEFAULT 'auto'")
        for name, statement in (("raw_text", "TEXT NOT NULL DEFAULT ''"),
                                ("structure_candidates", "TEXT NOT NULL DEFAULT '[]'")):
            if name not in columns:
                db.execute(f"ALTER TABLE text_records ADD COLUMN {name} {statement}")
        for column in ("created_at", "script", "source"):
            db.execute(f"CREATE INDEX IF NOT EXISTS ix_text_{column} ON text_records({column})")
        db.commit()


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def _samples():
    manifest = _read_json(BASE_DIR / "static" / "samples" / "manifest.json", [])
    result = []
    if not isinstance(manifest, list):
        return result
    for item in manifest:
        if not isinstance(item, dict):
            continue
        sample_id, filename = item.get("id", ""), item.get("filename", "")
        if not isinstance(sample_id, str) or not re.fullmatch(r"[a-z0-9_-]{1,50}", sample_id):
            continue
        if not isinstance(filename, str) or Path(filename).name != filename:
            continue
        if not (BASE_DIR / "static" / "samples" / filename).is_file():
            continue
        result.append({**item, "url": url_for("static", filename=f"samples/{filename}")})
    return result


def _json_body():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("请求内容必须是有效的 JSON 对象")
    return body


def _extract_image():
    body = _json_body() if request.is_json else None
    if body is not None:
        recognition_mode = body.get("recognition_mode", "chinese")
    else:
        if "recognition_mode" in request.files:
            raise ValueError("识别模式必须为文字字段")
        modes = request.form.getlist("recognition_mode")
        if len(modes) > 1:
            raise ValueError("识别模式只能指定一次")
        recognition_mode = request.form.get("recognition_mode", "chinese")
    if not isinstance(recognition_mode, str) or recognition_mode not in RECOGNITION_MODES:
        raise ValueError("识别模式无效，请选择汉字或字母／数字模式")
    if body is not None and "sample" in body:
        sample_id = body["sample"]
        if not isinstance(sample_id, str) or not re.fullmatch(r"[a-z0-9_-]{1,50}", sample_id):
            raise ValueError("样例编号无效")
        sample = next((item for item in _samples() if item["id"] == sample_id), None)
        if sample is None:
            raise ValueError("该样例不存在")
        image = validate_image_bytes((BASE_DIR / "static" / "samples" / sample["filename"]).read_bytes())
        return prepare_image(image), "sample", recognition_mode
    if body is not None:
        image = decode_base64_image(body.get("image"))
        source = body.get("source", "canvas")
    else:
        uploaded = request.files.get("image") or request.files.get("file")
        if uploaded is None:
            raise ValueError("请提供画板图片或上传文件")
        suffix = Path(uploaded.filename or "").suffix.lower().lstrip(".")
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("请选择 PNG、JPEG 或 WEBP 图片")
        image = validate_image_bytes(uploaded.stream.read(MAX_IMAGE_BYTES + 1))
        source = request.form.get("source", "upload")
    if not isinstance(source, str) or source not in SOURCES or source == "sample":
        raise ValueError("图片来源必须为 canvas 或 upload")
    return prepare_image(image), source, recognition_mode


def _filters():
    conditions, values = [], []
    query = request.args.get("q", "")
    if len(query) > 100:
        raise ValueError("搜索内容不能超过 100 个字符")
    if query:
        conditions.append("(instr(lower(text),lower(?))>0 OR instr(lower(coalesce(corrected_text,'')),lower(?))>0 OR instr(lower(note),lower(?))>0)")
        values.extend([query] * 3)
    for key, valid in (("script", SCRIPTS), ("source", SOURCES)):
        value = request.args.get(key, "")
        if value:
            if value not in valid:
                raise ValueError("文字类别或输入来源无效")
            conditions.append(f"{key}=?")
            values.append(value)
    dates = {}
    for key, operator in (("date_from", ">="), ("date_to", "<=")):
        value = request.args.get(key, "")
        if value:
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError()
            except ValueError:
                raise ValueError("日期格式必须为 YYYY-MM-DD") from None
            dates[key] = value
            conditions.append(f"substr(created_at,1,10) {operator} ?")
            values.append(value)
    if dates.get("date_from", "") > dates.get("date_to", "9999-12-31"):
        raise ValueError("开始日期不能晚于结束日期")
    return (" WHERE " + " AND ".join(conditions) if conditions else ""), values


def _page_argument(name, default, maximum=None):
    value = request.args.get(name, str(default))
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} 参数无效")
    number = int(value)
    if number < 1 or (maximum is not None and number > maximum):
        raise ValueError(f"{name} 参数超出范围")
    return number


def _record(row):
    result = dict(row)
    result["lines"] = json.loads(result["lines"])
    result["warnings"] = json.loads(result["warnings"])
    result["preview"] = result["thumbnail"]
    result["raw_text"] = result.get("raw_text") or result["text"]
    result["structure_candidates"] = json.loads(result.get("structure_candidates") or "[]")
    return result


def _classify(text):
    types = set()
    for char in unicodedata.normalize("NFKC", text):
        if char.isspace() or unicodedata.category(char).startswith("P"):
            continue
        if char.isdecimal():
            types.add("digits")
        elif "\u3400" <= char <= "\u9fff" or "CJK" in unicodedata.name(char, ""):
            types.add("han")
        elif "LATIN" in unicodedata.name(char, ""):
            types.add("latin")
        else:
            types.add("mixed")
    return next(iter(types)) if len(types) == 1 else "mixed"


def _prediction_result(result, recognition_mode):
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise RuntimeError("Invalid OCR output")
    text = result["text"].strip()
    if not text:
        label = {"chinese": "汉字", "alphanumeric": "字母／数字", "auto": "兼容自动"}[recognition_mode]
        raise ValueError(f"当前{label}模式未识别出文字，请写得更大、更清晰并留出字符间距，或切换另一识别模式")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError("图片文字超过 2000 字，请分批识别")
    confidence = float(result.get("confidence", 0))
    elapsed = float(result.get("elapsed_ms", 0))
    if not math.isfinite(confidence) or not 0 <= confidence <= 1 or not math.isfinite(elapsed):
        raise RuntimeError("Invalid OCR confidence or timing")
    actual_mode = result.get("recognition_mode", recognition_mode)
    if not isinstance(actual_mode, str) or actual_mode not in RECOGNITION_MODES:
        raise RuntimeError("Invalid OCR recognition mode")
    if recognition_mode != "auto" and actual_mode != recognition_mode:
        raise RuntimeError("OCR returned a different recognition mode")
    return {
        "text": text, "confidence": confidence, "script": _classify(text),
        "lines": result.get("lines", []), "elapsed_ms": max(0, elapsed),
        "warnings": result.get("warnings", []), "engine": str(result.get("engine", "RapidOCR")),
        "recognition_mode": actual_mode,
        "raw_text": str(result.get("raw_text", text))[:MAX_TEXT_LENGTH],
        "structure_candidates": result.get("structure_candidates", []),
    }


def create_app(test_config=None):
    app = Flask(__name__, static_folder=str(BASE_DIR / "static"), template_folder=str(BASE_DIR / "templates"))
    app.config.from_mapping(
        DATABASE=DATA_DIR / "inklab_ocr.db",
        ASSISTANT_DATABASE=DATA_DIR / "assistant.sqlite3",
        DEMO_ACCESS_TOKEN=os.environ.get("INKLAB_DEMO_TOKEN", ""),
        OCR_MODEL_DIR=BASE_DIR / "models" / "ocr-v6",
        OCR_VALIDATION_PATH=BASE_DIR / "docs" / "ocr_validation.json",
        LEGACY_METRICS_PATH=BASE_DIR / "models" / "mnist_baseline_metrics.json",
        MAX_CONTENT_LENGTH=3 * 1024 * 1024,
        MAX_FORM_MEMORY_SIZE=3 * 1024 * 1024, MAX_FORM_PARTS=8,
    )
    if test_config:
        app.config.update(test_config)
        if "DATABASE" in test_config and "ASSISTANT_DATABASE" not in test_config:
            app.config["ASSISTANT_DATABASE"] = Path(app.config["DATABASE"]).with_name("assistant.sqlite3")
    app.json.ensure_ascii = False
    app.json.sort_keys = False
    app.extensions["recognizer"] = UnifiedRecognizer(model_dir=Path(app.config["OCR_MODEL_DIR"]))
    init_db(app.config["DATABASE"])
    from assistant_routes import init_assistant
    init_assistant(app)
    from demo_access import install_demo_access
    install_demo_access(app)

    @app.teardown_appcontext
    def close_db(_error):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.before_request
    def same_origin_writes():
        if request.method in {"POST", "PATCH", "DELETE", "PUT"}:
            origin = request.headers.get("Origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                    allowed = parsed.scheme in {"http", "https"} and parsed.netloc.lower() == request.host.lower()
                except ValueError:
                    allowed = False
                if not allowed:
                    return jsonify(success=False, error="请从本应用页面提交操作"), 403

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(ValueError)
    def invalid_value(error):
        return jsonify(success=False, error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {400: "请求格式无效", 404: "请求的资源不存在", 405: "不支持此请求方法"}
        message = "请求过大，请选择不超过 2 MB 的图片" if isinstance(error, RequestEntityTooLarge) else messages.get(error.code, "请求未能完成")
        return jsonify(success=False, error=message), error.code

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        status = app.extensions["recognizer"].status()
        return jsonify(status="ok", ready=status["ready"], engine=status.get("engine"), device="cpu")

    @app.get("/api/info")
    @app.get("/api/status")
    def info():
        return jsonify(success=True, info={
            **app.extensions["recognizer"].status(),
            "name": "墨迹 INKLAB 手写资料学习助手", "version": "3.0.0",
            "features": ["ocr", "knowledge_base", "bm25_retrieval", "tool_assistant", "mcp", "project_skill"],
            "device": "cpu", "auto_recognition": False,
            "supported_modes": SUPPORTED_MODES,
            "limits": {"max_image_bytes": MAX_IMAGE_BYTES, "max_dimension": MAX_DIMENSION},
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
            "legacy_url": "http://127.0.0.1:5001/", "source_url": SOURCE_URL,
            "ocr_source_url": OCR_SOURCE_URL, "samples": _samples(),
        })

    @app.get("/api/samples")
    def samples():
        return jsonify(success=True, samples=_samples())

    @app.get("/api/metrics")
    def metrics():
        baseline = _read_json(app.config["LEGACY_METRICS_PATH"], {})
        if isinstance(baseline, dict):
            baseline.pop("model_path", None)
        return jsonify(success=True, metrics={
            "legacy_mnist": baseline,
            "ocr": {
                "engine": app.extensions["recognizer"].status().get("engine", "RapidOCR"),
                "validation": _read_json(app.config["OCR_VALIDATION_PATH"], {}),
                "note": "通用 OCR 使用预训练模型，未在本项目训练；MNIST 的 98.92% 仅属于保留的原单数字模型，不代表汉字、英文或多字识别准确率。",
                "sources": [OCR_SOURCE_URL, "https://github.com/PaddlePaddle/PaddleOCR"],
            },
        })


    @app.post("/api/predict")
    def predict():
        image, source, recognition_mode = _extract_image()
        recognizer = app.extensions["recognizer"]
        if not recognizer.status().get("ready"):
            return jsonify(success=False, error="文字识别模型暂不可用，请检查模型文件和依赖"), 503
        try:
            result = _prediction_result(recognizer.predict(image, recognition_mode=recognition_mode), recognition_mode)
        except ValueError:
            raise
        except Exception as error:
            app.logger.warning("OCR unavailable: %s", type(error).__name__)
            return jsonify(success=False, error="文字识别引擎暂不可用，请检查依赖或稍后重试"), 503
        preview = make_preview(image)
        created_at = datetime.now(SHANGHAI).isoformat(timespec="seconds")
        db = get_db()
        cursor = db.execute("""INSERT INTO text_records
            (text,confidence,script,source,lines,thumbnail,image_width,image_height,
             engine,warnings,created_at,elapsed_ms,recognition_mode,raw_text,structure_candidates)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                result["text"], result["confidence"], result["script"], source,
                json.dumps(result["lines"], ensure_ascii=False), preview, image.width, image.height,
                result["engine"], json.dumps(result["warnings"], ensure_ascii=False),
                created_at, result["elapsed_ms"], result["recognition_mode"],
                result["raw_text"], json.dumps(result["structure_candidates"], ensure_ascii=False),
            ))
        db.commit()
        return jsonify(
            success=True, **result, preview=preview, thumbnail=preview,
            image_width=image.width, image_height=image.height,
            record_id=cursor.lastrowid, source=source, created_at=created_at,
        )

    @app.get("/api/history")
    def history():
        where, values = _filters()
        page = _page_argument("page", 1)
        per_page = _page_argument("per_page", 10, 100)
        db = get_db()
        total = db.execute("SELECT COUNT(*) FROM text_records" + where, values).fetchone()[0]
        rows = db.execute("SELECT * FROM text_records" + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
                          [*values, per_page, (page - 1) * per_page]).fetchall()
        return jsonify(success=True, records=[_record(row) for row in rows], total=total,
                       page=page, per_page=per_page, pages=max(1, math.ceil(total / per_page)))

    @app.get("/api/history/<int:record_id>")
    def detail(record_id):
        record = get_db().execute("SELECT * FROM text_records WHERE id=?", (record_id,)).fetchone()
        if record is None:
            return jsonify(success=False, error="该识别记录不存在"), 404
        return jsonify(success=True, record=_record(record))

    @app.patch("/api/history/<int:record_id>")
    def update_record(record_id):
        body = _json_body()
        if not body or set(body) - {"corrected_text", "note"}:
            raise ValueError("仅支持修改人工校对文字和备注")
        changes, values = [], []
        if "corrected_text" in body:
            corrected = body["corrected_text"]
            if corrected is not None and (not isinstance(corrected, str) or len(corrected) > MAX_TEXT_LENGTH):
                raise ValueError("校对内容必须是 2000 字以内的文字或 null")
            changes.append("corrected_text=?")
            values.append(corrected.strip() if corrected is not None else None)
        if "note" in body:
            note = body["note"]
            if not isinstance(note, str) or len(note) > 300:
                raise ValueError("备注必须是 300 字以内的文本")
            changes.append("note=?")
            values.append(note.strip())
        db = get_db()
        cursor = db.execute("UPDATE text_records SET " + ",".join(changes) + " WHERE id=?",
                            [*values, record_id])
        db.commit()
        if cursor.rowcount == 0:
            return jsonify(success=False, error="该识别记录不存在"), 404
        return detail(record_id)

    @app.delete("/api/history/<int:record_id>")
    def delete_record(record_id):
        db = get_db()
        cursor = db.execute("DELETE FROM text_records WHERE id=?", (record_id,))
        db.commit()
        if cursor.rowcount == 0:
            return jsonify(success=False, error="该识别记录不存在"), 404
        return jsonify(success=True, deleted_id=record_id)

    @app.delete("/api/history")
    def clear_history():
        if _json_body().get("confirm") != "DELETE_ALL":
            raise ValueError("清空操作需要明确确认")
        db = get_db()
        cursor = db.execute("DELETE FROM text_records")
        db.commit()
        return jsonify(success=True, deleted_count=cursor.rowcount)

    @app.get("/api/stats")
    def stats():
        where, values = _filters()
        db = get_db()
        summary = db.execute("""SELECT COUNT(*) AS total, AVG(confidence) AS mean_confidence,
            AVG(elapsed_ms) AS mean_elapsed_ms,
            SUM(CASE WHEN corrected_text IS NOT NULL OR note!='' THEN 1 ELSE 0 END) AS feedback_count,
            SUM(length(text)) AS total_chars FROM text_records""" + where, values).fetchone()
        scripts = dict(db.execute("SELECT script,COUNT(*) FROM text_records" + where + " GROUP BY script", values))
        sources = dict(db.execute("SELECT source,COUNT(*) FROM text_records" + where + " GROUP BY source", values))
        daily = db.execute("SELECT substr(created_at,1,10) AS day,COUNT(*) AS count FROM text_records" +
                           where + " GROUP BY day ORDER BY day DESC LIMIT 14", values).fetchall()
        return jsonify(success=True, stats={
            "total": summary["total"], "mean_confidence": summary["mean_confidence"] or 0,
            "mean_elapsed_ms": summary["mean_elapsed_ms"] or 0,
            "feedback_count": summary["feedback_count"] or 0, "total_chars": summary["total_chars"] or 0,
            "by_script": [{"script": script, "count": scripts.get(script, 0)}
                          for script in ("digits", "latin", "han", "mixed")],
            "by_source": [{"source": source, "count": sources.get(source, 0)}
                          for source in ("canvas", "upload", "sample")],
            "daily": [{"date": row["day"], "count": row["count"]} for row in reversed(daily)],
        })

    @app.get("/api/export/csv")
    @app.get("/api/history/export")
    def export_csv():
        where, values = _filters()
        rows = get_db().execute("SELECT * FROM text_records" + where + " ORDER BY id DESC", values).fetchall()
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["ID", "识别文字", "文字类别", "置信度", "来源", "识别时间",
                         "推理耗时ms", "人工校对文字", "备注", "引擎", "识别模式"])
        def cell(value):
            text = str(value if value is not None else "")
            return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else text
        for row in rows:
            writer.writerow([
                row["id"], cell(row["text"]), row["script"], f"{row['confidence']:.6f}", row["source"],
                row["created_at"], row["elapsed_ms"], cell(row["corrected_text"]), cell(row["note"]),
                cell(row["engine"]), row["recognition_mode"],
            ])
        return Response("\ufeff" + stream.getvalue(), content_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=inklab_text_records.csv"})

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "5003")), debug=False)

