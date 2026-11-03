"""墨迹 INKLAB：Flask 与 SQLite 全栈应用。
基于 CodeArts 初版，由 Codex 协作完善；原实验来源见 SOURCE_URL。
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, current_app, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from inference import (
    ALLOWED_EXTENSIONS, MAX_DIMENSION, MAX_IMAGE_BYTES, MODEL_PATH,
    DigitRecognizer, ImageValidationError, decode_base64_image, validate_image_bytes,
)

BASE_DIR = Path(__file__).resolve().parent
SOURCE_URL = "https://support.huaweicloud.com/qs-modelarts/modelarts_06_0035.html"
SOURCES = {"canvas", "upload", "sample"}
SHANGHAI = timezone(timedelta(hours=8))


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(current_app.config["DATABASE"]), timeout=10)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA busy_timeout=10000")
    return g.db


def init_db(database):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(database), timeout=10)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE IF NOT EXISTS recognition_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            digit INTEGER NOT NULL, confidence REAL NOT NULL,
            probabilities TEXT NOT NULL, source TEXT NOT NULL,
            thumbnail TEXT, created_at TEXT NOT NULL,
            elapsed_ms REAL NOT NULL DEFAULT 0,
            corrected_digit INTEGER, note TEXT NOT NULL DEFAULT ''
        )""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(recognition_records)")}
        additions = {
            "elapsed_ms": "REAL NOT NULL DEFAULT 0",
            "corrected_digit": "INTEGER",
            "note": "TEXT NOT NULL DEFAULT ''",
        }
        for column, definition in additions.items():
            if column not in columns:
                db.execute(f"ALTER TABLE recognition_records ADD COLUMN {column} {definition}")
        for column in ("created_at", "digit", "source"):
            db.execute(f"CREATE INDEX IF NOT EXISTS ix_records_{column} ON recognition_records({column})")
        db.commit()


def _metrics():
    path = Path(current_app.config["METRICS_PATH"])
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    result.pop("model_path", None)
    return result


def _digit(value, name="数字"):
    if isinstance(value, bool):
        raise ValueError(f"{name}必须为 0 至 9 的整数")
    if isinstance(value, int) and 0 <= value <= 9:
        return value
    if isinstance(value, str) and value in tuple(str(i) for i in range(10)):
        return int(value)
    raise ValueError(f"{name}必须为 0 至 9 的整数")


def _json_body():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("请求内容必须是有效的 JSON 对象")
    return body


def _filters():
    conditions, values = [], []
    digit = request.args.get("digit", "")
    if digit != "":
        conditions.append("digit = ?")
        values.append(_digit(digit, "筛选数字"))
    source = request.args.get("source", "")
    if source:
        if source not in SOURCES:
            raise ValueError("来源必须是 canvas、upload 或 sample")
        conditions.append("source = ?")
        values.append(source)
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
            conditions.append(f"substr(created_at, 1, 10) {operator} ?")
            values.append(value)
    if dates.get("date_from", "") > dates.get("date_to", "9999-12-31"):
        raise ValueError("开始日期不能晚于结束日期")
    return (" WHERE " + " AND ".join(conditions) if conditions else ""), values


def _page_argument(name, default, maximum=None):
    text = request.args.get(name, str(default))
    try:
        if not text.isascii() or not text.isdecimal():
            raise ValueError()
        value = int(text)
        if value < 1 or (maximum is not None and value > maximum):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(f"{name} 参数超出有效范围") from None
    return value


def _record(row):
    result = dict(row)
    result["probabilities"] = json.loads(result["probabilities"])
    return result


def _extract_image():
    body = _json_body() if request.is_json else None
    sample, has_sample = None, False
    if body is not None and "sample" in body:
        sample, has_sample = body["sample"], True
    elif "sample" in request.form:
        sample, has_sample = request.form["sample"], True
    elif "sample" in request.args:
        sample, has_sample = request.args["sample"], True
    if has_sample:
        digit = _digit(sample, "样例编号")
        path = BASE_DIR / "static" / "samples" / f"{digit}.png"
        if not path.is_file():
            raise ImageValidationError("样例图片不存在")
        return validate_image_bytes(path.read_bytes()), "sample"
    if body is not None:
        if "image" not in body:
            raise ValueError("请提供 image 图片数据或 sample 样例编号")
        source = body.get("source", "canvas")
        image = decode_base64_image(body["image"])
    else:
        upload = request.files.get("image")
        if upload is None:
            upload = request.files.get("file")
        if upload is None:
            raise ValueError("请上传图片或提供有效的 JSON 图片数据")
        image = validate_image_bytes(upload.read(MAX_IMAGE_BYTES + 1))
        source = request.form.get("source", "upload")
    if not isinstance(source, str) or source not in SOURCES:
        raise ValueError("图片来源无效")
    return image, source


def create_app(test_config=None):
    app = Flask(__name__, static_folder=str(BASE_DIR / "static"),
                template_folder=str(BASE_DIR / "templates"))
    app.config.from_mapping(
        DATABASE=BASE_DIR / "instance" / "inklab.db",
        MODEL_PATH=MODEL_PATH, METRICS_PATH=BASE_DIR / "metrics.json",
        MAX_CONTENT_LENGTH=3 * 1024 * 1024,
        MAX_FORM_MEMORY_SIZE=3 * 1024 * 1024, MAX_FORM_PARTS=8,
    )
    if test_config:
        app.config.update(test_config)
    app.json.ensure_ascii = False
    app.json.sort_keys = False
    app.extensions["recognizer"] = DigitRecognizer(model_path=Path(app.config["MODEL_PATH"]))
    init_db(app.config["DATABASE"])

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
                    parts = urlsplit(origin)
                    permitted = parts.scheme in {"http", "https"} and parts.netloc.lower() == request.host.lower()
                except ValueError:
                    permitted = False
                if not permitted:
                    return jsonify(success=False, error="请从本应用页面提交操作"), 403

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(ValueError)
    def invalid_value(error):
        return jsonify(success=False, error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        if isinstance(error, RequestEntityTooLarge):
            message = "请求过大，请上传不超过 2MB 的图片"
        else:
            messages = {400: "请求格式无效", 404: "请求的资源不存在", 405: "不支持此请求方法"}
            message = messages.get(error.code, "请求未能完成")
        return jsonify(success=False, error=message), error.code

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        status = app.extensions["recognizer"].status()
        return jsonify(status="ok", ready=status["ready"], model_exists=status["model_exists"])

    @app.get("/api/info")
    @app.get("/api/status")
    def info():
        status = app.extensions["recognizer"].status()
        samples = [{"digit": i, "url": f"/static/samples/{i}.png"} for i in range(10)
                   if (BASE_DIR / "static" / "samples" / f"{i}.png").is_file()]
        return jsonify(success=True, info={
            "name": "墨迹 INKLAB 手写数字识别", "version": "1.0.0",
            "source_url": SOURCE_URL, **status,
            "metrics_available": Path(app.config["METRICS_PATH"]).is_file(),
            "limits": {"max_image_bytes": MAX_IMAGE_BYTES, "max_dimension": MAX_DIMENSION},
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS), "samples": samples,
        })

    @app.get("/api/metrics")
    def metrics():
        result = _metrics()
        if result is None:
            return jsonify(success=False, error="暂无训练评估文件，请先完成训练"), 404
        return jsonify(success=True, metrics=result)

    @app.get("/api/samples")
    def samples():
        result = [{"digit": i, "url": f"/static/samples/{i}.png"} for i in range(10)
                  if (BASE_DIR / "static" / "samples" / f"{i}.png").is_file()]
        return jsonify(success=True, samples=result)

    @app.post("/api/predict")
    def recognize():
        image, source = _extract_image()
        try:
            result = app.extensions["recognizer"].predict(image)
        except ImageValidationError:
            raise
        except (FileNotFoundError, RuntimeError) as error:
            app.logger.warning("识别服务暂不可用: %s", type(error).__name__)
            return jsonify(success=False, error="模型不可用，请检查模型文件和依赖环境"), 503
        created_at = datetime.now(SHANGHAI).isoformat(timespec="seconds")
        thumbnail = result["preview"]
        db = get_db()
        cursor = db.execute("""INSERT INTO recognition_records
            (digit,confidence,probabilities,source,thumbnail,created_at,elapsed_ms)
            VALUES (?,?,?,?,?,?,?)""", (
                result["digit"], result["confidence"], json.dumps(result["probabilities"]),
                source, thumbnail, created_at, result["elapsed_ms"],
            ))
        db.commit()
        return jsonify(success=True, **result, record_id=cursor.lastrowid,
                       thumbnail=thumbnail, source=source, created_at=created_at)

    @app.get("/api/history")
    def history():
        where, values = _filters()
        page, per_page = _page_argument("page", 1), _page_argument("per_page", 10, 100)
        db = get_db()
        total = db.execute("SELECT COUNT(*) FROM recognition_records" + where, values).fetchone()[0]
        rows = db.execute("SELECT * FROM recognition_records" + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
                          [*values, per_page, (page - 1) * per_page]).fetchall()
        return jsonify(success=True, records=[_record(row) for row in rows], total=total,
                       page=page, per_page=per_page, pages=max(1, math.ceil(total / per_page)))

    @app.get("/api/history/<int:record_id>")
    def detail(record_id):
        row = get_db().execute("SELECT * FROM recognition_records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            return jsonify(success=False, error="该识别记录不存在"), 404
        return jsonify(success=True, record=_record(row))

    @app.patch("/api/history/<int:record_id>")
    def update_record(record_id):
        body = _json_body()
        if not body or set(body) - {"corrected_digit", "note"}:
            raise ValueError("仅支持修改人工纠错数字和备注")
        changes, values = [], []
        if "corrected_digit" in body:
            value = body["corrected_digit"]
            changes.append("corrected_digit=?")
            values.append(None if value is None else _digit(value, "人工纠错数字"))
        if "note" in body:
            value = body["note"]
            if not isinstance(value, str) or len(value) > 300:
                raise ValueError("备注必须是 300 字以内的文本")
            changes.append("note=?")
            values.append(value.strip())
        db = get_db()
        cursor = db.execute("UPDATE recognition_records SET " + ",".join(changes) + " WHERE id=?",
                            [*values, record_id])
        if cursor.rowcount == 0:
            return jsonify(success=False, error="该识别记录不存在"), 404
        db.commit()
        return detail(record_id)

    @app.delete("/api/history/<int:record_id>")
    def delete_record(record_id):
        db = get_db()
        cursor = db.execute("DELETE FROM recognition_records WHERE id=?", (record_id,))
        db.commit()
        if cursor.rowcount == 0:
            return jsonify(success=False, error="该识别记录不存在"), 404
        return jsonify(success=True, deleted_id=record_id)

    @app.delete("/api/history")
    def clear_history():
        if _json_body().get("confirm") != "DELETE_ALL":
            raise ValueError("清空操作需要明确确认")
        db = get_db()
        cursor = db.execute("DELETE FROM recognition_records")
        db.commit()
        return jsonify(success=True, deleted_count=cursor.rowcount)

    @app.get("/api/stats")
    def statistics():
        where, values = _filters()
        db = get_db()
        summary = db.execute("""SELECT COUNT(*) AS total, AVG(confidence) AS mean_confidence,
            AVG(elapsed_ms) AS mean_elapsed_ms,
            SUM(CASE WHEN corrected_digit IS NOT NULL OR note != '' THEN 1 ELSE 0 END) AS feedback_count
            FROM recognition_records""" + where, values).fetchone()
        digits = dict(db.execute("SELECT digit,COUNT(*) FROM recognition_records" + where + " GROUP BY digit", values))
        sources = dict(db.execute("SELECT source,COUNT(*) FROM recognition_records" + where + " GROUP BY source", values))
        daily = db.execute("SELECT substr(created_at,1,10) AS day,COUNT(*) AS count FROM recognition_records" +
                           where + " GROUP BY day ORDER BY day DESC LIMIT 14", values).fetchall()
        return jsonify(success=True, stats={
            "total": summary["total"], "mean_confidence": summary["mean_confidence"] or 0,
            "mean_elapsed_ms": summary["mean_elapsed_ms"] or 0,
            "feedback_count": summary["feedback_count"] or 0,
            "by_digit": [{"digit": i, "count": digits.get(i, 0)} for i in range(10)],
            "by_source": [{"source": source, "count": sources.get(source, 0)}
                          for source in ("canvas", "upload", "sample")],
            "daily": [{"date": row["day"], "count": row["count"]} for row in reversed(daily)],
        })

    @app.get("/api/export/csv")
    @app.get("/api/history/export")
    def export_csv():
        where, values = _filters()
        rows = get_db().execute("SELECT * FROM recognition_records" + where + " ORDER BY id DESC", values).fetchall()
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["ID", "预测数字", "置信度", "来源", "创建时间", "推理耗时ms", "人工纠错", "备注", "概率分布"])
        for row in rows:
            note = row["note"]
            if note.startswith(("=", "+", "-", "@", "\t", "\r")):
                note = "'" + note
            writer.writerow([
                row["id"], row["digit"], f"{row['confidence']:.6f}", row["source"], row["created_at"],
                row["elapsed_ms"], "" if row["corrected_digit"] is None else row["corrected_digit"], note,
                ";".join(f"{p:.6f}" for p in json.loads(row["probabilities"])),
            ])
        return Response("\ufeff" + stream.getvalue(), content_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=inklab_history.csv"})

    @app.post("/api/model/reload")
    def reload_model():
        app.extensions["recognizer"] = DigitRecognizer(model_path=Path(app.config["MODEL_PATH"]))
        status = app.extensions["recognizer"].status()
        return jsonify(success=status["ready"], info=status), 200 if status["ready"] else 503

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "5001")), debug=False)

