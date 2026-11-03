"""Optional isolated-demo access gate, enabled only with an explicit token."""
import hmac
from collections import defaultdict, deque
from threading import Lock
import time
from flask import jsonify, redirect, request, make_response


def install_demo_access(app):
    token = app.config.get("DEMO_ACCESS_TOKEN", "")
    if not token:
        return
    if not isinstance(token, str) or len(token) < 24:
        raise ValueError("A demo token must contain at least 24 characters")
    attempts = defaultdict(deque)
    lock = Lock()

    @app.get("/demo-login")
    def demo_login():
        supplied = request.args.get("token", "")
        if not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
            return "演示入口无效，请使用项目提供的完整演示链接。", 401
        response = make_response(redirect("/", code=303))
        response.set_cookie("inklab_demo", token, max_age=8 * 3600, httponly=True,
                            samesite="Lax", secure=request.headers.get("X-Forwarded-Proto") == "https" or request.is_secure)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.before_request
    def check_demo_access():
        if request.path in {"/health", "/demo-login"}:
            return None
        supplied = request.cookies.get("inklab_demo", "")
        if not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
            if request.path.startswith("/api/"):
                return jsonify(success=False, error="请先从完整演示链接进入。"), 401
            return "请使用带授权入口的演示链接访问本项目。", 401
        if request.method in {"POST", "PATCH", "DELETE", "PUT"}:
            # Shared, bounded demo workspace: avoid exhausting local inference.
            key = request.remote_addr or "demo"
            now = time.monotonic()
            with lock:
                queue = attempts[key]
                while queue and queue[0] < now - 60:
                    queue.popleft()
                if len(queue) >= 20:
                    return jsonify(success=False, error="演示操作较频繁，请稍后再试。"), 429
                queue.append(now)
        return None
