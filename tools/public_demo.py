"""One-click isolated ngrok demonstration; reuse verified sessions safely.

No account token is read, written or requested. Only the random demonstration
access link is saved under gitignored instance/public-demo for sharing.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPCookieProcessor, HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "instance" / "public-demo"
SESSION = DATA / "session.json"
LINK_FILE = DATA / "演示地址.txt"
PORT = 5005
LOCAL = f"http://127.0.0.1:{PORT}"
INSPECTOR = "http://127.0.0.1:4040/api/tunnels"


class DemoError(Exception):
    pass


def _stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path):
    try:
        if path.stat().st_size > 16384:
            return None
        result = json.loads(path.read_text("utf-8-sig"))
        return result if type(result) is dict else None
    except (OSError, ValueError):
        return None


def _valid_session(session):
    return (isinstance(session, dict) and type(session.get("token")) is str
            and 24 <= len(session["token"]) <= 512
            and session.get("demo_port", PORT) == PORT)


def _fetch_json(url, timeout=4):
    with build_opener(ProxyHandler({})).open(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
        raw = response.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("Response too large")
    return json.loads(raw.decode("utf-8"))


def _port_in_use(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.4):
            return True
    except OSError:
        return False


def _forwarded_to_demo(addr):
    if addr in {str(PORT), f"localhost:{PORT}", f"127.0.0.1:{PORT}"}:
        return True
    try:
        parsed = urlsplit(str(addr))
        return (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                and parsed.port == PORT and parsed.path in {"", "/"}
                and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


def _public_tunnel():
    try:
        result = _fetch_json(INSPECTOR, timeout=2)
        for tunnel in result.get("tunnels", []):
            url = tunnel.get("public_url", "")
            parsed = urlsplit(url)
            if (parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password
                    and not parsed.query and not parsed.fragment and parsed.path in {"", "/"}
                    and _forwarded_to_demo(tunnel.get("config", {}).get("addr"))):
                return url.rstrip("/")
    except (OSError, ValueError, AttributeError):
        pass
    return None


class _SameOriginRedirect(HTTPRedirectHandler):
    def __init__(self, origin):
        super().__init__()
        self.origin = urlsplit(origin).netloc

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.netloc != self.origin or target.scheme not in {"http", "https"}:
            raise DemoError("演示服务返回了不受支持的跳转。")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _authenticated_info(origin, token):
    opener = build_opener(ProxyHandler({}), HTTPCookieProcessor(CookieJar()), _SameOriginRedirect(origin))
    headers = {"ngrok-skip-browser-warning": "true"}
    login = origin + "/demo-login?" + urlencode({"token": token})
    with opener.open(Request(login, headers=headers), timeout=10) as response:
        response.read(256 * 1024)
    with opener.open(Request(origin + "/api/info", headers=headers), timeout=10) as response:
        raw = response.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        return None
    result = json.loads(raw.decode("utf-8"))
    info = result.get("info", {})
    if result.get("success") is True and info.get("ready") and "INKLAB" in info.get("name", ""):
        return info
    return None


def _write_session(session, public_url):
    DATA.mkdir(parents=True, exist_ok=True)
    saved = {key: session[key] for key in ("token", "demo_pid", "ngrok_pid", "managed_by", "started_at") if key in session}
    saved.update({"project": str(ROOT), "demo_port": PORT, "public_url": public_url, "verified_at": _stamp()})
    temporary = SESSION.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SESSION)
    link = public_url + "/demo-login?" + urlencode({"token": saved["token"]})
    LINK_FILE.write_text("墨迹 INKLAB 独立演示入口\n\n" + link + "\n\n这是本轮演示访问授权，不是 ngrok 账号令牌。\n演示数据独立于本机 5003 与原版 5001。\n首次遇到 ngrok 免费提示页时，请点击 Visit Site。\n", encoding="utf-8")
    return link


def _show(link, reused):
    print("\n" + ("已验证并复用正在运行的独立演示。" if reused else "外网演示已经就绪。"), flush=True)
    print("\n复制下面完整地址，在手机或其他电脑浏览器打开：\n", flush=True)
    print(link, flush=True)
    print("\n地址也已保存到：" + str(LINK_FILE), flush=True)
    print("首次遇到 ngrok 免费提示页，请点击 Visit Site。", flush=True)
    if reused:
        print("此入口仅复用；演示继续由原启动进程保持运行。", flush=True)
    else:
        print("\n保持本窗口运行；按 Ctrl+C 结束本次启动的演示和隧道。", flush=True)


def _reuse_existing():
    public_url = _public_tunnel()
    if not public_url:
        return False
    candidates = [_read_json(SESSION)]
    for session in candidates:
        if not _valid_session(session):
            continue
        try:
            if _authenticated_info(public_url, session["token"]):
                if "managed_by" not in session:
                    session = {**session, "managed_by": "existing-rollout-session"}
                _show(_write_session(session, public_url), reused=True)
                return True
        except (OSError, ValueError, DemoError):
            continue
    return False


def _ngrok_path():
    shared = ROOT.parents[1] / ".tools" / "ngrok" / "ngrok.exe"
    found = shared if shared.is_file() and shared.stat().st_size else shutil.which("ngrok")
    if not found:
        raise DemoError("未找到 ngrok。请先把官方 ngrok 放在项目外的 .tools/ngrok，或加入 PATH；此入口不会自动下载或配置账号。")
    return str(found)


def _model_ready():
    try:
        result = _fetch_json("http://127.0.0.1:8087/v1/models", timeout=1)
        return any(row.get("id") == "inklab-local" for row in result.get("data", []))
    except (OSError, ValueError, AttributeError):
        return False


def _launch(arguments, environment, log):
    options = {"cwd": str(ROOT), "env": environment, "stdin": subprocess.DEVNULL, "stdout": log, "stderr": log}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(arguments, **options)


def _stop_owned(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Only an actual Popen object created in THIS invocation is accepted.
        # /T also closes the venv launcher child instead of leaving Python behind.
        subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=15, check=False)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def main():
    # A verified existing session is returned before any process launch or stop.
    if _reuse_existing():
        return 0
    occupied = [str(port) for port in (PORT, 4040) if _port_in_use(port)]
    if occupied:
        raise DemoError("端口 " + "/".join(occupied) + " 已被服务占用，但未验证到匹配的演示会话。为避免干扰，此入口已退出，没有停止任何服务。")
    executable = _ngrok_path()
    DATA.mkdir(parents=True, exist_ok=True)
    owned = []
    session = {"token": secrets.token_urlsafe(32), "demo_port": PORT, "managed_by": "public_demo.py", "started_at": _stamp()}
    # Keep public-demo history separate and do not start/own a second LLM.
    environment = os.environ.copy()
    environment.update({"APP_HOST": "127.0.0.1", "PORT": str(PORT), "INKLAB_DATA_DIR": str(DATA),
                        "INKLAB_DEMO_TOKEN": session["token"], "INKLAB_DISABLE_LOCAL_LLM": "1", "PYTHONUTF8": "1"})
    for key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        environment[key] = ""  # An existing cloud configuration is not inherited by the public demo.
    if _model_ready():
        environment.update({"LLM_BASE_URL": "http://127.0.0.1:8087/v1", "LLM_MODEL": "inklab-local"})
    try:
        with ExitStack() as stack:
            app_log = stack.enter_context((DATA / "application.log").open("ab"))
            ngrok_log = stack.enter_context((DATA / "ngrok.log").open("ab"))
            app = _launch([sys.executable, "-X", "utf8", str(ROOT / "serve.py")], environment, app_log)
            owned.append(app)
            session["demo_pid"] = app.pid
            for _ in range(120):
                if app.poll() is not None:
                    raise DemoError("独立演示应用未启动，请检查 instance/public-demo/application.log。")
                try:
                    if _authenticated_info(LOCAL, session["token"]):
                        break
                except (OSError, ValueError, DemoError):
                    pass
                time.sleep(.5)
            else:
                raise DemoError("演示应用启动超时；本次启动的进程将被清理。")
            tunnel = _launch([executable, "http", str(PORT), "--log=stdout", "--log-format=json"], environment, ngrok_log)
            owned.append(tunnel)
            session["ngrok_pid"] = tunnel.pid
            for _ in range(60):
                if tunnel.poll() is not None:
                    raise DemoError("ngrok 未能启动。请确认已在本机配置账号；不要把账号令牌粘贴到聊天或代码。日志位于 instance/public-demo/ngrok.log。")
                public_url = _public_tunnel()
                try:
                    if public_url and _authenticated_info(public_url, session["token"]):
                        break
                except (OSError, ValueError, DemoError):
                    pass
                time.sleep(.5)
            else:
                raise DemoError("公网授权访问尚未通过，请检查 ngrok 网络和账号状态；本次进程将被清理。")
            _show(_write_session(session, public_url), reused=False)
            while all(process.poll() is None for process in owned):
                time.sleep(.5)
            raise DemoError("本次演示的一个子进程已停止，其余本次进程将被清理。")
    finally:
        for process in reversed(owned):
            try:
                _stop_owned(process)
            except (OSError, subprocess.SubprocessError):
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n本次启动的演示与隧道已结束；原有服务保持运行。")
        raise SystemExit(0)
    except DemoError as error:
        print("\n" + str(error))
        raise SystemExit(1)
    except (OSError, ValueError):
        print("\n演示入口未完成，请检查本机服务、网络和模型文件。未停止任何非本次服务。")
        raise SystemExit(1)
