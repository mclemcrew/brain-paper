"""Loopback-only app server, also used as the packaged Tauri sidecar."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import sys
import tempfile
import re
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .catalog import PROFILES, catalog, hardware
from .analysis import pack_status
from .jobs import Jobs
from .store import Store

def export_name(asset):
    """Readable DAW filenames without exposing internal paths or header syntax."""
    title = unicodedata.normalize('NFKD', str(asset.get('name', 'take'))).encode('ascii', 'ignore').decode()
    title = re.sub(r'[^a-zA-Z0-9]+', '-', Path(title).stem).strip('-').lower()[:64] or 'take'
    kind = re.sub(r'[^a-zA-Z0-9]', '', str(asset.get('kind', 'audio'))) or 'audio'
    version = 'personal' if asset.get('adapter_id') else 'original'
    ident = re.sub(r'[^a-zA-Z0-9]', '', str(asset.get('id', '')))[:8]
    extension = Path(asset['path']).suffix.lower()
    return f'{title}-{kind}-{version}-{ident}{extension}'

def locations():
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
        return base / "studio-ui", base / "sa3-mlx"
    repo = Path(__file__).resolve().parents[2]
    return repo / "desktop" / "dist", repo / ".local-runtime" / "stable-audio-3" / "optimized" / "mlx"

class AppServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, root, ui=None, runtime=None):
        self.store = Store(Path(root))
        self.ui, default_runtime = locations()
        if ui: self.ui = Path(ui)
        self.jobs = Jobs(self.store, Path(runtime) if runtime else default_runtime)
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, Handler)

    def server_close(self):
        self.jobs.close()
        super().server_close()

class Handler(BaseHTTPRequestHandler):
    server: AppServer
    def log_message(self, *args): pass  # Audio URLs can contain the session token.

    def send(self, value, code=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def guard(self, auth=True):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed: raise PermissionError("Invalid local host")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{h}" for h in allowed}: raise PermissionError("Cross-origin requests are disabled")
        if self.headers.get("Sec-Fetch-Site") == "cross-site": raise PermissionError("Cross-site requests are disabled")
        query = parse_qs(urlparse(self.path).query)
        supplied = self.headers.get("X-Ravi-Token", query.get("token", [""])[0])
        if auth and not secrets.compare_digest(supplied, self.server.token): raise PermissionError("Invalid session")

    def body(self):
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size <= 65536: raise ValueError("Invalid request size")
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict): raise ValueError("Expected a request object")
        return value

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            self.guard(auth=path.startswith("/api/"))
            if path == "/api/state":
                self.send({"projects": self.server.store.list(), "jobs": self.server.jobs.list(),
                    "hardware": hardware(), "profiles": PROFILES,
                    "models": catalog(self.server.store.root / "models", self.server.jobs.runtime),
                    "tagger": pack_status(self.server.store.root),
                    "root": str(self.server.store.root)})
            elif path == "/api/readiness":
                from .metadata import readiness
                q = parse_qs(urlparse(self.path).query)
                self.send(readiness(self.server.store.get(q["project_id"][0])))
            elif path == "/api/training-preview":
                from .metadata import preview
                q = parse_qs(urlparse(self.path).query)
                self.send(preview(self.server.store.get(q["project_id"][0])))
            elif path == "/api/checkpoints":
                q = parse_qs(urlparse(self.path).query)
                self.send(self.server.jobs.checkpoints(q["project_id"][0]))
            elif path == "/api/asset":
                q = parse_qs(urlparse(self.path).query)
                pid, aid = q["project_id"][0], q["asset_id"][0]
                p = self.server.store.get(pid)
                a = next(x for x in p["tracks"]+p["assets"] if x["id"] == aid)
                self.file(self.server.store.asset_path(pid, a["path"]), download="download" in q, filename=export_name(a))
            else:
                relative = "index.html" if path == "/" else path.lstrip("/")
                target = (self.server.ui / relative).resolve()
                if not target.is_relative_to(self.server.ui.resolve()): raise PermissionError("Invalid asset path")
                self.file(target, html=relative == "index.html")
        except (PermissionError,) as exc: self.send({"error": str(exc)}, 403)
        except (ValueError, KeyError, StopIteration, FileNotFoundError) as exc: self.send({"error": str(exc) or "Not found"}, 400)
        except (BrokenPipeError, ConnectionResetError): pass

    def file(self, path, html=False, download=False, filename=None):
        size = path.stat().st_size
        data = None
        if html:
            data = path.read_bytes().replace(b"__RAVI_SESSION_TOKEN__", self.server.token.encode())
            size = len(data)
        start, end = 0, size-1
        range_header = self.headers.get("Range") if not html else None
        if range_header:
            import re
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            if not match: raise ValueError("Unsupported byte range")
            start = int(match[1]); end = min(size-1, int(match[2]) if match[2] else size-1)
            if start > end: self.send_error(416); return
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", "text/html; charset=utf-8" if html else mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end-start+1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if range_header: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download: self.send_header("Content-Disposition", f'attachment; filename="{filename or path.name}"')
        self.end_headers()
        if data is not None: self.wfile.write(data); return
        with path.open("rb") as f:
            f.seek(start)
            remaining = end-start+1
            while remaining:
                chunk = f.read(min(65536, remaining))
                if not chunk: break
                self.wfile.write(chunk); remaining -= len(chunk)

    def do_POST(self):
        try:
            self.guard()
            path = urlparse(self.path).path
            if path == "/api/import":
                q = parse_qs(urlparse(self.path).query)
                pid, name = q["project_id"][0], Path(q["name"][0]).name
                self.server.store.get(pid)
                if Path(name).suffix.lower() not in (".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg"):
                    raise ValueError("Import WAV, AIFF, FLAC, MP3, or OGG audio.")
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 512*1024**2: raise ValueError("Each imported file must be between 1 byte and 512 MB.")
                with tempfile.TemporaryDirectory(dir=self.server.store.root) as tmp:
                    upload = Path(tmp) / name
                    with upload.open("wb") as f:
                        remaining = size
                        while remaining:
                            block = self.rfile.read(min(65536, remaining))
                            if not block: raise ValueError("Upload interrupted")
                            f.write(block); remaining -= len(block)
                    self.send(self.server.store.import_file(pid, upload, name))
                return
            body = self.body()
            store, jobs = self.server.store, self.server.jobs
            if path == "/api/projects": self.send(store.create(body["name"]))
            elif path == "/api/project": self.send(store.update(body["project_id"], body))
            elif path == "/api/split": self.send(store.split(body["project_id"]))
            elif path == "/api/select": self.send(store.select(body["project_id"], body.get("asset_id"), body.get("undo", False)))
            elif path == "/api/jobs": self.send(jobs.submit(body), 202)
            elif path == "/api/cancel": jobs.cancel(body["id"]); self.send({"ok": True})
            else: self.send({"error": "Unknown operation"}, 404)
        except PermissionError as exc: self.send({"error": str(exc)}, 403)
        except (ValueError, KeyError, StopIteration, FileNotFoundError, RuntimeError) as exc: self.send({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as exc: self.send({"error": f"Operation failed: {type(exc).__name__}: {exc}"}, 500)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.argv.pop(1)
        from .worker import main as worker_main
        worker_main(); return
    parser = argparse.ArgumentParser(description="rAvI offline composer studio")
    parser.add_argument("--root", type=Path, default=Path.home() / "Library/Application Support/rAvI Studio")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ui", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--parent-stdin", action="store_true", help="Stop when the desktop parent closes stdin")
    args = parser.parse_args()
    server = AppServer(("127.0.0.1", args.port), args.root, args.ui, args.runtime)
    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}"}), flush=True)
    if args.parent_stdin:
        import threading
        def parent_watch():
            sys.stdin.buffer.read()
            server.shutdown()
        threading.Thread(target=parent_watch, daemon=True).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == "__main__": main()
