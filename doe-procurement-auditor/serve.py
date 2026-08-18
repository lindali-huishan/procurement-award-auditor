#!/usr/bin/env python3
"""Open the audit site in a browser.

The site is static and works from ``file://`` -- double-clicking
``site/index.html`` is enough. This exists for anyone who would rather have a
real origin, which some browsers require for local font loading.

    python3 serve.py            # serves site/ on 8000 and opens a browser
    python3 serve.py 8080       # pick a port
"""

from __future__ import annotations

import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

SITE = Path(__file__).resolve().parent / "site"


def main() -> int:
    if not (SITE / "index.html").exists():
        print("site/ not built yet. Run:\n"
              "  python3 src/export_webapp.py && python3 src/build_site.py")
        return 1

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(SITE), **kwargs)

        def log_message(self, *args):  # keep the console quiet
            pass

    # Without this a restart inside the socket's TIME_WAIT window fails with
    # "address already in use", which is the most common way this script
    # annoys someone.
    socketserver.TCPServer.allow_reuse_address = True

    try:
        server = socketserver.TCPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        print(f"could not bind port {port}: {exc}\nTry: python3 serve.py {port + 1}")
        return 1

    url = f"http://127.0.0.1:{port}/index.html"
    print(f"DOE Procurement Auditor -> {url}\nCtrl-C to stop.")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
