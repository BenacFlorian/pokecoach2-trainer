"""Serveur HTTP simulé pour tests intégration collecte."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    pages: dict[str, list[dict[str, Any]]] = {}

    def do_GET(self) -> None:
        if self.path.startswith("/search.json"):
            body = json.dumps(_Handler.pages.get("search", [])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.endswith(".json"):
            rid = self.path.strip("/").replace(".json", "")
            payload = _Handler.pages.get(rid, {"id": rid, "uploadtime": 1, "log": "|t|1\n"})
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def start_server() -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    return server, port
