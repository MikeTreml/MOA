"""A minimal fake OpenAI-compatible chat-completions upstream.

Shared by the e2e harnesses so they exercise the real providers.py client and
workflow without needing a live LM Studio / model. Returns stage-appropriate
JSON for json_mode calls (orchestrate/evaluate) and an SSE token stream for
streaming calls (workers/synthesize/refine).
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Gate-eval call counter (stateful, per server process) so a loop fails a couple
# of times then passes — lets the browser test watch a real refine loop.
_GATE_CALLS = {"n": 0}

STREAM_TOKENS = ["Hello", " from", " fake", " upstream."]
STREAM_TEXT = "".join(STREAM_TOKENS)

# A 1x1 PNG so /v1/images/generations returns a real, renderable data URI.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def make_handler(token_delay: float = 0.0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # cheap answer for any probe (e.g. /v1/models)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[]}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            if self.path == "/__test__/reset-gate":
                _GATE_CALLS["n"] = 0
                self.send_response(204)
                self.end_headers()
                return
            if "images" in self.path:
                n = int(body.get("n", 1) or 1)
                payload = {"created": 0, "data": [{"b64_json": TINY_PNG_B64} for _ in range(n)]}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            last = (body.get("messages") or [{}])[-1].get("content", "")
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                # A node prompt asking for a JSON list streams one, so a fanout
                # over it expands; everything else streams the normal sentence.
                if any(k in last.lower() for k in ("json list", "angles", "json array")):
                    tokens = ['["angle one"', ',"angle two"', ',"angle three"]']
                else:
                    tokens = STREAM_TOKENS
                for tok in tokens:
                    chunk = {"choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    if token_delay:
                        time.sleep(token_delay)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            # Match graph-gate prompts specifically. Ordinary evaluator prompts
            # also request JSON containing a score and used to consume this
            # stateful counter, making the loop test depend on execution order.
            if "return only json" in last.lower() and "criteria:" in last.lower():
                # Gate scoring: fail the first two evaluations, then pass.
                _GATE_CALLS["n"] += 1
                value = 0.2 if _GATE_CALLS["n"] <= 2 else 0.9
                match = re.search(r"Criteria:\s*([^.\n]+)", last)
                terms = [t.strip() for t in match.group(1).split(",")] if match else ["quality"]
                content = json.dumps({t: {"score": value, "note": "auto"} for t in terms})
            elif "subtask" in last.lower():
                content = '{"subtasks":[{"title":"A","prompt":"a"},{"title":"B","prompt":"b"}]}'
            elif "evaluate" in last.lower():
                content = '{"status":"PASS","feedback":"good","score":0.9}'
            else:
                content = "plain answer"
            payload = {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def start(port: int, token_delay: float = 0.0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(token_delay))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
