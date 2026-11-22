# MJPEG server: open http://localhost:8000/ in a browser

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class FrameServer:
    def __init__(self, port: int = 8000, host: str = "127.0.0.1", quality: int = 80, max_fps: int = 30):
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._quality = quality
        self._interval = 1.0 / max_fps
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        with outer._lock:
                            jpeg = outer._jpeg
                        if jpeg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                             + jpeg + b"\r\n")
                        time.sleep(outer._interval)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"Serving MJPEG at http://{host}:{port}/")

    def update(self, frame) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def close(self) -> None:
        self._server.shutdown()
