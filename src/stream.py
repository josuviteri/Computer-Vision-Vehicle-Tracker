import os
import queue
import threading
import yt_dlp
import cv2
import time

"""
Lee el stream y lo entrega a ritmo constante.

El streaming llega en rafagas (un segmento de 150 frames cada 5 segundos). Un hilo
lee lo mas rapido que puede y mete los frames en una cola.
read() los saca uno a uno a la velocidad definida en FPSs del video, tras llenar la cola para empezar (prebuffer inicial).
El coste es una latencia fija de ~PREBUFFER_SECONDS respecto al directo.
"""

# Variables de control de la lectura del stream: 
# reconexion, timeout de red, tiempo de prebuffer, etc.

OPEN_TIMEOUT_MS = 10_000
# HLS en directo entrega segmentos de ~5 s en rafaga: read() puede esperar
# ese tiempo con normalidad, asi que el timeout debe ser bastante mayor.
READ_TIMEOUT_MS = 15_000
MAX_FAILED_READS = 3      # lecturas fallidas seguidas antes de reconectar
STALL_SECONDS = 20.0      # sin frames nuevos durante este tiempo -> reconectar


DEFAULT_FPS = 30.0
BUFFER_SECONDS = 10.0     # capacidad maxima de la cola (segundos de video)
PREBUFFER_SECONDS = 6.0   # video acumulado antes de empezar a emitir

def resolve_stream(url: str) -> str:
    options = {
        "format": "best[height<=720]",
        "quiet": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android"],
            },
        },
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


# Opciones de FFmpeg: reconexion HTTP y timeout de red (microsegundos).
# Deben fijarse ANTES de crear el primer VideoCapture.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "reconnect;1|reconnect_streamed;1|reconnect_delay_max;5|rw_timeout;5000000",
)

class StreamReader():
    def __init__(self, url: str, fps: float = DEFAULT_FPS):
        self.url = url
        self.fallback_fps = fps
        self.fps = fps
        self.stop_event = threading.Event()
        self.capture = None
        self.reconnects = 0
        self.frame_id = 0          # frames leidos del stream (hilo lector)
        self.last_emitted_id = -1

        self._open()
        self.interval = 1.0 / self.fps
        self.queue = queue.Queue(maxsize=max(1, int(BUFFER_SECONDS * self.fps)))
        self.prebuffer_frames = min(
            int(PREBUFFER_SECONDS * self.fps), self.queue.maxsize
        )
        self._buffering = True
        self._next_due = 0.0

        self.thread = threading.Thread(daemon=True, target=self._reader_loop)
        self.thread.start()

    def _open(self):
        # La URL directa de YouTube caduca: se resuelve de nuevo en cada apertura.
        src = resolve_stream(self.url)
        capture = cv2.VideoCapture(
            src,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS,
            ],
        )
        self.capture = capture
        self._last_ok = time.monotonic()
        reported = capture.get(cv2.CAP_PROP_FPS)
        if 1.0 <= reported <= 120.0:   # descarta 0, NaN y valores absurdos
            self.fps = reported
        else:
            self.fps = self.fallback_fps
        return capture.isOpened()

    def _reconnect(self):
        self.reconnects += 1
        print(f"[stream] reconectando (#{self.reconnects})", flush=True)
        if self.capture is not None:
            self.capture.release()
        while not self.stop_event.is_set():
            try:
                if self._open():
                    return
            except Exception as e:  # yt_dlp puede fallar de forma transitoria
                print(f"[stream] error al reabrir: {e}", flush=True)
            self.stop_event.wait(1.0)

    def _put(self, item):
        # put con espera: si la cola esta llena el lector se frena (backpressure)
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def _reader_loop(self):
        failed = 0
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            ok, frame = self.capture.read()
            dt = time.monotonic() - t0

            if dt > 1.0 and not ok:
                print(f"[stream] read() fallo tras {dt:.2f}s", flush=True)

            if ok:
                failed = 0
                self._last_ok = time.monotonic()
                self.frame_id += 1
                self._put((frame, self.frame_id))
            else:
                failed += 1
                stalled = time.monotonic() - self._last_ok > STALL_SECONDS
                if failed >= MAX_FAILED_READS or stalled:
                    failed = 0
                    self._reconnect()
                else:
                    time.sleep(0.05)  # evita un bucle activo

    def read(self):
        """No bloquea. Devuelve (ok, frame, video_time, frame_id).

        ok=True solo cuando toca emitir un frame nuevo (a ritmo constante).
        video_time son los segundos que llevamos de video: frame_id / fps.
        """
        now = time.monotonic()

        if self._buffering:
            if self.queue.qsize() < self.prebuffer_frames:
                return False, None, None, self.last_emitted_id
            self._buffering = False
            self._next_due = now

        if now < self._next_due:
            return False, None, None, self.last_emitted_id

        try:
            frame, frame_id = self.queue.get_nowait()
        except queue.Empty:
            # underrun: volver a acumular buffer
            print("[stream] buffer vacio, reacumulando frames", flush=True)
            self._buffering = True
            return False, None, None, self.last_emitted_id

        self._next_due += self.interval
        if self._next_due < now - self.interval:
            self._next_due = now  # consumidor lento: no intentar recuperar en rafaga
        self.last_emitted_id = frame_id
        return True, frame, frame_id / self.fps, frame_id

    def stop(self):
        self.stop_event.set()
        # read() tiene timeout, asi que el hilo termina en unos segundos como mucho
        self.thread.join(timeout=READ_TIMEOUT_MS / 1000 + 2)
        if self.capture is not None:
            self.capture.release()
