import os
import threading
import yt_dlp
import cv2
import time


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

OPEN_TIMEOUT_MS = 10_000
# HLS en directo entrega segmentos de ~5 s en rafaga: read() puede esperar
# ese tiempo con normalidad, asi que el timeout debe ser bastante mayor.
READ_TIMEOUT_MS = 15_000
MAX_FAILED_READS = 3      # lecturas fallidas seguidas antes de reconectar
STALL_SECONDS = 20.0      # sin frames nuevos durante este tiempo -> reconectar


class StreamReader():

    def __init__(self, url: str):
        self.url = url
        self.frame = None
        self.frame_time = time.monotonic()
        self.frame_id = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture = None
        self.reconnects = 0

        self._open()

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
                with self.lock:
                    self.frame = frame
                    self.frame_time = self._last_ok
                    self.frame_id += 1
            else:
                failed += 1
                stalled = time.monotonic() - self._last_ok > STALL_SECONDS
                if failed >= MAX_FAILED_READS or stalled:
                    failed = 0
                    self._reconnect()
                else:
                    time.sleep(0.05)  # evita un bucle activo


    # devuelve la ultima pareja de frame t y un contador
    def read(self):
        with self.lock:
            frame = self.frame
            timestamp = self.frame_time
            frame_id = self.frame_id


        if frame is not None and timestamp is not None and frame_id is not None:
            ok = True
        else:
            ok = False

        return ok, frame, timestamp, frame_id

    def stop(self):
        self.stop_event.set()
        # esperar a que el hilo termine
        # read() tiene timeout, asi que el hilo termina en unos segundos como mucho
        self.thread.join(timeout=READ_TIMEOUT_MS / 1000 + 2)
        if self.capture is not None:
            self.capture.release()


class StreamProcessor():

    def __init__(self, url: str, fps: int):
        self.reader = StreamReader(url)
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = time.monotonic()

    def process(self):
        while True:
            ok, frame, timestamp, frame_id = self.reader.read()
            if not ok:
                continue

            current_time = time.monotonic()
            if current_time - self.last_frame_time >= self.frame_interval:
                # Aquí puedes procesar el frame como desees
                print(f"Procesando frame {frame_id} a tiempo {timestamp}")
                self.last_frame_time = current_time

    def stop(self):
        self.reader.stop()


if __name__ == "__main__":
    reader = StreamReader("https://www.youtube.com/watch?v=Jy1Y9f8NEY0")
    last_frame_id = -1
    leer = True

    while leer:
        ok, frame, timestamp, frame_id = reader.read()
        #print(f"ok: {ok}, frame: {frame}, timestamp: {timestamp}, frame_id: {frame_id}")
        if ok and frame_id != last_frame_id:
            #print("Hay un frame nuevo")
            last_frame_id = frame_id 
    reader.stop()