"""
Punto de entrada del proyecto.
python src/main.py --url <url>

urls de ejemplo:
    https://www.youtube.com/watch?v=Jy1Y9f8NEY0
"""

import argparse
import time
import stream
import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL of the video to process", required=True)
    parser.add_argument("--fps", help="Fallback FPS if the stream does not report one", type=int, default=30)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than 0")

    print(f"Processing video from URL: {args.url}")
    print(f"Desired FPS: {args.fps}")
    # El StreamReader ya emite los frames a ritmo constante (FPS del video);
    # --fps solo se usa si el stream no informa de su FPS.
    reader = stream.StreamReader(args.url, fps=args.fps)
    leer = True

    while leer:
        ok, frame, video_time, frame_id = reader.read()

        if ok:
            cv2.imshow("Stream", frame)
            print(f"Frame ID: {frame_id}, video_time: {video_time:.3f}s")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            leer = False
        if not ok:
            time.sleep(0.002)

    reader.stop()
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()