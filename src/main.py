"""
Punto de entrada del proyecto.
python src/main.py --url <url>
"""

import argparse
import time
import stream
import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL of the video to process", required=True)
    parser.add_argument("--fps", help="Desired FPS for the video processing", type=int, default=60)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than 0")

    print(f"Processing video from URL: {args.url}")
    print(f"Desired FPS: {args.fps}")
    reader = stream.StreamReader(args.url)
    frame_interval = 1.0 / args.fps
    last_frame_id = -1
    next_frame_time = time.monotonic()
    leer = True

    while leer:
        ok, frame, timestamp, frame_id = reader.read()

        #print(reader.capture.isOpened())
        now = time.monotonic()
        if now < next_frame_time:
            time.sleep(next_frame_time - now)
            continue

        if ok and frame_id != last_frame_id:
            last_frame_id = frame_id
            print(f"Frame nuevo: {frame_id}, timestamp: {timestamp}")
            cv2.imshow("Stream", frame)
            print(f"Frame ID: {frame_id}, Timestamp: {timestamp}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                leer = False

        next_frame_time += frame_interval
        if next_frame_time < time.monotonic():
            next_frame_time = time.monotonic()

    reader.stop()
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()