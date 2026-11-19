"""
Punto de entrada del proyecto.
python src/main.py --url <url>

urls de ejemplo:
    https://www.youtube.com/watch?v=Jy1Y9f8NEY0
"""

import argparse
import time
import cv2
import stream
import roi_selector
import detector
import tracker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL of the video to process", required=True)
    parser.add_argument("--fps", help="Fallback FPS if the stream does not report one", type=int, default=30)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than 0")

    print(f"Processing video from URL: {args.url}")
    print(f"Desired FPS: {args.fps}")
    # El StreamReader ya emite los frames a ritmo constante (FPS del video)
    # --fps solo se usa si el stream no informa de sus FPSs
    reader = stream.StreamReader(args.url, fps=args.fps)
    

    first_frame = reader.first_frame()
    if first_frame is not None:
        selector = roi_selector.RoiSelector(first_frame)
        roi = selector.select()  # lee un frame para seleccionar el ROI
        if roi is not None:
            print(f"ROI selected: {roi}")
        else:
            print("ROI selection cancelled")
            return 1
    else:
        print("Could not read first frame")
        return 1

    reader.start()  # inicia el hilo de lectura de frames

    read = True
    detector_instance = None  # se crea una sola vez para conservar el modelo de fondo
    trackers = []  # lista de objetos trackeados
    next_id = 0
    while read:
        ok, frame, video_time, frame_id = reader.read()

        if ok:
            cv2.polylines(frame, [roi], isClosed=True, color=(0, 255, 255), thickness=2)

            if detector_instance is None:
                detector_instance = detector.Detector(frame.shape, roi, min_area_frac=0.0005, max_area_frac=0.05)
            detections = detector_instance.detect(frame)
            #print(f"Detections: {len(detections)}")

            for track in trackers:
                track.update(frame)

            for detection in detections:
                if not tracker.overlaps(detection, trackers):
                    trackers.append(tracker.Track(next_id, frame, detection))
                    next_id += 1
            trackers = [t for t in trackers if t.missed < 10]

            for track in trackers:
                x, y, w, h = track.state
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(track.id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            cv2.imshow("Stream", frame)
            print(f"Frame ID: {frame_id}, video_time: {video_time:.3f}s")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            read = False
        if not ok:
            time.sleep(0.002)

    reader.stop()
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()