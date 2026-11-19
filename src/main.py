"""
Proyect entry point
python src/main.py --url <url>

proposed url:
    https://www.youtube.com/watch?v=Jy1Y9f8NEY0

The more objects tracked at once, the slower the program will run. The detection frequency can be reduced with --detect-every N.

This script uses the RoiSelector to select a region of interest (ROI) in the first frame of the video. The ROI is used to limit the area where detections are made, improving performance and reducing false positives.
The Detector uses background subtraction to detect moving objects in the ROI. It returns a list of Detection objects, each containing a bounding box and area. The Detector is updated in every frame, but 
detections are only extracted every N frames, as specified by the --detect-every argument. This allows for a trade-off between detection accuracy and performance.
The TrackManager manages the tracks of detected objects. It uses KCF to track the position of objects between detections, and Kalman to predict the position of objects when they are not detected in order to recover 
them in case they where lost but are still in present in the ROI.

Detector: background subtraction
KCF: where the object is
Kalman: where the object is expected to be

More information about tracking implmenetation is available in their respective files.
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
    parser.add_argument("--url", help="URL of the video to process", default="https://www.youtube.com/watch?v=Jy1Y9f8NEY0")
    parser.add_argument("--fps", help="Fallback FPS if the stream does not report one", type=int, default=30)
    parser.add_argument("--detect-every", help="Extract detections every N frames", type=int, default=3)
    parser.add_argument("--tracker", help="Tracker type", choices=["KCF", "CSRT"], default="KCF")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than 0")

    print(f"Processing video from URL: {args.url}")
    print(f"Desired FPS: {args.fps}")
    print(f"Detection frequency: every {args.detect_every} frames")
    print(f"Tracker type: {args.tracker}")
    # El StreamReader ya emite los frames a ritmo constante (FPS del video)
    # --fps solo se usa si el stream no informa de sus FPSs
    reader = stream.StreamReader(args.url, fps=args.fps)
    

    first_frame = reader.first_frame()
    if first_frame is not None:
        selector = roi_selector.RoiSelector(first_frame)
        roi = selector.select()  # lee un frame para seleccionar el ROI
        if roi is not None:
            print(f"\nROI selected:\n {roi}")
        else:
            print("ROI selection cancelled")
            return 1
    else:
        print("Could not read first frame")
        return 1

    reader.start()  # inicia el hilo de lectura de frames

    read = True
    detector_instance = None  # se crea una sola vez para conservar el modelo de fondo
    manager = None
    n = 0
    while read:
        ok, frame, video_time, frame_id = reader.read()

        if ok:
            if detector_instance is None:
                detector_instance = detector.Detector(frame.shape, roi, min_area_frac=0.0005, max_area_frac=0.05)
                manager = tracker.TrackManager(roi, tracker_type=args.tracker)

            # Primero detectar/trackear sobre el frame limpio; el overlay se dibuja despues
            detections = detector_instance.detect(frame, extract=(n % args.detect_every == 0))
            manager.step(frame, detections)
            n += 1

            cv2.polylines(frame, [roi], isClosed=True, color=(0, 255, 255), thickness=2)
            for track in manager.confirmed:
                x, y, w, h = track.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(track.id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Total: {manager.total_count}  Activos: {len(manager.confirmed)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Stream", frame)
            print(f"\rFrame ID: {frame_id}, video_time: {video_time:.3f}s, Total: {manager.total_count}, Active: {len(manager.confirmed)}", end="", flush=True) # ahora la terminal no se llena de prints y se actualiza en la misma linea

        if cv2.waitKey(1) & 0xFF == ord("q"):
            read = False
        if not ok:
            time.sleep(0.002)

    reader.stop()
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()