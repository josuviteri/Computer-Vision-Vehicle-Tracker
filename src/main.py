"""
Project entry point
python src/main.py --url <url>

proposed urls:
    Carretera 1 :https://www.youtube.com/watch?v=Jy1Y9f8NEY0
    Cruce 2 :https://www.youtube.com/watch?v=4cA_-WAm-SE
    Cruce 3 (Suele estar oscuro): https://www.youtube.com/watch?v=2juLrCH5w9U

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
import json
import os
import time
import cv2
import numpy as np
import web_view
import stream
import roi_selector
import detector
import tracker
import draw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL of the video to process", default="https://www.youtube.com/watch?v=Jy1Y9f8NEY0")
    parser.add_argument("--fps", help="Fallback FPS if the stream does not report one", type=int, default=30)
    parser.add_argument("--detect-every", help="Extract detections every N frames", type=int, default=3)
    parser.add_argument("--tracker", help="Tracker type", choices=["KCF", "CSRT"], default="KCF")
    parser.add_argument("--serve", help="Serve the output as MJPEG on this port instead of opening a window", type=int, default=None)
    parser.add_argument("--roi", help="JSON file to load the ROI from (or save it to if missing)", default=None)
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
        roi = None
        if args.roi and os.path.exists(args.roi):
            with open(args.roi) as f:
                roi = np.array(json.load(f), dtype=np.int32)
        elif args.serve is not None:
            print("--serve needs an existing --roi file (select it once locally without --serve)")
            return 1
        else:
            selector = roi_selector.RoiSelector(first_frame)
            # lee un frame para seleccionar el ROI
            roi = selector.select()  
            if roi is not None and args.roi:
                os.makedirs(os.path.dirname(args.roi) or ".", exist_ok=True)
                with open(args.roi, "w") as f:
                    json.dump(roi.tolist(), f)
        if roi is not None:
            print(f"\nROI selected:\n {roi}")
        else:
            print("ROI selection cancelled")
            return 1
    else:
        print("Could not read first frame")
        return 1

    # inicia el hilo de lectura de frames
    reader.start()  
    server = web_view.FrameServer(args.serve, "127.0.0.1") if args.serve is not None else None

    read = True
    # se crea una sola vez para conservar el modelo de fondo
    detector_instance = None  
    manager = None
    n = 0
    try:
        while read:
            ok, frame, video_time, frame_id = reader.read()

            if ok:
                if detector_instance is None:
                    detector_instance = detector.Detector(frame.shape, roi, min_area_frac=0.0001, max_area_frac=0.05)
                    manager = tracker.TrackManager(roi, tracker_type=args.tracker)

                # Primero detectar/trackear sobre el frame limpio; el overlay se dibuja despues
                detections = detector_instance.detect(frame, extract=(n % args.detect_every == 0))
                manager.step(frame, detections)
                n += 1

                cv2.polylines(frame, [roi], isClosed=True, color=(0, 255, 255), thickness=2)
                for track in manager.confirmed:
                    draw.draw_track(frame, track)
                cv2.putText(frame, f"Total: {manager.total_count}  Activos: {len(manager.confirmed)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
                if server:
                    server.update(frame)
                else:
                    cv2.imshow("Stream", frame)
                # ahora la terminal no se llena de prints y se actualiza en la misma linea
                print(f"\rFrame ID: {frame_id}, video_time: {video_time:.3f}s, Total: {manager.total_count}, Active: {len(manager.confirmed)}", end="", flush=True) 

            if server is None and cv2.waitKey(1) & 0xFF == ord("q"):
                read = False
            if not ok:
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass

    reader.stop()
    if server:
        server.close()
    else:
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()