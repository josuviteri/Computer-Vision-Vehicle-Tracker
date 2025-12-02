import cv2
import numpy as np

# 1. Load video
video_path = "media/proyecto/Abajo_Corto.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error loading video")
    exit()

print("Video loaded successfully")

# 2. Background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# 3. ROI polygon & mask
roi_points = np.array([[360, 420], [540, 390], [800, 570], [470, 680]])
mask = np.zeros((720, 1280), dtype=np.uint8)
cv2.fillPoly(mask, [roi_points], 255)

distance_threshold = 55
car_total_count = 0

# KCF trackers list
trackers = []
tracked_ids = []  # to prevent recounting

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    frame_id += 1

    # Update all trackers first
    new_trackers = []
    new_tracked_ids = []

    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            new_trackers.append(tracker)
            new_tracked_ids.append(tracked_ids[i])
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

    trackers = new_trackers
    tracked_ids = new_tracked_ids


    # Detect every N frames (less frequent = better performance)

    detection_interval = 2
    if frame_id % detection_interval == 0:

        fgMask = backSub.apply(frame)
        fgMask = cv2.bitwise_and(fgMask, fgMask, mask=mask)
        fgMask[fgMask == 127] = 0
        fgMask = cv2.GaussianBlur(fgMask, (9, 9), 0)
        _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fgMask = cv2.dilate(fgMask, kernel, iterations=1)

        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 120 < area < 2000:  # vehicle-sized movement
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2

                # Check distance from existing trackers
                too_close = False
                for tracker in trackers:
                    success, tb = tracker.update(frame)
                    if success:
                        tx, ty, tw, th = map(int, tb)
                        tcx, tcy = tx + tw//2, ty + th//2
                        dist = np.linalg.norm([cx-tcx, cy-tcy])
                        if dist < distance_threshold:
                            too_close = True
                            break
                if too_close:
                    continue

                # Create a new KCF tracker for the object
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)

                tracked_ids.append(car_total_count)  # unique ID

                car_total_count += 1  # count only ONCE

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Draw ROI
    cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)

    # Display total car count
    cv2.putText(frame, f"Total Cars: {car_total_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(20) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
