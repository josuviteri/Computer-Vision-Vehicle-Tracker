import cv2
import numpy as np

# 1. Load video
video_path = "media/proyecto/480.mp4"
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

distance_threshold = 50
car_total_count = 0

cars_moving_y = 0
cars_moving_x = 0

# KCF trackers list
trackers = []
tracked_ids = []  # to prevent recounting

frame_id = 0

trajectory = {}
max_trail_length = 40

going_down = 0
going_right = 0

# Per-object last computed angles (degrees)
angles = {}



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    frame_id += 1

    # Update all trackers first
    new_trackers = []
    new_tracked_ids = []
    new_trajectory = {}

    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            new_trackers.append(tracker)
            car_id = tracked_ids[i]
            new_tracked_ids.append(car_id)

            x, y, w, h = map(int, bbox)
            cx, cy = x + w//2, y + h//2

            # Track centroid trail
            if car_id not in trajectory:
                trajectory[car_id] = []
            trajectory[car_id].append((cx, cy))

            # Limit trail length to 40 frames
            if len(trajectory[car_id]) > max_trail_length:
                trajectory[car_id].pop(0)

            # Draw trail (all stored points)
            for j in range(1, len(trajectory[car_id])):
                cv2.line(frame, trajectory[car_id][j-1], trajectory[car_id][j], (0, 255, 255), 2)
                cv2.circle(frame, trajectory[car_id][j], 2, (0, 255, 255), -1)

            # Compute angle every 5 frames when at least 2 points exist
            angle_deg = angles.get(car_id, 0.0)
            if frame_id % 5 == 0 and len(trajectory[car_id]) > 2:
                pts = trajectory[car_id]
                k = min(3, len(pts))
                # average of first k points
                x1 = sum(p[0] for p in pts[:k]) / k
                y1 = sum(p[1] for p in pts[:k]) / k
                # average of last k points
                x2 = sum(p[0] for p in pts[-k:]) / k
                y2 = sum(p[1] for p in pts[-k:]) / k

                dx = x2 - x1
                dy = y2 - y1
                # In image coords y increases downwards — invert dy to get math convention
                angle_rad = np.arctan2(-dy, dx)
                angle_deg = (np.degrees(angle_rad) + 360.0) % 360.0
                angles[car_id] = angle_deg

            integ_angle = int(angle_deg)
            var_angle = ""
            color_ang = 0

            #TODO ARREGLAR LOGICA DE ANGULOS

            #filtro para y
            if  integ_angle < 7 and integ_angle > 35:
                cars_moving_y += 1
                color_ang = 128
                var_angle = "y"

            #filtro para x
            elif integ_angle > 8 and integ_angle < 35:
                cars_moving_x += 1
                color_ang = 255
                var_angle = "x"

            # Draw bounding box and ID + angle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,color_ang,0), 2)
            cv2.putText(frame, f"ID {car_id} - {integ_angle}- {var_angle}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,color_ang,0), 2)


            for i in range(1, len(trajectory[car_id])):
                cv2.line(frame,
                        trajectory[car_id][i - 1],
                        trajectory[car_id][i],
                        (0, 255, color_ang), 2)

    # Keep the updated lists
    trackers = new_trackers
    tracked_ids = new_tracked_ids

    current_centers = []
    for tid in tracked_ids:
        if tid in trajectory and len(trajectory[tid]) > 0:
            current_centers.append(trajectory[tid][-1])

    # Detect every N frames (less frequent = better performance)

    detection_interval = 10
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
                for (tcx, tcy) in current_centers:
                    dist = np.linalg.norm([cx - tcx, cy - tcy])
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

    cv2.putText(frame, f"Cars Moving in the Y axis: {cars_moving_y}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    cv2.putText(frame, f"Cars Moving in the X axis: {cars_moving_x}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(20) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
