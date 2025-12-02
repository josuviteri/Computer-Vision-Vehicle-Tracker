import cv2
import numpy as np
from collections import OrderedDict
import math

class CentroidTracker:
    def __init__(self, maxDisappeared=30, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.centroids = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, bbox):
        x, y, w, h = bbox
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        self.objects[self.nextObjectID] = (x, y, w, h)
        self.centroids[self.nextObjectID] = (cx, cy)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
            del self.centroids[objectID]
            del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            ids = list(self.disappeared.keys())
            for objectID in ids:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = []
        for (x, y, w, h) in rects:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids.append((cx, cy))

        if len(self.centroids) == 0:
            for rect in rects:
                self.register(rect)
            return self.objects

        objectIDs = list(self.centroids.keys())
        objectCentroids = list(self.centroids.values())

        D = np.zeros((len(objectCentroids), len(input_centroids)), dtype="float")
        for i, (ocx, ocy) in enumerate(objectCentroids):
            for j, (icx, icy) in enumerate(input_centroids):
                D[i, j] = math.hypot(ocx - icx, ocy - icy)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = rects[col]
            self.centroids[objectID] = input_centroids[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])) - usedRows
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        unusedCols = set(range(0, D.shape[1])) - usedCols
        for col in unusedCols:
            self.register(rects[col])

        return self.objects

# MAIN PROCESS

video_path = "media/proyecto/480.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

roi_points = np.array([[360, 420], [540, 390], [800, 570], [470, 680]])
mask = np.zeros((720, 1280), dtype=np.uint8)
cv2.fillPoly(mask, [roi_points], 255)

ct = CentroidTracker(maxDisappeared=20, maxDistance=60)

min_area = 200
max_area = 700

# 1 CREATE TRAJECTORY STORAGE

trajectory = {}
max_trail_length = 40

going_down = 0
going_right = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))

    fgMask = backSub.apply(frame)
    fgMask = cv2.bitwise_and(fgMask, fgMask, mask=mask)
    fgMask[fgMask == 127] = 0

    fgMask = cv2.GaussianBlur(fgMask, (7, 7), 0)
    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgMask)
    rects = []

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            continue

        aspect = w / float(h) if h != 0 else 0
        if aspect < 0.3 or aspect > 4.0:
            pass

        rects.append((x, y, w, h))

    objects = ct.update(rects)

    # 2. UPDATE TRAJECTORY HISTORY

    active_ids = set(objects.keys())

    # Remove old trajectories
    dead_ids = [oid for oid in trajectory.keys() if oid not in active_ids]
    for oid in dead_ids:
        del trajectory[oid]

    # Add/update active trajectories
    for objectID, bbox in objects.items():
        x, y, w, h = bbox
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        if objectID not in trajectory:
            trajectory[objectID] = []

        trajectory[objectID].append((cx, cy))

        if len(trajectory[objectID]) > max_trail_length:
            trajectory[objectID].pop(0)

    # DRAW BBOXES

    for objectID, bbox in objects.items():
        x, y, w, h = bbox
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objectID}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    # 3. DRAW TRAJECTORY LINES

    for objectID, pts in trajectory.items():
        if len(pts) < 2:
            continue
        start = pts[0]
        end = len(pts) - 1

        # Compute angle (degrees) between the vector from first to last point
        dy = pts[end][1] - pts[0][1]
        dx = pts[end][0] - pts[0][0]
        angle = math.degrees(math.atan2(dy, dx))

        if objectID == 24:
                print(f"Object {objectID} angle: {angle}")

        if 26 <= angle <= 38:
            going_down += 1
        elif -10 < angle < -30:
            going_right += 1

        for i in range(1, len(pts)):
            cv2.line(frame,
                     pts[i - 1],
                     pts[i],
                     (0, 150, 255), 2)

    # UI

    cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)
    cv2.putText(frame, f"Tracked: {len(objects)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.putText(frame, f"Going Down: {going_down}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.putText(frame, f"Going Right: {going_right }", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Mask", fgMask)
    cv2.imshow("Hybrid Tracker", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
