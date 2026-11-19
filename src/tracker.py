"""
Sistema de tracking de vehiculos.

Detector (sustraccion de fondo): donde hay algo moviendose
KCF: donde esta ese coche
Kalman (velocidad constante): donde DEBERIA estar

Estados: TENTATIVE -> CONFIRMED (los que se cuentan) -> LOST -> despues se elimina
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from detector import Detection

TENTATIVE, CONFIRMED, LOST = "TENTATIVE", "CONFIRMED", "LOST"


def iou(a, b) -> float:
    # Interseccion sobre la union (IoU) entre dos bboxes (x, y, w, h).
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


class Track():
    def __init__(self, id: int, frame, detection: Detection, tracker_type: str = "KCF"):
        self.id = id
        self.tracker_type = tracker_type
        self.state = TENTATIVE
        self.hits = 1
        self.misses = 0
        self.dup_frames = 0
        self.age = 0
        self.bbox = tuple(int(v) for v in detection.bbox)
        self.history = [center(self.bbox)]
        self.anchor = self.history[0]  # ultima posicion "significativa"
        self.still_frames = 0

        cx, cy = center(self.bbox)
        kf = cv2.KalmanFilter(4, 2)  # estado (cx, cy, vx, vy), medida (cx, cy)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman = kf
        self.pred = (cx, cy)

        self._init_kcf(frame, self.bbox)

    # --- KCF / CSRT ---
    def _init_kcf(self, frame, bbox):
        if self.tracker_type == "CSRT":
            self.kcf = cv2.TrackerCSRT_create()
        else:
            self.kcf = cv2.TrackerKCF_create()
        self.kcf.init(frame, tuple(int(v) for v in bbox))

    # --- ciclo por frame ---
    def predict(self):
        p = self.kalman.predict()
        self.pred = (float(p[0, 0]), float(p[1, 0]))
        return self.pred

    def update_still(self, min_move):
        # Cuenta frames seguidos sin alejarse de la posicion ancla
        c = center(self.bbox)
        if np.hypot(c[0] - self.anchor[0], c[1] - self.anchor[1]) > min_move:
            self.anchor = c
            self.still_frames = 0
        else:
            self.still_frames += 1

    def predicted_bbox(self):
        _, _, w, h = self.bbox
        return (self.pred[0] - w / 2.0, self.pred[1] - h / 2.0, w, h)

    def update_kcf(self, frame):
        # KCF, y si falla Kalman
        self.age += 1
        ok, box = self.kcf.update(frame)
        if ok:
            box = tuple(int(v) for v in box)
            cx, cy = center(box)
            max_jump = max(box[2], box[3])
            if np.hypot(cx - self.pred[0], cy - self.pred[1]) < max_jump:
                self.kalman.correct(np.array([[cx], [cy]], np.float32))
                self.bbox = box
                self.history.append((cx, cy))
                return True
        self.misses += 1
        _, _, w, h = self.bbox
        x, y = self.pred[0] - w / 2.0, self.pred[1] - h / 2.0
        self.bbox = (int(x), int(y), w, h)
        self.history.append(self.pred)
        return False

    def match(self, frame, detection: Detection, min_hits: int):
        #Detector confirma el track: corrige Kalman y KCF
        cx, cy = center(detection.bbox)
        self.kalman.correct(np.array([[cx], [cy]], np.float32))
        self.bbox = tuple(int(v) for v in detection.bbox)
        self.history[-1] = (cx, cy)
        self._init_kcf(frame, self.bbox)
        self.misses = 0
        self.hits += 1
        if self.state == LOST or (self.state == TENTATIVE and self.hits >= min_hits):
            self.state = CONFIRMED


class TrackManager():
    def __init__(self, roi, min_hits=3, max_misses_lost=10, max_misses_delete=60,
                 max_assoc_cost=0.7, dup_iou=0.6, dup_frames=3, tracker_type="KCF",
                 max_still_frames=150, still_move=5):
        self.max_still_frames = max_still_frames  # frames quieto antes de eliminar
        self.still_move = still_move              # px minimos para considerar movimiento
        self.roi = np.asarray(roi, np.int32)
        self.min_hits = min_hits
        self.max_misses_lost = max_misses_lost
        self.max_misses_delete = max_misses_delete
        self.max_assoc_cost = max_assoc_cost
        self.dup_iou = dup_iou
        self.dup_frames = dup_frames
        self.tracker_type = tracker_type
        self.tracks: list[Track] = []
        self.next_id = 0
        self.total_count = 0  # vehiculos contados al confirmarse

    @property
    def confirmed(self):
        return [t for t in self.tracks if t.state == CONFIRMED]

    def step(self, frame, detections=None):
        #detections=None es un frame sin deteccion (solo KCF + Kalman)
        for t in self.tracks:
            t.predict()
            t.update_kcf(frame)

        if detections is not None:
            self._associate(frame, detections)

        self._lifecycle()
        self._merge_duplicates()

    def _associate(self, frame, detections):
        unmatched = set(range(len(detections)))
        if self.tracks and detections:
            cost = np.ones((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    cost[i, j] = 1.0 - iou(t.predicted_bbox(), d.bbox)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] <= self.max_assoc_cost:
                    self._matched(self.tracks[i], frame, detections[j])
                    unmatched.discard(j)

        for j in unmatched:
            d = detections[j]
            lost = self._find_lost(d)
            if lost is not None:  # redeteccion conserva el ID
                self._matched(lost, frame, d)
            else:
                self.tracks.append(Track(self.next_id, frame, d, self.tracker_type))
                self.next_id += 1

    def _matched(self, track, frame, detection):
        was_confirmed = track.state == CONFIRMED
        track.match(frame, detection, self.min_hits)
        if track.state == CONFIRMED and not was_confirmed and not getattr(track, "counted", False):
            track.counted = True
            self.total_count += 1

    def _find_lost(self, detection):
        cx, cy = center(detection.bbox)
        best, best_d = None, None
        for t in self.tracks:
            if t.state != LOST:
                continue
            d = np.hypot(cx - t.pred[0], cy - t.pred[1])
            if d < max(t.bbox[2], t.bbox[3]) * 1.5 and (best is None or d < best_d):
                best, best_d = t, d
        return best

    def _lifecycle(self):
        keep = []
        for t in self.tracks:
            if t.state == TENTATIVE and t.misses > 2:
                continue  # FP de pocos frames
            if t.misses > self.max_misses_delete:
                continue
            t.update_still(self.still_move)
            if t.still_frames > self.max_still_frames:
                continue  # lleva mucho tiempo sin moverse
            if t.state == CONFIRMED and t.misses > self.max_misses_lost:
                t.state = LOST
            # el ROI ya esta fuera
            if cv2.pointPolygonTest(self.roi, (float(t.pred[0]), float(t.pred[1])), True) < -20:
                continue
            keep.append(t)
        self.tracks = keep

    def _merge_duplicates(self):
        #Entre dos tracks muy solapados varios frames se conserva el de id mas antiguo
        remove = set()
        for i, a in enumerate(self.tracks):
            for b in self.tracks[i + 1:]:
                if a.id in remove or b.id in remove:
                    continue
                if iou(a.bbox, b.bbox) > self.dup_iou:
                    a.dup_frames += 1
                    b.dup_frames += 1
                    if a.dup_frames >= self.dup_frames:
                        old, new = (a, b) if a.id < b.id else (b, a)
                        old.hits = max(old.hits, new.hits)
                        if old.state != CONFIRMED and new.state == CONFIRMED:
                            old.state = CONFIRMED
                        remove.add(new.id)
                else:
                    a.dup_frames = b.dup_frames = 0
        if remove:
            self.tracks = [t for t in self.tracks if t.id not in remove]
