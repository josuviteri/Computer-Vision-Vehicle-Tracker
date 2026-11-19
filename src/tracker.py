from detector import Detection
import cv2


def iou(bbox1, bbox2) -> float:
    """IoU entre dos bboxes (x, y, w, h)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = ix * iy
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union if union > 0 else 0.0


class Track():
    def __init__(self, id: int, frame, detection: Detection):
        self.id = id
        self.state = tuple(int(v) for v in detection.bbox)
        self.history = [self.state]
        self.missed = 0  # frames consecutivos en los que KCF pierde el objeto
        # OpenCV 5 ya no incluye KCF; TrackerMIL tiene la misma API (init/update)
        self.tracker = cv2.TrackerMIL_create()
        self.tracker.init(frame, self.state)

    def update(self, frame) -> bool:
        ok, bbox = self.tracker.update(frame)
        if ok:
            self.state = tuple(int(v) for v in bbox)
            self.history.append(self.state)
            self.missed = 0
        else:
            self.missed += 1
        return ok


def overlaps(detection: Detection, tracks, threshold: float = 0.3) -> bool:
    """True si la deteccion ya corresponde a algun track existente."""
    return any(iou(detection.bbox, t.state) > threshold for t in tracks)
