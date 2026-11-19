import cv2
import numpy as np

class Detection():
    def __init__(self, bbox, area):
        self.bbox = bbox
        self.area = area

class Detector():
    def __init__(self, frame_shape, roi, min_area_frac, max_area_frac):
        self.frame_shape = frame_shape
        self.roi = roi
        self.mask = cv2.fillPoly(np.zeros(frame_shape[:2], dtype=np.uint8), [roi], 255)
        self.min_area = self.mask.shape[0] * self.mask.shape[1] * min_area_frac
        self.max_area = self.mask.shape[0] * self.mask.shape[1] * max_area_frac
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.frames_seen = 0

    def detect(self, frame, extract=True) -> list[Detection] | None:
        # MOG2 se actualiza en TODOS los frames, los contornos solo si extract=True
        detections = []
        fg = self.backSub.apply(frame)
        self.frames_seen += 1
        if not extract:
            return None
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # cierre grande: une los trozos de una misma deteccion
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        fgMask = cv2.bitwise_and(fg, self.mask)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                detections.append(Detection((x, y, w, h), area))
        # al llegar a 100 frames se termina el warmup y se empieza a contar
        if self.frames_seen >= 100:
            return detections
        return []  # warmup: sin detecciones