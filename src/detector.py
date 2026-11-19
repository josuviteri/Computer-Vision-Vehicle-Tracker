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

    def detect(self, frame) -> list[Detection]:
        detections = []
        fg = self.backSub.apply(frame)
        fgMask = cv2.bitwise_and(fg, self.mask)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                detections.append(Detection((x, y, w, h), area))
        self.frames_seen += 1 # al llegar a 100 frames se termina el warmup y se empieza a contar coches
        if self.frames_seen >= 100:
            return detections
        return []  # warmup: sin detecciones