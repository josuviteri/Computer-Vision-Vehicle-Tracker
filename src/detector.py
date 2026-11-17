import cv2
import numpy as np

class Detector():
    def __init__(self, frame_shape, roi, min_area_frac, max_area_frac):
        self.frame_shape = frame_shape
        self.roi = roi
        self.detections = None
        self.mask = cv2.fillPoly(np.zeros(frame_shape[:2], dtype=np.uint8), [roi], 255)
        self.min_area = self.mask.shape[0] * self.mask.shape[1] * min_area_frac
        self.max_area = self.mask.shape[0] * self.mask.shape[1] * max_area_frac
                
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.frames_seen = 0

    def detect(self, frame) -> list[tuple[int,int,int,int]]:
        detections = []
        self.detections = detections

        fg = self.backSub.apply(frame)
        fgMask = cv2.bitwise_and(fg, self.mask)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if self.min_area <= area <= self.max_area:
                detections.append((x, y, w, h))
        self.frames_seen += 1
        return detections