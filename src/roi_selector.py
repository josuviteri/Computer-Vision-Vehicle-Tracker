from typing import Optional
import cv2
import numpy as np


Point = tuple[int, int]
Rectangle = np.ndarray  # int32 array of shape (4, 2): the four clicked corners

class RoiSelector:
    """Select a quadrilateral ROI by clicking its four corners."""

    def __init__(self, frame):
        self.frame = frame
        self.points: list[Point] = []
        self.roi: Optional[Rectangle] = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(self.points) >= 4:
            return

        self.points.append((x, y))
        print(f"Point {len(self.points)}/4: {(x, y)}")

        if len(self.points) == 4:
            self.roi = np.array(self.points, dtype=np.int32)

    def select(self, window_name: str = "ROI Selection") -> Optional[Rectangle]:
        """Display the frame until four corners are selected or selection is cancelled."""
        if self.frame is None:
            raise ValueError("A frame is required to select an ROI")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while self.roi is None:
            preview = self.frame.copy()
            for point in self.points:
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
            if len(self.points) > 1:
                cv2.polylines(
                    preview,
                    [np.array(self.points, dtype=np.int32)],
                    False,
                    (0, 255, 255),
                    2,
                )
            text = "Select 4 corners of the ROI by clicking its corners or press 'q' to cancel"
            text_position = (10, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, text_thickness
            )

            overlay = preview.copy()
            cv2.rectangle(
                overlay,
                (text_position[0] - 8, text_position[1] - text_height - 8),
                (
                    text_position[0] + text_width + 8,
                    text_position[1] + baseline + 8,
                ),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.addWeighted(overlay, 0.65, preview, 0.35, 0, preview)
            cv2.putText(
                preview,
                text,
                text_position,
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("ROI selection cancelled")
                self.roi = None
                break

        cv2.destroyWindow(window_name)
        return self.roi
