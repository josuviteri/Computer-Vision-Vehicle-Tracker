import cv2
import numpy as np

# 1. Open the video stream
video_url = "media/proyecto/Abajo_Corto.mp4" 
cap = cv2.VideoCapture(video_url)

# 2. Create background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# 3. ROI
roi_points = np.array([[330, 380], [480, 360], [790, 580], [470, 700]])

# Create mask for ROI
mask = np.zeros((720, 1280), dtype=np.uint8) 
cv2.fillPoly(mask, [roi_points], 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance (optional)
    frame = cv2.resize(frame, (1280, 720))

    # 4. Apply background subtraction
    fgMask = backSub.apply(frame)

    # 5. Apply ROI mask
    fgMask = cv2.bitwise_and(fgMask, fgMask, mask=mask)

    # 6. Clean up the mask
    fgMask = cv2.medianBlur(fgMask, 5)
    fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1]

    # 7. Find contours
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    car_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50 and area < 1000: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            car_count += 1

    # Draw ROI and count
    cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)
    cv2.putText(frame, f"Cars: {car_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Car Counter", frame)
    cv2.imshow("Mask", fgMask)

    if cv2.waitKey(30) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
