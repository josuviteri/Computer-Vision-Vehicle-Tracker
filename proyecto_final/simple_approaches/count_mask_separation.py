import cv2
import numpy as np

# 1. Open the video stream
video_path = "media/proyecto/Abajo_Corto.mp4" 

try:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Error: Could not open video file")
        
    print("Video loaded successfully")
    
except Exception as e:
    print(f"Error: {e}")
    print("Script terminated due to loading error")
    exit(1)

# 2. Create background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# 3. ROI
roi_points = np.array([[360, 420], [540, 390], [800, 570], [470, 680]])

# Create mask for ROI
mask = np.zeros((720, 1280), dtype=np.uint8) 
cv2.fillPoly(mask, [roi_points], 255)


distance_threshold = 55

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance 
    frame = cv2.resize(frame, (1280, 720))

    # 4. Apply background subtraction
    fgMask = backSub.apply(frame)

    # 5. Apply ROI mask
    fgMask = cv2.bitwise_and(fgMask, fgMask, mask=mask)

    # Remove shadows (MOG2 marks shadows with value 127)
    fgMask[fgMask == 127] = 0

    # 6. Clean up the mask: blur + morphological ops to reduce noise and fill holes
    fgMask = cv2.GaussianBlur(fgMask, (9, 9), 0)
    # increase if noisy 7 7 or 9 9
    # decrease to preserve more details 3 3

    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY) 
    # lowering: more permisive mask
    # raising, stricter mask

    # Morphological opening to remove small noise, closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional dilation to connect nearby components (tune iterations as needed)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)

    # 7. Find contours
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    car_count = 0

    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 70 and area < 700: 
            x, y, w, h = cv2.boundingRect(cnt)

            cx, cy = x + w // 2, y + h // 2 #centroid

            too_close = False

            for (px, py) in boxes:
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < distance_threshold:  
                    too_close = True
                    break

            if too_close: 
                continue

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            boxes.append((cx, cy))
            car_count += 1

    # Draw ROI and count
    cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)
    cv2.putText(frame, f"Cars: {car_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Mask", fgMask)
    cv2.imshow("Car Counter", frame)


    if cv2.waitKey(30) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

#TODO adjust the mask in order to correctly see the windshields of the cars