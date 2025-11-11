import cv2
import numpy as np

video_path = 'media/proyecto/Abajo_Corto.mp4'

try:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Error: Could not open video file")
        
    print("Video loaded successfully")
    
except Exception as e:
    print(f"Error: {e}")
    print("Script terminated due to loading error")
    exit(1)


backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

up_count, down_count = 0, 0
track_points = []

line_up = 250
line_down = 400

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi_points = np.array([[190,250],[300,240],[470,370],[280,430]])

    # ROI mask
    roi_mask = np.zeros_like(frame[:,:,0])
    cv2.fillPoly(roi_mask, [roi_points], 255)
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0,255,255), thickness=2)

    #apply mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    fgmask = backSub.apply(masked_frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_points = []


    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > 50 and area < 1000):
            x,y,w,h = cv2.boundingRect(cnt)
            cx, cy = int(x+w/2), int(y+h/2)
            new_points.append((cx,cy))
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Comparar viejos y nuevos centroides para contar
    """
    for (cx,cy) in new_points:
        for (pcx,pcy) in track_points:
            if abs(cx-pcx) < 50 and abs(cy-pcy) < 50:
                if pcy >= line_down and cy < line_down:
                    up_count += 1
                elif pcy <= line_up and cy > line_up:
                    down_count += 1
    """
    track_points = new_points

    cv2.line(frame, (0,line_up), (frame.shape[1],line_up), (255,0,0), 2)
    cv2.line(frame, (0,line_down), (frame.shape[1],line_down), (0,0,255), 2)
    cv2.putText(frame, f'Up: {up_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f'Down: {down_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
