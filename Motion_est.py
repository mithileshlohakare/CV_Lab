import cv2
import numpy as np

cap = cv2.VideoCapture('op.mp4') 

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read first frame")
    exit()

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
step = 16  
scale = 3 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_color = cv2.applyColorMap(mag_norm.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.7, mag_color, 0.3, 0)


    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            fx, fy = flow[y, x]
            if np.hypot(fx, fy) > 1:  # draw only significant motion
                cv2.arrowedLine(overlay,
                                (x, y),
                                (int(x + scale*fx), int(y + scale*fy)),
                                color=(255, 255, 255),
                                thickness=1,
                                tipLength=0.3)

    cv2.imshow('Enhanced Motion Estimation', overlay)
    prev_gray = gray

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
