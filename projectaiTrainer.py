import cv2 as cv
import numpy as np
import time
import poseModule as pm

cap= cv.VideoCapture(0)
ptime = 0
ctime = 0
detector = pm.poseDetector()
count = 0
dir=0
 
while True:
    success,img= cap.read()
    img=detector.findPose(img,False)
    lmlist=detector.findPosition(img,draw=False)
    # print(lmlist)
    if len(lmlist) != 0:
        # angle = detector.findangle(img, 11, 13, 15, draw=True)  # Example for left shoulder, elbow, wrist
        angle =detector.findangle(img, 12, 14, 16, draw=True)  # Example for right shoulder, elbow, wrist
        per= np.interp(angle, (60, 140), (100, 0))
        print(f"Angle: {angle}, Percentage: {per}%")
        bar = np.interp(angle, (60, 140), (100, 400))
        color = (0, 255, 0)
        if per == 100:
            color = (0, 0, 255)
            if dir==0:
                count += 0.5
                dir=1
        if per == 0:
            color = (0, 0, 255)
            if dir==1:
                count += 0.5
                dir=0
        print(count)
        cv.rectangle(img, (50, 100), (85, 400), (color), 3)
        cv.rectangle(img, (50, int(bar)), (85, 400), (color), cv.FILLED)
        cv.putText(img, f'{int(per)}%', (10, 500), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv.putText(img, f'Angle: {int(angle)}', (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv.putText(img, f'Count: {int(count)}', (100, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv.putText(img, f'Direction: {"Up" if dir == 1 else "Down"}', (100, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv.putText(img, f'Count: {int(count)}', (1000, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    print(img.shape)
    cv.putText(img, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break