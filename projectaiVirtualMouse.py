import cv2 as cv
import time
import numpy as np
import hand_tracking_module as htm
import autopy

cap=cv.VideoCapture(0)
ptime=0
ctime=0
detector=htm.handDetector(maxHands=1)
fingers = [0, 0, 0, 0, 0]  # Initialize fingers state
frameR=100  # Frame Reduction
smoothening = 7  # Smoothing factor for mouse movement
plocX, plocY = 0, 0  # Previous location of the mouse
clocX, clocY = 0, 0  # Current location of the mouse
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmLis,bbox=detector.findPosition(img)

    # get tip of index finger and middle finger
    
    if len(lmLis) != 0:
        x1, y1 = lmLis[8][1], lmLis[8][2]
        x2, y2 = lmLis[12][1], lmLis[12][2]
        # print(f'Index Finger: {x1}, {y1}, Middle Finger: {x2}, {y2}') 
    # check which finger is up
        fingers = detector.fingersUp()
        # print(fingers)
        cv.rectangle(img,(frameR,frameR), (img.shape[1] - frameR, img.shape[0] - frameR), (255, 0, 255), 2)
    # Index finger moving mode
    if fingers[1] ==1 and fingers[2] == 0:
        # Convert coordinates to screen coordinates
        screen_width, screen_height = autopy.screen.size()
        x3 = np.interp(x1, (frameR, img.shape[1]-frameR), (0, screen_width))
        y3 = np.interp(y1, (frameR, img.shape[0]-frameR), (0, screen_height))
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
          #
        # Smooth the movement
        autopy.mouse.move(screen_width - clocX,clocY )
        cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        plocX, plocY = clocX, clocY
    
    if fingers[1] ==1 and fingers[2] == 1:
        length, img, lineInfo = detector.findDistance(8, 12, img)
        print(length)
        if length < 110:
            cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv.FILLED)
            autopy.mouse.click()

    # Frame rate calculation
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,f'FPS: {int(fps)}',(10,70),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)  
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 